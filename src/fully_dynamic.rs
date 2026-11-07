//! FullyDynark, the fully-dynamic zkSNARK (Section "Fully-Dynamic Groth16" of
//! the paper): updates apply to the current witness rather than to a fixed
//! base, and the preprocessed data is rebuilt in the background so that its
//! cost is de-amortized across updates.
//!
//! [`FullyDynark`] keeps two [`FullyCheckpoint`]s. The active one answers
//! updates, while the other recomputes its projection data step by step
//! with a [`StepwiseProjection`]. When the rebuild finishes, the two swap roles.
use crate::{
    decomposition::{
        cross::cross_term,
        projection::{precompute_projection, projection_term, StepwiseProjection},
    },
    groth16::{
        prover::{blind, raw_proof},
        qap::{lagrange_of_dense_vec, lagrange_of_sparse_vec},
    },
    Proof, UpdatingKey,
};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::{UniformRand, Zero};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::{ConstraintMatrices, SynthesisError};
use ark_std::{rand::Rng, vec::Vec};
use rayon::prelude::*;

/// Pending sparse changes to a length-`n` vector, accumulated by index and
/// cleared in `O(1)` by bumping a timestamp.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseRecord<E: Pairing> {
    /// Current Timestamp
    cur_tim: usize,
    /// Iteration for each slot
    tims: Vec<usize>,
    /// Value maintained
    vals: Vec<E::ScalarField>,
    /// useful indices
    u_ids: Vec<usize>,
}

impl<E: Pairing> SparseRecord<E> {
    /// Define a new SparseRecord
    pub fn new(n: usize) -> Self {
        let cur_tim: usize = 1;
        let tims: Vec<usize> = vec![0; n];
        let vals: Vec<E::ScalarField> = vec![E::ScalarField::zero(); n];
        let u_ids: Vec<usize> = Vec::with_capacity(n);
        Self {
            cur_tim,
            tims,
            vals,
            u_ids,
        }
    }
    /// Start a new iterate version
    pub fn clr(&mut self) {
        self.cur_tim += 1;
        self.u_ids.clear();
    }
    /// Update on specific slot
    pub fn ins(&mut self, id: usize, vl: E::ScalarField) {
        if self.tims[id] != self.cur_tim {
            self.tims[id] = self.cur_tim;
            self.vals[id] = vl;
            self.u_ids.push(id);
        } else {
            self.vals[id] += vl;
        }
    }
    /// Check the i the bit
    pub fn get(&self, id: usize) -> E::ScalarField {
        if self.tims[id] == self.cur_tim {
            return self.vals[id];
        }
        E::ScalarField::zero()
    }
    /// Parse all the ids
    pub fn parse(&mut self) -> Vec<(usize, E::ScalarField)> {
        self.u_ids.sort();
        self.u_ids
            .par_iter()
            .map(|id| (*id, self.vals[*id]))
            .collect::<Vec<(usize, E::ScalarField)>>()
    }
}

/// A proof for the current witness, kept without blinding, together with the
/// preprocessed data it is updated against.
#[derive(Clone, Debug, PartialEq)]
pub struct FullyCheckpoint<E: Pairing> {
    /// Domain size.
    n: usize,
    /// Number of instance variables, including the constant one.
    num_instance: usize,
    /// The Lagrange coefficients of `A * z` that the projection data was (or
    /// is being) computed for.
    a: Vec<E::ScalarField>,
    /// Likewise for `B * z`.
    b: Vec<E::ScalarField>,
    /// The projection data of `a`.
    q_a: Vec<E::G1>,
    /// The projection data of `b`.
    q_b: Vec<E::G1>,
    /// Changes to `a` since the projection data was last rebuilt.
    pending_a: SparseRecord<E>,
    /// Changes to `b` since the projection data was last rebuilt.
    pending_b: SparseRecord<E>,
    /// The in-progress rebuild of `q_a` and `q_b`.
    projection: StepwiseProjection<E>,
    /// The `A` element in `G1`.
    pub raw_a_g1: E::G1,
    /// The `B` element in `G1`.
    pub raw_b_g1: E::G1,
    /// The `B` element in `G2`.
    pub raw_b_g2: E::G2,
    /// The `C` element in `G1`.
    pub raw_c_g1: E::G1,
}

impl<E: Pairing> FullyCheckpoint<E> {
    /// A checkpoint for `(instance, witness)`, including its preprocessing.
    pub fn new(
        n: usize,
        uk: &UpdatingKey<E>,
        r1cs: &ConstraintMatrices<E::ScalarField>,
        instance: &[E::ScalarField],
        witness: &[E::ScalarField],
    ) -> Self {
        let domain = GeneralEvaluationDomain::<E::ScalarField>::new(n)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)
            .unwrap();
        let abc = lagrange_of_dense_vec(r1cs, &[instance, witness].concat()).unwrap();
        let (q_a, q_b) = precompute_projection(&uk.basis, domain, &abc.0, &abc.1);
        Self::from_parts(n, domain, uk, instance, witness, abc, q_a, q_b)
    }

    /// Like [`FullyCheckpoint::new`], reusing projection data that was already
    /// computed, e.g. by [`SemiDynark::preprocess`](crate::SemiDynark::preprocess).
    pub fn new_with_cached_quotient(
        n: usize,
        uk: &UpdatingKey<E>,
        r1cs: &ConstraintMatrices<E::ScalarField>,
        instance: &[E::ScalarField],
        witness: &[E::ScalarField],
        qa: &[E::G1],
        qb: &[E::G1],
    ) -> Self {
        let domain = GeneralEvaluationDomain::<E::ScalarField>::new(n)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)
            .unwrap();
        let abc = lagrange_of_dense_vec(r1cs, &[instance, witness].concat()).unwrap();
        Self::from_parts(
            n,
            domain,
            uk,
            instance,
            witness,
            abc,
            qa.to_vec(),
            qb.to_vec(),
        )
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn from_parts(
        n: usize,
        domain: GeneralEvaluationDomain<E::ScalarField>,
        uk: &UpdatingKey<E>,
        instance: &[E::ScalarField],
        witness: &[E::ScalarField],
        (a, b, c): (
            Vec<E::ScalarField>,
            Vec<E::ScalarField>,
            Vec<E::ScalarField>,
        ),
        q_a: Vec<E::G1>,
        q_b: Vec<E::G1>,
    ) -> Self {
        let (a_g1, b_g1, b_g2, c_g1) =
            raw_proof(&uk.pk, &domain, instance, witness, &a, &b, &c).unwrap();

        Self {
            n,
            num_instance: instance.len(),
            a,
            b,
            q_a,
            q_b,
            pending_a: SparseRecord::new(n),
            pending_b: SparseRecord::new(n),
            projection: StepwiseProjection::new(n),
            raw_a_g1: a_g1,
            raw_b_g1: b_g1,
            raw_b_g2: b_g2,
            raw_c_g1: c_g1,
        }
    }

    /// Records a change to the Lagrange coefficients of `A * z`.
    fn increment_a(&mut self, a_lag: &[(usize, E::ScalarField)]) {
        a_lag.iter().for_each(|(i, vl)| {
            self.pending_a.ins(*i, *vl);
        });
    }

    /// Records a change to the Lagrange coefficients of `B * z`.
    fn increment_b(&mut self, b_lag: &[(usize, E::ScalarField)]) {
        b_lag.iter().for_each(|(i, vl)| {
            self.pending_b.ins(*i, *vl);
        });
    }

    /// Folds the pending changes into `a` and `b` and restarts the rebuild of
    /// their projection data.
    fn prepare_for_reconstruct(&mut self) {
        self.projection.restart();
        self.pending_a
            .parse()
            .into_iter()
            .for_each(|(i, vl)| self.a[i] += vl);
        self.pending_b
            .parse()
            .into_iter()
            .for_each(|(i, vl)| self.b[i] += vl);
        self.pending_a.clr();
        self.pending_b.clr();
    }

    /// Runs up to `steps` steps of the rebuild. Returns whether the rebuild
    /// finished, and how many steps are left over.
    fn mv_forward(&mut self, uk: &UpdatingKey<E>, steps: usize) -> (bool, usize) {
        self.projection.step(
            &uk.basis,
            &self.a,
            &self.b,
            &mut self.q_a,
            &mut self.q_b,
            steps,
        )
    }

    /// Applies a sparse change to the raw proof. Indices are into the full
    /// assignment, so witness indices start at the number of instance
    /// variables.
    fn update_from(
        &mut self,
        uk: &UpdatingKey<E>,
        instance_update: &[(usize, E::ScalarField)],
        witness_update: &[(usize, E::ScalarField)],
    ) {
        let (u_g1, v_g1, v_g2, t_p_g1) = (
            &uk.pk.a_query,
            &uk.pk.b_g1_query,
            &uk.pk.b_g2_query,
            &uk.pk.l_query,
        );

        let update = [instance_update, witness_update].concat();
        let a_g1_new: E::G1 = update
            .par_iter()
            .map(|(i, update_i)| u_g1[*i].into_group() * update_i)
            .sum();

        let b_g1_new: E::G1 = update
            .par_iter()
            .map(|(i, update_i)| v_g1[*i].into_group() * update_i)
            .sum();

        let b_g2_new: E::G2 = update
            .par_iter()
            .map(|(i, update_i)| v_g2[*i].into_group() * update_i)
            .sum();

        let (a_lagrange_sparse, b_lagrange_sparse) =
            lagrange_of_sparse_vec(&uk.a_matrix_cols, &uk.b_matrix_cols, &update).unwrap();

        let c_project = projection_term::<E>(&self.q_b, &a_lagrange_sparse)
            + projection_term::<E>(&self.q_a, &b_lagrange_sparse);

        let num_instance = self.num_instance;
        let update_c_g1_linear_combination: E::G1 = witness_update
            .par_iter()
            .map(|(i, update_i)| t_p_g1[*i - num_instance].into_group() * update_i)
            .sum::<E::G1>();

        self.increment_a(&a_lagrange_sparse);
        let new_a_parse = self.pending_a.parse();
        let old_b_parse = self.pending_b.parse();
        self.increment_b(&b_lagrange_sparse);

        let c_new = update_c_g1_linear_combination
            + c_project
            + cross_term(&uk.basis, &new_a_parse, &b_lagrange_sparse)
            + cross_term(&uk.basis, &a_lagrange_sparse, &old_b_parse);

        self.raw_a_g1 += a_g1_new;
        self.raw_b_g1 += b_g1_new;
        self.raw_b_g2 += b_g2_new;
        self.raw_c_g1 += c_new;
    }

    /// Records a change without updating the raw proof. The rebuilding
    /// checkpoint uses this to keep its witness current.
    fn trivial_append(
        &mut self,
        uk: &UpdatingKey<E>,
        instance_update: &[(usize, E::ScalarField)],
        witness_update: &[(usize, E::ScalarField)],
    ) {
        let update = [instance_update, witness_update].concat();
        let (a_lagrange_sparse, b_lagrange_sparse) =
            lagrange_of_sparse_vec(&uk.a_matrix_cols, &uk.b_matrix_cols, &update).unwrap();
        self.increment_a(&a_lagrange_sparse);
        self.increment_b(&b_lagrange_sparse);
    }

    /// A proof for the current witness, with fresh blinding factors drawn
    /// from `rng`.
    pub fn blinded_proof<R: Rng>(&self, uk: &UpdatingKey<E>, rng: &mut R) -> Proof<E> {
        let r = E::ScalarField::rand(rng);
        let s = E::ScalarField::rand(rng);
        let raw = (self.raw_a_g1, self.raw_b_g1, self.raw_b_g2, self.raw_c_g1);
        let (a_g1, _, b_g2, c_g1) = blind(&uk.pk, raw, r, s);
        Proof {
            a: a_g1.into_affine(),
            b: b_g2.into_affine(),
            c: c_g1.into_affine(),
        }
    }

    /// Copies the raw proof from `other`.
    fn copy_proof_from(&mut self, other: &FullyCheckpoint<E>) {
        self.raw_a_g1 = other.raw_a_g1;
        self.raw_b_g1 = other.raw_b_g1;
        self.raw_b_g2 = other.raw_b_g2;
        self.raw_c_g1 = other.raw_c_g1;
    }
}

/// How far one call to [`FullyDynark::advance_rebuild`] got.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RebuildProgress {
    /// Rebuild steps spent.
    pub steps_used: usize,
    /// Rebuilds that finished, after which the checkpoints swapped roles.
    pub rebuilds_completed: usize,
}

/// The fully-dynamic prover, with de-amortized rebuilds.
#[derive(Clone, Debug)]
pub struct FullyDynark<E: Pairing> {
    active: FullyCheckpoint<E>,
    rebuilding: FullyCheckpoint<E>,
}

impl<E: Pairing> FullyDynark<E> {
    /// Starts from a checkpoint for the initial witness.
    pub fn new(checkpoint: FullyCheckpoint<E>) -> Self {
        let mut rebuilding = checkpoint.clone();
        rebuilding.prepare_for_reconstruct();
        Self {
            active: checkpoint,
            rebuilding,
        }
    }

    /// The checkpoint that currently answers updates.
    pub fn active(&self) -> &FullyCheckpoint<E> {
        &self.active
    }

    /// Spends up to `budget` steps on the background rebuild, stopping early
    /// after the second rebuild that finishes in this call.
    pub fn advance_rebuild(&mut self, uk: &UpdatingKey<E>, budget: usize) -> RebuildProgress {
        let mut progress = RebuildProgress::default();
        let mut rem_stp = budget;
        while rem_stp > 0 {
            let (finished, left) = self.rebuilding.mv_forward(uk, rem_stp);
            rem_stp = left;
            if finished {
                self.rebuilding.copy_proof_from(&self.active);
                core::mem::swap(&mut self.rebuilding, &mut self.active);
                self.rebuilding.prepare_for_reconstruct();
                progress.rebuilds_completed += 1;
                if progress.rebuilds_completed == 2 {
                    break;
                }
            }
        }
        progress.steps_used = budget - rem_stp;
        progress
    }

    /// Applies a sparse change to both checkpoints and returns a proof for
    /// the new witness, blinded with randomness from `rng`.
    pub fn apply_update<R: Rng>(
        &mut self,
        uk: &UpdatingKey<E>,
        instance_update: &[(usize, E::ScalarField)],
        witness_update: &[(usize, E::ScalarField)],
        rng: &mut R,
    ) -> Proof<E> {
        self.rebuilding
            .trivial_append(uk, instance_update, witness_update);
        self.active.update_from(uk, instance_update, witness_update);
        self.active.blinded_proof(uk, rng)
    }

    /// `U`: [`advance_rebuild`](Self::advance_rebuild) with `rebuild_budget`
    /// steps, then [`apply_update`](Self::apply_update).
    pub fn update<R: Rng>(
        &mut self,
        uk: &UpdatingKey<E>,
        instance_update: &[(usize, E::ScalarField)],
        witness_update: &[(usize, E::ScalarField)],
        rebuild_budget: usize,
        rng: &mut R,
    ) -> Proof<E> {
        self.advance_rebuild(uk, rebuild_budget);
        self.apply_update(uk, instance_update, witness_update, rng)
    }
}
