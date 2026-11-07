//! SemiDynark, the semi-dynamic zkSNARK (Section "Semi-Dynamic zkSNARK" of
//! the paper): Groth16's `G`, `P` and `V`, plus preprocessing `K` on a base
//! witness and updates `U` relative to that base witness.
use crate::{
    decomposition::{
        cross::cross_term,
        projection::{precompute_projection, projection_term},
        QuotientBasis,
    },
    groth16::{
        self,
        qap::{lagrange_of_dense_vec, lagrange_of_sparse_vec, matrix_columns},
    },
    PreparedVerifyingKey, Proof, ProvingKey, VerifyingKey,
};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::UniformRand;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::{ConstraintMatrices, Result as R1CSResult, SynthesisError};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{marker::PhantomData, rand::Rng, vec::Vec};
use rayon::prelude::*;

/// The semi-dynamic zkSNARK over the pairing `E`.
pub struct SemiDynark<E: Pairing> {
    _p: PhantomData<E>,
}

/// The updating key `uk`: the prover key, the quotient basis derived from it,
/// and the R1CS matrices by columns. It contains no trapdoor.
#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct UpdatingKey<E: Pairing> {
    /// The Groth16 prover key.
    pub pk: ProvingKey<E>,
    /// The quotient basis.
    pub basis: QuotientBasis<E>,
    /// Matrix `A` by columns: for each variable `j`, the pairs `(row, A[row][j])`.
    pub a_matrix_cols: Vec<Vec<(usize, E::ScalarField)>>,
    /// Matrix `B` by columns: for each variable `j`, the pairs `(row, B[row][j])`.
    pub b_matrix_cols: Vec<Vec<(usize, E::ScalarField)>>,
}

/// What an update starts from: the base proof, its randomness, and
/// (after [`SemiDynark::preprocess`]) the preprocessed data `D`, the projection
/// quotients of the base witness.
#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct SemiDynarkData<E: Pairing> {
    /// The evaluation domain.
    pub domain: GeneralEvaluationDomain<E::ScalarField>,
    /// Number of instance variables, including the constant one.
    pub num_instance: usize,
    /// The blinding factor `r` of the base proof.
    pub r: E::ScalarField,
    /// The blinding factor `s` of the base proof.
    pub s: E::ScalarField,
    /// The projection data for the Lagrange coefficients of `A * z` (the
    /// paper's projection quotients), precomputed by [`SemiDynark::preprocess`].
    pub q_a: Vec<E::G1>,
    /// The projection data for the Lagrange coefficients of `B * z`.
    pub q_b: Vec<E::G1>,
    /// The `A` element in `G1`.
    pub a_g1: E::G1Affine,
    /// The `B` element in `G1`.
    pub b_g1: E::G1Affine,
    /// The `B` element in `G2`.
    pub b_g2: E::G2Affine,
    /// The `C` element in `G1`.
    pub c_g1: E::G1Affine,
}

impl<E: Pairing> SemiDynark<E> {
    /// `G`: see [`groth16::setup()`].
    #[allow(clippy::type_complexity)]
    pub fn setup<R: Rng>(
        r1cs: ConstraintMatrices<E::ScalarField>,
        rng: &mut R,
    ) -> Result<
        (
            ProvingKey<E>,
            VerifyingKey<E>,
            GeneralEvaluationDomain<E::ScalarField>,
        ),
        SynthesisError,
    > {
        groth16::setup(r1cs, rng)
    }

    /// Derives the [`UpdatingKey`] from the prover key.
    pub fn generate_updating_key(
        r1cs: &ConstraintMatrices<E::ScalarField>,
        domain: &GeneralEvaluationDomain<E::ScalarField>,
        pk: &ProvingKey<E>,
    ) -> R1CSResult<UpdatingKey<E>> {
        let basis = QuotientBasis::new(domain, &pk.h_query);

        let num_variables = r1cs.num_instance_variables + r1cs.num_witness_variables;
        let a_matrix_cols = matrix_columns(&r1cs.a, num_variables);
        let b_matrix_cols = matrix_columns(&r1cs.b, num_variables);

        Ok(UpdatingKey {
            pk: pk.clone(),
            basis,
            a_matrix_cols,
            b_matrix_cols,
        })
    }

    /// `P`: see [`groth16::prove()`]. The returned data has empty projection
    /// quotients; run [`SemiDynark::preprocess`] on it before the first
    /// [`SemiDynark::update`].
    pub fn prove<R: Rng>(
        pk: &ProvingKey<E>,
        r1cs: &ConstraintMatrices<E::ScalarField>,
        instance: &[E::ScalarField],
        witness: &[E::ScalarField],
        rng: &mut R,
    ) -> Result<(Proof<E>, SemiDynarkData<E>), SynthesisError> {
        let (proof, randomness) = groth16::prove(pk, r1cs, instance, witness, rng)?;
        let domain = GeneralEvaluationDomain::<E::ScalarField>::new(r1cs.num_constraints)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let data = SemiDynarkData {
            domain,
            num_instance: instance.len(),
            a_g1: proof.a,
            b_g1: randomness.b_g1,
            b_g2: proof.b,
            c_g1: proof.c,
            r: randomness.r,
            s: randomness.s,
            q_a: Default::default(),
            q_b: Default::default(),
        };
        Ok((proof, data))
    }

    /// `V`: see [`groth16::verify()`].
    pub fn verify(
        pvk: &PreparedVerifyingKey<E>,
        proof: &Proof<E>,
        instance: &[E::ScalarField],
    ) -> R1CSResult<bool> {
        groth16::verify(pvk, proof, instance)
    }

    /// `K`: precomputes the projection data of the base witness into `data`.
    pub fn preprocess(
        uk: &UpdatingKey<E>,
        r1cs: &ConstraintMatrices<E::ScalarField>,
        instance: &[E::ScalarField],
        witness: &[E::ScalarField],
        data: &mut SemiDynarkData<E>,
    ) -> Result<(), SynthesisError> {
        let (a, b, _c) = lagrange_of_dense_vec(r1cs, &[instance, witness].concat())?;
        let (q_a, q_b) = precompute_projection(&uk.basis, data.domain, &a, &b);
        data.q_a = q_a;
        data.q_b = q_b;
        Ok(())
    }

    /// `U`: a proof for the base witness plus a sparse change, given as
    /// `(variable index, delta)` pairs. Indices are into the full assignment,
    /// so witness indices start at `num_instance`.
    ///
    /// The new proof uses fresh blinding `(r', s')` drawn from `rng`. With `dz`
    /// the change to
    /// the assignment and `dw` its witness part:
    ///
    /// ```text
    /// A' = A + (r' - r) * delta + sum_j dz_j * u_g1[j]
    /// B' = B + (s' - s) * delta + sum_j dz_j * v_g1[j]   (and likewise in G2)
    /// C' = C + s' * A' - s * A + r' * B' - r * B - (r' * s' - r * s) * delta
    ///        + sum_j dw_j * t_p_g1[j] + projection terms + cross term
    /// ```
    pub fn update<R: Rng>(
        uk: &UpdatingKey<E>,
        instance_update: &[(usize, E::ScalarField)],
        witness_update: &[(usize, E::ScalarField)],
        data: &SemiDynarkData<E>,
        rng: &mut R,
    ) -> Result<Proof<E>, SynthesisError> {
        let (u_g1, v_g1, v_g2, t_p_g1) = (
            &uk.pk.a_query,
            &uk.pk.b_g1_query,
            &uk.pk.b_g2_query,
            &uk.pk.l_query,
        );

        let update = [instance_update, witness_update].concat();
        let r_new = E::ScalarField::rand(rng);
        let s_new = E::ScalarField::rand(rng);

        let delta_g1 = uk.pk.delta_g1;
        let delta_g2 = uk.pk.vk.delta_g2;

        let mut update_c_g1_linear_combination: E::G1 = witness_update
            .par_iter()
            .map(|(i, update_i)| t_p_g1[*i - data.num_instance].into_group() * update_i)
            .sum();

        let update_a_linear_combine: E::G1 = update
            .par_iter()
            .map(|(i, update_i)| u_g1[*i].into_group() * update_i)
            .sum();
        let a_g1_new =
            data.a_g1.into_group() + delta_g1 * (r_new - data.r) + update_a_linear_combine;

        let update_b_g1_linear_combine: E::G1 = update
            .par_iter()
            .map(|(i, update_i)| v_g1[*i].into_group() * update_i)
            .sum();
        let b_g1_new =
            data.b_g1.into_group() + delta_g1 * (s_new - data.s) + update_b_g1_linear_combine;

        let update_b_g2_linear_combine: E::G2 = update
            .par_iter()
            .map(|(i, update_i)| v_g2[*i].into_group() * update_i)
            .sum();
        let b_g2_new =
            data.b_g2.into_group() + delta_g2 * (s_new - data.s) + update_b_g2_linear_combine;

        update_c_g1_linear_combination += a_g1_new * &s_new - data.a_g1 * &data.s
            + b_g1_new * &r_new
            - data.b_g1 * &data.r
            - delta_g1 * (r_new * &s_new - data.r * &data.s);

        let (a_lagrange_sparse, b_lagrange_sparse) =
            lagrange_of_sparse_vec(&uk.a_matrix_cols, &uk.b_matrix_cols, &update)?;

        let c_project = projection_term::<E>(&data.q_b, &a_lagrange_sparse)
            + projection_term::<E>(&data.q_a, &b_lagrange_sparse);

        let c_cross = cross_term(&uk.basis, &a_lagrange_sparse, &b_lagrange_sparse);

        let c_new = data.c_g1.into_group() + update_c_g1_linear_combination + c_project + c_cross;

        Ok(Proof {
            a: a_g1_new.into_affine(),
            b: b_g2_new.into_affine(),
            c: c_new.into_affine(),
        })
    }
}
