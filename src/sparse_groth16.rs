//! SparseGroth16: Groth16 with a prover whose cost depends only on the number
//! of nonzero entries in the assignment and in `A * z` and `B * z`, not on the
//! circuit size (paragraph "Sparse zkSNARKs" in the paper's appendix). Setup,
//! proofs and verification are those of Groth16; only the prover differs.
//!
//! The quotient commitment is the cross term of `A * z` and `B * z` with
//! themselves, `sum_{i,j} a_i * b_j * phi_ij`, so it is computed by
//! [`cross_term`], which uses sparse convolution for large inputs.
use crate::{
    decomposition::{cross::cross_term, QuotientBasis},
    groth16::{
        self,
        prover::{blind, Randomness},
        qap::{lagrange_of_sparse_vec, matrix_columns},
    },
    PreparedVerifyingKey, Proof, ProvingKey, VerifyingKey,
};
use ark_ec::{pairing::Pairing, CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField, UniformRand};
use ark_poly::GeneralEvaluationDomain;
use ark_relations::r1cs::{ConstraintMatrices, Result as R1CSResult, SynthesisError};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{marker::PhantomData, rand::Rng, vec::Vec};
use rayon::prelude::*;

/// Groth16 with the sparse prover over the pairing `E`.
pub struct SparseGroth16<E: Pairing> {
    _p: PhantomData<E>,
}

/// What [`SparseGroth16`] needs beyond the prover key: the quotient basis and
/// the matrices `A` and `B` by columns, derived once per circuit, with no
/// trapdoor.
#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct SparseGroth16Key<E: Pairing> {
    /// The quotient basis.
    pub basis: QuotientBasis<E>,
    /// Matrix `A` by columns, one per variable.
    pub a_matrix_cols: Vec<Vec<(usize, E::ScalarField)>>,
    /// Matrix `B` by columns, one per variable.
    pub b_matrix_cols: Vec<Vec<(usize, E::ScalarField)>>,
}

impl<E: Pairing> SparseGroth16Key<E> {
    /// Derives the key from the prover key.
    pub fn new(
        r1cs: &ConstraintMatrices<E::ScalarField>,
        domain: &GeneralEvaluationDomain<E::ScalarField>,
        pk: &ProvingKey<E>,
    ) -> Self {
        let num_variables = r1cs.num_instance_variables + r1cs.num_witness_variables;
        Self {
            basis: QuotientBasis::new(domain, &pk.h_query),
            a_matrix_cols: matrix_columns(&r1cs.a, num_variables),
            b_matrix_cols: matrix_columns(&r1cs.b, num_variables),
        }
    }
}

impl<E: Pairing> SparseGroth16<E> {
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

    /// `P` for a sparse assignment, given as `(variable index, value)` pairs
    /// with distinct indices; variable 0 is the constant one, and the first
    /// `num_instance` variables are the instance. Produces the same proof as
    /// [`groth16::prove()`] for the same randomness.
    pub fn prove<R: Rng>(
        pk: &ProvingKey<E>,
        key: &SparseGroth16Key<E>,
        num_instance: usize,
        assignment: &[(usize, E::ScalarField)],
        rng: &mut R,
    ) -> Result<(Proof<E>, Randomness<E>), SynthesisError> {
        let (u_g1, v_g1, v_g2, t_p_g1) = (&pk.a_query, &pk.b_g1_query, &pk.b_g2_query, &pk.l_query);

        let (a, b) = lagrange_of_sparse_vec(&key.a_matrix_cols, &key.b_matrix_cols, assignment)?;
        let q_g1 = cross_term(&key.basis, &a, &b);

        let scalars: Vec<_> = assignment
            .par_iter()
            .map(|(_, z)| z.into_bigint())
            .collect();
        let gather = |bases: &[E::G1Affine]| -> Vec<E::G1Affine> {
            assignment.par_iter().map(|(j, _)| bases[*j]).collect()
        };
        let a_g1 = pk.vk.alpha_g1 + E::G1::msm_bigint(&gather(u_g1), &scalars);
        let b_g1 = pk.beta_g1 + E::G1::msm_bigint(&gather(v_g1), &scalars);
        let b_g2_bases: Vec<E::G2Affine> = assignment.par_iter().map(|(j, _)| v_g2[*j]).collect();
        let b_g2 = pk.vk.beta_g2 + E::G2::msm_bigint(&b_g2_bases, &scalars);

        let (witness_bases, witness_scalars): (Vec<E::G1Affine>, Vec<_>) = assignment
            .iter()
            .filter(|(j, _)| *j >= num_instance)
            .map(|(j, z)| (t_p_g1[*j - num_instance], z.into_bigint()))
            .unzip();
        let c_g1 = E::G1::msm_bigint(&witness_bases, &witness_scalars) + q_g1;

        let r = E::ScalarField::rand(rng);
        let s = E::ScalarField::rand(rng);
        let (a_g1, b_g1, b_g2, c_g1) = blind(pk, (a_g1, b_g1, b_g2, c_g1), r, s);

        let proof = Proof {
            a: a_g1.into_affine(),
            b: b_g2.into_affine(),
            c: c_g1.into_affine(),
        };
        let randomness = Randomness {
            r,
            s,
            b_g1: b_g1.into_affine(),
        };
        Ok((proof, randomness))
    }

    /// `V`: see [`groth16::verify()`].
    pub fn verify(
        pvk: &PreparedVerifyingKey<E>,
        proof: &Proof<E>,
        instance: &[E::ScalarField],
    ) -> R1CSResult<bool> {
        groth16::verify(pvk, proof, instance)
    }
}
