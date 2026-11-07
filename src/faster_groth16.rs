//! FasterGroth16: Groth16 with the quotient computed by 4 DFTs (appendix
//! "Proof Quotient" of the paper). Setup, proofs and verification are those
//! of Groth16; only the prover differs.
//!
//! The quotient commitment is written in the quotient basis:
//!
//! ```text
//! q~  = a .* conv(w_prime, b) + b .* conv(w_prime, a)
//! [Q] = sum_i q~_i * ell_bar[i] + c_i * diag[i]
//! ```
//!
//! where `a`, `b`, `c` are the Lagrange coefficients of `A * z`, `B * z` and
//! `C * z`. The two convolutions take 4 DFTs over the domain itself, instead
//! of the 6 or 7 DFTs over a coset that [`groth16::prove_6dft`] and
//! [`groth16::prove`] use, at the cost of a second MSM of size `n`.
use crate::{
    decomposition::QuotientBasis,
    groth16::{
        self,
        prover::{blind, raw_proof_with_quotient, Randomness},
        qap::lagrange_of_dense_vec,
    },
    poly::convolution::convolution_by_dft_f_f,
    PreparedVerifyingKey, Proof, ProvingKey, VerifyingKey,
};
use ark_ec::{pairing::Pairing, CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField, UniformRand};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::{ConstraintMatrices, Result as R1CSResult, SynthesisError};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{marker::PhantomData, rand::Rng, vec::Vec};
use rayon::prelude::*;

/// Groth16 with the 4-DFT prover over the pairing `E`.
pub struct FasterGroth16<E: Pairing> {
    _p: PhantomData<E>,
}

/// What [`FasterGroth16`] needs beyond the prover key: parts of the
/// [`QuotientBasis`], derived once per circuit from the prover key, with no
/// trapdoor.
#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct FasterGroth16Key<E: Pairing> {
    /// The quotient basis's `ell_bar`, in affine form for MSMs.
    pub ell_bar: Vec<E::G1Affine>,
    /// The quotient basis's `diag`, in affine form for MSMs.
    pub diag: Vec<E::G1Affine>,
    /// The DFT of the quotient basis's `w_prime`.
    pub w_prime_dft: Vec<E::ScalarField>,
}

impl<E: Pairing> FasterGroth16Key<E> {
    /// Derives the key from the prover key.
    pub fn new(domain: &GeneralEvaluationDomain<E::ScalarField>, pk: &ProvingKey<E>) -> Self {
        let basis = QuotientBasis::<E>::new(domain, &pk.h_query);
        Self {
            ell_bar: E::G1::normalize_batch(&basis.ell_bar),
            diag: E::G1::normalize_batch(&basis.diag),
            w_prime_dft: basis.w_prime_dft,
        }
    }
}

impl<E: Pairing> FasterGroth16<E> {
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

    /// `P` with 4 DFTs. Produces the same proof as [`groth16::prove()`] for
    /// the same randomness. `instance` starts with the constant one.
    pub fn prove<R: Rng>(
        pk: &ProvingKey<E>,
        key: &FasterGroth16Key<E>,
        r1cs: &ConstraintMatrices<E::ScalarField>,
        instance: &[E::ScalarField],
        witness: &[E::ScalarField],
        rng: &mut R,
    ) -> Result<(Proof<E>, Randomness<E>), SynthesisError> {
        let domain = GeneralEvaluationDomain::<E::ScalarField>::new(r1cs.num_constraints)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;

        let (a, b, c) = lagrange_of_dense_vec(r1cs, &[instance, witness].concat())?;
        let w_prime_dft = &key.w_prime_dft;
        let (w_b, w_a) = rayon::join(
            || convolution_by_dft_f_f::<E>(&domain, w_prime_dft, &b),
            || convolution_by_dft_f_f::<E>(&domain, w_prime_dft, &a),
        );
        let q_tilde: Vec<_> = a
            .par_iter()
            .zip(b.par_iter())
            .zip(w_b.par_iter().zip(w_a.par_iter()))
            .map(|((a_i, b_i), (w_b_i, w_a_i))| (*a_i * w_b_i + *b_i * w_a_i).into_bigint())
            .collect();
        let c: Vec<_> = c.par_iter().map(|c_i| c_i.into_bigint()).collect();
        let q_g1 = E::G1::msm_bigint(&key.ell_bar, &q_tilde) + E::G1::msm_bigint(&key.diag, &c);

        let raw = raw_proof_with_quotient(pk, instance, witness, q_g1);
        let r = E::ScalarField::rand(rng);
        let s = E::ScalarField::rand(rng);
        let (a_g1, b_g1, b_g2, c_g1) = blind(pk, raw, r, s);

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
