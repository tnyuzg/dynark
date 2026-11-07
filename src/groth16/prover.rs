//! The Groth16 prover `P` over Dynark's QAP, in two versions that produce the
//! same proof:
//!
//! - [`prove`] computes the quotient with 7 DFTs, as arkworks does, and needs
//!   only the prover key.
//! - [`prove_6dft`] skips the last DFT by committing to the quotient's
//!   evaluations on a coset, which needs a [`CosetBasis`] precomputed once per
//!   circuit. This is the version snarkjs uses, and the baseline that the
//!   benchmarks compare against.
use crate::{
    groth16::qap::{compute_quotient, lagrange_of_dense_vec, quotient_on_coset},
    Proof, ProvingKey,
};
use ark_ec::{pairing::Pairing, CurveGroup, VariableBaseMSM};
use ark_ff::{FftField, Field, PrimeField, UniformRand, Zero};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::{ConstraintMatrices, SynthesisError};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{rand::Rng, vec::Vec};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

type D<F> = GeneralEvaluationDomain<F>;

/// The prover's randomness `(r, s)` (the paper's `\rho`), together with `B` in
/// G1, which re-blinding the proof later needs.
#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Randomness<E: Pairing> {
    /// The blinding factor `r`.
    pub r: E::ScalarField,
    /// The blinding factor `s`.
    pub s: E::ScalarField,
    /// The `B` element in `G1`.
    pub b_g1: E::G1Affine,
}

/// A proof before blinding, as `(A, B in G1, B in G2, C)`. `B` is kept in both
/// groups because blinding `C` needs it in `G1`.
pub(crate) type RawProof<E> = (
    <E as Pairing>::G1,
    <E as Pairing>::G1,
    <E as Pairing>::G2,
    <E as Pairing>::G1,
);

/// `P`: proves that `(instance, witness)` satisfies `r1cs`. `instance` starts
/// with the constant one.
pub fn prove<E: Pairing, R: Rng>(
    pk: &ProvingKey<E>,
    r1cs: &ConstraintMatrices<E::ScalarField>,
    instance: &[E::ScalarField],
    witness: &[E::ScalarField],
    rng: &mut R,
) -> Result<(Proof<E>, Randomness<E>), SynthesisError> {
    let domain = GeneralEvaluationDomain::<E::ScalarField>::new(r1cs.num_constraints)
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;

    let (a, b, c) = lagrange_of_dense_vec(r1cs, &[instance, witness].concat())?;
    let raw = raw_proof(pk, &domain, instance, witness, &a, &b, &c)?;

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

/// The Lagrange basis of the coset `g * H`, with `g = F::GENERATOR`, scaled
/// like `h_query`: element `i` is `L_i(tau) * Z_H(tau) / delta` in G1, where
/// `L_i` is the `i`-th Lagrange polynomial of the coset, truncated to degree
/// `n - 2`. It is derived once per circuit from `h_query`, with no trapdoor.
///
/// Truncating is safe because the quotient has degree at most `n - 2`: the
/// dropped `X^(n-1)` terms of the basis contribute the quotient's
/// `X^(n-1)` coefficient, which is zero.
#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct CosetBasis<E: Pairing> {
    /// The basis elements, in affine form for MSMs.
    pub elements: Vec<E::G1Affine>,
}

impl<E: Pairing> CosetBasis<E> {
    /// Derives the basis from `h_query`, the elements `tau^k * Z_H(tau) / delta`
    /// in G1 for `k < n - 1`. Element `i` is
    /// `(1/n) * sum_k (g * omega^i)^(-k) * h_query[k]`, i.e. the inverse DFT of
    /// `g^(-k) * h_query[k]`.
    pub fn new(domain: &GeneralEvaluationDomain<E::ScalarField>, h_query: &[E::G1Affine]) -> Self {
        let n = domain.size();
        let g_inv = E::ScalarField::GENERATOR.inverse().unwrap();
        let mut scaled: Vec<E::G1> = h_query
            .iter()
            .scan(E::ScalarField::from(1u64), |g_pow, h| {
                let term = *h * *g_pow;
                *g_pow *= g_inv;
                Some(term)
            })
            .collect();
        scaled.resize(n, E::G1::zero());
        domain.ifft_in_place(&mut scaled);
        Self {
            elements: E::G1::normalize_batch(&scaled),
        }
    }
}

/// `P` with 6 DFTs: the quotient's evaluations on a coset, committed directly
/// with the [`CosetBasis`]. Produces the same proof as [`prove`] for the same
/// randomness. `instance` starts with the constant one.
pub fn prove_6dft<E: Pairing, R: Rng>(
    pk: &ProvingKey<E>,
    coset_basis: &CosetBasis<E>,
    r1cs: &ConstraintMatrices<E::ScalarField>,
    instance: &[E::ScalarField],
    witness: &[E::ScalarField],
    rng: &mut R,
) -> Result<(Proof<E>, Randomness<E>), SynthesisError> {
    let domain = GeneralEvaluationDomain::<E::ScalarField>::new(r1cs.num_constraints)
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;

    let (a, b, c) = lagrange_of_dense_vec(r1cs, &[instance, witness].concat())?;
    let h_on_coset = quotient_on_coset::<E::ScalarField, D<E::ScalarField>>(&domain, a, b, c)?;
    let h_on_coset = cfg_into_iter!(h_on_coset)
        .map(|h_i| h_i.into_bigint())
        .collect::<Vec<_>>();
    let q_g1 = E::G1::msm_bigint(&coset_basis.elements, &h_on_coset);
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

/// The proof with blinding factors `r = s = 0`, from the Lagrange coefficients
/// `a`, `b`, `c` of `A * z`, `B * z`, `C * z`.
pub(crate) fn raw_proof<E: Pairing>(
    pk: &ProvingKey<E>,
    domain: &GeneralEvaluationDomain<E::ScalarField>,
    instance: &[E::ScalarField],
    witness: &[E::ScalarField],
    a: &[E::ScalarField],
    b: &[E::ScalarField],
    c: &[E::ScalarField],
) -> Result<RawProof<E>, SynthesisError> {
    let q_g1 = quotient_commitment(pk, domain, a, b, c)?;
    Ok(raw_proof_with_quotient(pk, instance, witness, q_g1))
}

/// The quotient commitment `sum_k q_k * tau_powers_g1[k]`, with the quotient
/// computed by 7 DFTs over a coset of the domain.
fn quotient_commitment<E: Pairing>(
    pk: &ProvingKey<E>,
    domain: &GeneralEvaluationDomain<E::ScalarField>,
    a: &[E::ScalarField],
    b: &[E::ScalarField],
    c: &[E::ScalarField],
) -> Result<E::G1, SynthesisError> {
    let tau_powers_g1 = &pk.h_query;
    let quotient = compute_quotient::<E::ScalarField, D<E::ScalarField>>(
        domain,
        a.to_vec(),
        b.to_vec(),
        c.to_vec(),
    )?;
    let quotient = cfg_into_iter!(quotient)
        .map(|q_i| q_i.into_bigint())
        .collect::<Vec<_>>();
    Ok(E::G1::msm_bigint(tau_powers_g1, &quotient))
}

/// The proof with blinding factors `r = s = 0`, given the quotient
/// commitment `q_g1`.
pub(crate) fn raw_proof_with_quotient<E: Pairing>(
    pk: &ProvingKey<E>,
    instance: &[E::ScalarField],
    witness: &[E::ScalarField],
    q_g1: E::G1,
) -> RawProof<E> {
    let (u_g1, v_g1, v_g2, t_p_g1) = (&pk.a_query, &pk.b_g1_query, &pk.b_g2_query, &pk.l_query);

    let instance = instance
        .iter()
        .map(|instance_i| instance_i.into_bigint())
        .collect::<Vec<_>>();

    let witness = cfg_iter!(witness)
        .map(|witness_i| witness_i.into_bigint())
        .collect::<Vec<_>>();

    let t_p_witness_g1 = E::G1::msm_bigint(t_p_g1, &witness);

    let assignment = [&instance[..], &witness[..]].concat();

    let a_g1 = pk.vk.alpha_g1 + E::G1::msm_bigint(u_g1, &assignment);
    let b_g1 = pk.beta_g1 + E::G1::msm_bigint(v_g1, &assignment);
    let b_g2 = pk.vk.beta_g2 + E::G2::msm_bigint(v_g2, &assignment);
    let c_g1 = t_p_witness_g1 + q_g1;

    (a_g1, b_g1, b_g2, c_g1)
}

/// Adds the zero-knowledge blinding `(r, s)` to a raw proof:
/// `A += r * delta`, `B += s * delta`, `C += s * A + r * B - r * s * delta`.
pub(crate) fn blind<E: Pairing>(
    pk: &ProvingKey<E>,
    raw: RawProof<E>,
    r: E::ScalarField,
    s: E::ScalarField,
) -> RawProof<E> {
    let (raw_a_g1, raw_b_g1, raw_b_g2, raw_c_g1) = raw;
    let a_g1 = raw_a_g1 + pk.delta_g1 * r;
    let b_g1 = raw_b_g1 + pk.delta_g1 * s;
    let b_g2 = raw_b_g2 + pk.vk.delta_g2 * s;
    let c_g1 = raw_c_g1 + a_g1 * s + b_g1 * r - pk.delta_g1 * (r * s);
    (a_g1, b_g1, b_g2, c_g1)
}
