//! The Groth16 setup `G` over Dynark's QAP.
//!
//! The prover and verifier keys are the [`ark_groth16`] types. The code uses
//! the paper's names for their fields:
//!
//! | Field | Name in this code | Contents (in the group) |
//! |---|---|---|
//! | `ProvingKey::a_query` | `u_g1` | `u_j(tau)` |
//! | `ProvingKey::b_g1_query` | `v_g1` | `v_j(tau)` |
//! | `ProvingKey::b_g2_query` | `v_g2` | `v_j(tau)`, in G2 |
//! | `ProvingKey::h_query` | `tau_powers_g1` | `tau^i * Z_H(tau) / delta` |
//! | `ProvingKey::l_query` | `t_p_g1` | `(beta * u_j + alpha * v_j + w_j)(tau) / delta`, for the witness |
//! | `VerifyingKey::gamma_abc_g1` | `t_v_g1` | the same, divided by `gamma`, for the instance |
use crate::{
    groth16::qap::{compute_u_v_w, tau_powers_field_lifted},
    ProvingKey, VerifyingKey,
};
use ark_ec::{pairing::Pairing, scalar_mul::BatchMulPreprocessing, AffineRepr, CurveGroup};
use ark_ff::{Field, UniformRand, Zero};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::{ConstraintMatrices, Result as R1CSResult, SynthesisError};
use ark_std::{rand::Rng, vec::Vec};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// `G`: samples fresh trapdoors and generators and returns the prover key,
/// the verifier key, and the evaluation domain.
pub fn setup<E: Pairing, R: Rng>(
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
    let domain_min_size = r1cs.num_constraints;
    let domain = GeneralEvaluationDomain::<E::ScalarField>::new(domain_min_size)
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;

    let tau = domain.sample_element_outside_domain(rng);
    let alpha = E::ScalarField::rand(rng);
    let beta = E::ScalarField::rand(rng);
    let gamma = E::ScalarField::rand(rng);
    let delta = E::ScalarField::rand(rng);

    let g1_gen = E::G1::rand(rng).into_affine();
    let g2_gen = E::G2::rand(rng).into_affine();

    let pk = generate_prover_key_with_matrices::<E>(
        domain,
        &r1cs,
        (tau, alpha, beta, gamma, delta, g1_gen, g2_gen),
    )?;

    let vk = pk.vk.clone();

    Ok((pk, vk, domain))
}

/// Builds the prover key from explicit trapdoors
/// `(tau, alpha, beta, gamma, delta)` and group generators.
pub fn generate_prover_key_with_matrices<E: Pairing>(
    domain: GeneralEvaluationDomain<E::ScalarField>,
    matrices: &ConstraintMatrices<E::ScalarField>,
    randomness: (
        E::ScalarField,
        E::ScalarField,
        E::ScalarField,
        E::ScalarField,
        E::ScalarField,
        E::G1Affine,
        E::G2Affine,
    ),
) -> R1CSResult<ProvingKey<E>> {
    type D<F> = GeneralEvaluationDomain<F>;
    let (tau, alpha, beta, gamma, delta, g1_gen, g2_gen) = randomness;
    let g1_gen = g1_gen.into_group();
    let g2_gen = g2_gen.into_group();

    let num_constraints = matrices.num_constraints;
    let num_instance = matrices.num_instance_variables;
    let num_witness = matrices.num_witness_variables;
    let domain_size = domain.size();
    let qap_num_variables = num_instance + num_witness;

    // process u, v, w
    ///////////////////////////////////////////////////////////////////////////

    let (u, v, w, vanish_tau) = compute_u_v_w::<E::ScalarField, D<E::ScalarField>>(
        &domain,
        num_constraints,
        num_instance,
        num_witness,
        matrices,
        &tau,
    )?;

    // Compute query densities
    let non_zero_u: usize = cfg_into_iter!(0..(qap_num_variables))
        .map(|i| usize::from(!u[i].is_zero()))
        .sum();

    let non_zero_v: usize = cfg_into_iter!(0..(qap_num_variables))
        .map(|i| usize::from(!v[i].is_zero()))
        .sum();

    // process field elements
    ///////////////////////////////////////////////////////////////////////////

    let gamma_inverse = gamma.inverse().ok_or(SynthesisError::UnexpectedIdentity)?;
    let delta_inverse = delta.inverse().ok_or(SynthesisError::UnexpectedIdentity)?;

    let t_v = cfg_iter!(u[..num_instance])
        .zip(&v[..num_instance])
        .zip(&w[..num_instance])
        .map(|((u_j, v_j), w_j)| (beta * u_j + &(alpha * v_j) + w_j) * &gamma_inverse)
        .collect::<Vec<_>>();

    let t_p = cfg_iter!(u[num_instance..])
        .zip(&v[num_instance..])
        .zip(&w[num_instance..])
        .map(|((u_j, v_j), w_j)| (beta * u_j + &(alpha * v_j) + w_j) * &delta_inverse)
        .collect::<Vec<_>>();

    let tau_powers = tau_powers_field_lifted::<_, D<E::ScalarField>>(
        domain_size - 1,
        tau,
        vanish_tau,
        delta_inverse,
    )?;

    // process group elements
    ///////////////////////////////////////////////////////////////////////////

    // Compute G window table
    let g1_table = BatchMulPreprocessing::new(
        g1_gen,
        non_zero_u + non_zero_v + qap_num_variables + domain_size,
    );
    let g2_table = BatchMulPreprocessing::new(g2_gen, non_zero_v);

    let alpha_g1 = g1_gen * &alpha;
    let beta_g1 = g1_gen * &beta;
    let beta_g2 = g2_gen * &beta;
    let delta_g1 = g1_gen * &delta;
    let delta_g2 = g2_gen * &delta;

    let u_g1 = g1_table.batch_mul(&u);

    let v_g1 = g1_table.batch_mul(&v);

    let v_g2 = g2_table.batch_mul(&v);

    let t_p_g1 = g1_table.batch_mul(&t_p);

    let t_v_g1 = g1_table.batch_mul(&t_v);

    let tau_powers_g1 = g1_table.batch_mul(&tau_powers);

    let gamma_g2 = g2_gen * &gamma;

    let vk = VerifyingKey::<E> {
        alpha_g1: alpha_g1.into_affine(),
        beta_g2: beta_g2.into_affine(),
        gamma_g2: gamma_g2.into_affine(),
        delta_g2: delta_g2.into_affine(),
        gamma_abc_g1: t_v_g1,
    };

    Ok(ProvingKey {
        vk,
        beta_g1: beta_g1.into_affine(),
        delta_g1: delta_g1.into_affine(),
        a_query: u_g1,
        b_g1_query: v_g1,
        b_g2_query: v_g2,
        h_query: tau_powers_g1,
        l_query: t_p_g1,
    })
}
