//! The verifier `V`: the unmodified Groth16 verifier from `ark_groth16`.
use crate::{PreparedVerifyingKey, Proof};
use ark_ec::pairing::Pairing;
use ark_groth16::Groth16;
use ark_relations::r1cs::{Result as R1CSResult, SynthesisError};

/// `V`: checks a fresh or updated proof. `instance` is the full instance,
/// starting with the constant one, as passed to [`prove`](super::prove).
pub fn verify<E: Pairing>(
    pvk: &PreparedVerifyingKey<E>,
    proof: &Proof<E>,
    instance: &[E::ScalarField],
) -> R1CSResult<bool> {
    let public_inputs = instance
        .get(1..)
        .ok_or(SynthesisError::MalformedVerifyingKey)?;
    Groth16::<E>::verify_proof(pvk, proof, public_inputs)
}
