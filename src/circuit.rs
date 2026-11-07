//! Running Dynark on circuits written against `ark-relations`.
use ark_ff::PrimeField;
use ark_relations::r1cs::{
    ConstraintMatrices, ConstraintSynthesizer, ConstraintSystem, OptimizationGoal, SynthesisError,
};
use ark_std::vec::Vec;

/// Synthesizes `circuit` into its constraint matrices and its instance and
/// witness assignments. The instance starts with the constant one, as
/// [`groth16::prove`](crate::groth16::prove()) expects.
#[allow(clippy::type_complexity)]
pub fn synthesize<F: PrimeField, C: ConstraintSynthesizer<F>>(
    circuit: C,
) -> Result<(ConstraintMatrices<F>, Vec<F>, Vec<F>), SynthesisError> {
    let cs = ConstraintSystem::new_ref();
    cs.set_optimization_goal(OptimizationGoal::Constraints);
    circuit.generate_constraints(cs.clone())?;
    cs.finalize();

    let matrices = cs.to_matrices().ok_or(SynthesisError::MissingCS)?;
    let cs = cs.into_inner().ok_or(SynthesisError::MissingCS)?;
    Ok((matrices, cs.instance_assignment, cs.witness_assignment))
}
