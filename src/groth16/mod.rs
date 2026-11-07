//! Baseline Groth16 over Dynark's QAP: setup `G`, prover `P` and verifier `V`,
//! with the [`ark_groth16`] key and proof types.
//!
//! Both provers produce the same proof for the same randomness: [`prove`]
//! computes the quotient with 7 DFTs, as arkworks does, and [`prove_6dft`]
//! with 6, as snarkjs does (the baseline for comparisons). FasterGroth16 and
//! SparseGroth16 are in their own modules.
pub mod prover;
pub mod qap;
pub mod setup;
pub mod verifier;

pub use prover::{prove, prove_6dft, CosetBasis, Randomness};
pub use setup::{generate_prover_key_with_matrices, setup};
pub use verifier::verify;
