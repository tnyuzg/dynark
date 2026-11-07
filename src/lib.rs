//! Reference implementation of [Dynark: Making Groth16 Dynamic][paper].
//!
//! Given a Groth16 proof for an extended witness `z`, Dynark produces a proof
//! for a witness `z'` that differs from `z` in `d` entries, using `O(d)` group
//! operations instead of re-running the prover.
//!
//! Dynark keeps the Groth16 prover key, so keys and proofs are the
//! [`ark_groth16`] types, and fresh and updated proofs alike are checked by
//! the unmodified `ark_groth16` verifier.
//!
//! # Where each part of the paper lives
//!
//! The modules are layered: each one uses only the ones above it here.
//!
//! | Paper | Code |
//! |---|---|
//! | Circular convolution; sparse convolution; stepwise DFT | [`poly`] |
//! | Quotient decomposition: `ell_bar`, `diag`, `w`, `w_prime` | [`decomposition::QuotientBasis`] |
//! | Projection term, and the per-witness data precomputed for it (in one go, step by step, or with the Feist-Khovratovich baseline) | [`decomposition::projection`] |
//! | Cross term | [`decomposition::cross`] |
//! | Groth16 setup `G`, prover `P` (7 and 6 DFTs) and verifier `V`, with Dynark's conversion of R1CS to polynomials | [`groth16`] |
//! | FasterGroth16: the prover with 4 DFTs and a second MSM (appendix "Proof Quotient"); not faster on dense witnesses | [`FasterGroth16`] |
//! | SparseGroth16: the sparse prover ("Sparse zkSNARKs") | [`SparseGroth16`] |
//! | SemiDynark: preprocessing `K` and update `U` | [`SemiDynark`] |
//! | FullyDynark, de-amortized | [`FullyDynark`] |
//! | Synthetic R1CS used in the evaluation | [`synthetic_r1cs`] |
//!
//! [paper]: https://eprint.iacr.org/2025/1897
#![warn(
    unused,
    future_incompatible,
    nonstandard_style,
    rust_2018_idioms,
    missing_docs
)]

#[macro_use]
extern crate ark_std;

pub mod circuit;
pub mod decomposition;
pub mod faster_groth16;
pub mod fully_dynamic;
pub mod groth16;
pub mod poly;
pub mod semi_dynamic;
pub mod sparse_groth16;
pub mod synthetic_r1cs;

pub use ark_groth16::{
    prepare_verifying_key, PreparedVerifyingKey, Proof, ProvingKey, VerifyingKey,
};
pub use decomposition::QuotientBasis;
pub use faster_groth16::{FasterGroth16, FasterGroth16Key};
pub use fully_dynamic::{FullyCheckpoint, FullyDynark};
pub use semi_dynamic::{SemiDynark, SemiDynarkData, UpdatingKey};
pub use sparse_groth16::{SparseGroth16, SparseGroth16Key};
