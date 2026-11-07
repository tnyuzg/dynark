//! Quotient decomposition: every quotient in Dynark is a combination of a
//! fixed [`QuotientBasis`].
//!
//! An update changes the proof's quotient by two parts: a projection term
//! ([`projection`]), which pairs a change on one side with the other side's
//! precomputed projection data, and a cross term ([`cross`]), which pairs the
//! changes on both sides.
pub mod basis;
pub mod cross;
pub mod projection;

pub use basis::QuotientBasis;
