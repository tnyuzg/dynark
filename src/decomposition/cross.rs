//! The cross term of the proof update (Section "Cross Term" of the paper).
use crate::{decomposition::QuotientBasis, poly::sparse_conv::cross_sums_sparse_conv};
use ark_ec::pairing::Pairing;
use ark_ff::Zero;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::vec::Vec;
use rayon::prelude::*;

/// The cross term between sparse changes `da` and `db` to the Lagrange
/// coefficients of `A * z` and `B * z`:
///
/// ```text
///   sum_i da_i * (sum_j w_prime[i - j] * db_j) * ell_bar[i]
/// + sum_j db_j * (sum_i w_prime[j - i] * da_i) * ell_bar[j]
/// + sum_i da_i * db_i * diag[i]
/// ```
///
/// The inner sums are computed by [`CrossTermMethod::Auto`]; see
/// [`cross_term_with`] to choose the method.
pub fn cross_term<E: Pairing>(
    basis: &QuotientBasis<E>,
    a_lagrange_sparse: &[(usize, E::ScalarField)],
    b_lagrange_sparse: &[(usize, E::ScalarField)],
) -> E::G1 {
    cross_term_with(basis, a_lagrange_sparse, b_lagrange_sparse, CrossTermMethod::Auto)
}

/// How [`cross_term_with`] computes the inner sums of the cross term.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossTermMethod {
    /// Directly, over every pair of changed rows: `|da| * |db|` field
    /// operations.
    Pairwise,
    /// By sparse convolution ([`crate::poly::sparse_conv`]): `O(m log^2 m)` field
    /// operations for `m` changed rows, with a large constant.
    SparseConv,
    /// [`SparseConv`](Self::SparseConv) when
    /// `|da| * |db| > SPARSE_CONV_COST_RATIO * m * log2(m)^2`, where `m` is the
    /// number of distinct changed rows, and [`Pairwise`](Self::Pairwise)
    /// otherwise.
    Auto,
}

/// The cost ratio that [`CrossTermMethod::Auto`] uses. At this ratio the two
/// methods took about the same time on an Apple M3 (8 cores); the measured
/// ratio was about 18 on 8 threads and 12 on 1 thread. Re-measure on other
/// machines with `cargo bench --bench cross_term`.
pub const SPARSE_CONV_COST_RATIO: usize = 16;

impl CrossTermMethod {
    /// The method to use for changes `a` and `b`: `self`, or for `Auto`, the
    /// one the cost model predicts is faster.
    pub fn resolve<F>(self, a: &[(usize, F)], b: &[(usize, F)]) -> Self {
        match self {
            Self::Auto => {
                let m = distinct_rows(a, b);
                let log_m = usize::BITS - m.max(1).leading_zeros();
                let sparse_cost = SPARSE_CONV_COST_RATIO * m * (log_m * log_m) as usize;
                if a.len().saturating_mul(b.len()) > sparse_cost {
                    Self::SparseConv
                } else {
                    Self::Pairwise
                }
            },
            method => method,
        }
    }
}

/// The number of distinct indices in `a` and `b`.
fn distinct_rows<F>(a: &[(usize, F)], b: &[(usize, F)]) -> usize {
    let mut rows: Vec<usize> = a.iter().chain(b).map(|(i, _)| *i).collect();
    rows.sort_unstable();
    rows.dedup();
    rows.len()
}

/// The cross term, with its inner sums computed by `method`.
pub fn cross_term_with<E: Pairing>(
    basis: &QuotientBasis<E>,
    a_lagrange_sparse: &[(usize, E::ScalarField)],
    b_lagrange_sparse: &[(usize, E::ScalarField)],
    method: CrossTermMethod,
) -> E::G1 {
    let n = basis.w.len();
    let off_diagonal = match method.resolve(a_lagrange_sparse, b_lagrange_sparse) {
        CrossTermMethod::SparseConv => {
            let omega = GeneralEvaluationDomain::<E::ScalarField>::new(n)
                .unwrap()
                .group_gen();
            let (s_b, s_a) =
                cross_sums_sparse_conv(omega, n, a_lagrange_sparse, b_lagrange_sparse);
            let c_cross_1 = a_lagrange_sparse
                .par_iter()
                .zip(s_b.par_iter())
                .map(|((i, a), s)| basis.ell_bar[*i] * (*a * s))
                .sum::<E::G1>();
            let c_cross_2 = b_lagrange_sparse
                .par_iter()
                .zip(s_a.par_iter())
                .map(|((j, b), s)| basis.ell_bar[*j] * (*b * s))
                .sum::<E::G1>();
            c_cross_1 + c_cross_2
        },
        _ => {
            let c_cross_1 = a_lagrange_sparse
                .par_iter()
                .map(|(i, a)| {
                    let ot_sum = b_lagrange_sparse
                        .par_iter()
                        .map(|(j, b)| {
                            if i == j {
                                E::ScalarField::zero()
                            } else {
                                basis.w[(n + j - i) % n] * b
                            }
                        })
                        .sum::<E::ScalarField>();
                    basis.ell_bar[*i] * (*a * ot_sum)
                })
                .sum::<E::G1>();

            let c_cross_2 = b_lagrange_sparse
                .par_iter()
                .map(|(j, b)| {
                    let ot_sum = a_lagrange_sparse
                        .par_iter()
                        .map(|(i, a)| {
                            if i == j {
                                E::ScalarField::zero()
                            } else {
                                basis.w[(n + *i - *j) % n] * a
                            }
                        })
                        .sum::<E::ScalarField>();
                    basis.ell_bar[*j] * (*b * ot_sum)
                })
                .sum::<E::G1>();
            c_cross_1 + c_cross_2
        },
    };
    off_diagonal + diagonal_term(basis, a_lagrange_sparse, b_lagrange_sparse)
}

/// `sum_i da_i * db_i * diag[i]` over the rows changed in both.
fn diagonal_term<E: Pairing>(
    basis: &QuotientBasis<E>,
    a_lagrange_sparse: &[(usize, E::ScalarField)],
    b_lagrange_sparse: &[(usize, E::ScalarField)],
) -> E::G1 {
    let mut results = Vec::<(usize, E::ScalarField)>::new();
    let mut cur_i = 0;
    let mut cur_j = 0;
    let mut a_copy = a_lagrange_sparse.to_vec();
    let mut b_copy = b_lagrange_sparse.to_vec();
    a_copy.sort_by_key(|x| x.0);
    b_copy.sort_by_key(|y| y.0);
    // two-pointer merge join
    while cur_i < a_copy.len() && cur_j < b_copy.len() {
        match a_copy[cur_i].0.cmp(&b_copy[cur_j].0) {
            std::cmp::Ordering::Less => cur_i += 1,
            std::cmp::Ordering::Greater => cur_j += 1,
            std::cmp::Ordering::Equal => {
                results.push((a_copy[cur_i].0, a_copy[cur_i].1 * b_copy[cur_j].1));
                cur_i += 1;
                cur_j += 1;
            },
        }
    }
    let c_cross_3 = results
        .par_iter()
        .map(|(i, val)| basis.diag[*i] * val)
        .sum::<E::G1>();

    c_cross_3
}
