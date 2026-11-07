//! Dynark's R1CS-to-QAP reduction.
//!
//! This differs from `ark_groth16`'s `LibsnarkReduction`: the evaluation domain
//! is sized to the number of constraints, and no extra rows are appended for
//! the public inputs. Lagrange basis index `i` is therefore exactly R1CS row
//! `i`, which is how the update algorithms index the sparse changes to `A * z`
//! and `B * z`. Keys built here are not interchangeable with keys from
//! `ark_groth16`'s own generator.
use ark_ff::PrimeField;
use ark_poly::EvaluationDomain;
use ark_relations::r1cs::{ConstraintMatrices, Matrix, SynthesisError};
use ark_std::vec::Vec;

use rayon::prelude::*;

/// The evaluations of `A * z`, `B * z` and `C * z` over the domain, i.e. their
/// coefficients in the Lagrange basis, zero-padded to the domain size.
#[inline]
pub fn lagrange_of_dense_vec<F: PrimeField>(
    matrices: &ConstraintMatrices<F>,
    dense_vec: &[F],
) -> Result<(Vec<F>, Vec<F>, Vec<F>), SynthesisError> {
    let num_constraints = matrices.num_constraints;
    let domain_size = num_constraints.next_power_of_two();

    // a matrix and z product
    let mut a_lagrg: Vec<F> = vec![F::zero(); domain_size];
    let mut b_lagrg = vec![F::zero(); domain_size];
    let mut c_lagrg = vec![F::zero(); domain_size];

    cfg_iter_mut!(a_lagrg[..num_constraints])
        .zip(cfg_iter!(&matrices.a))
        .for_each(|(a_lagrg_i, a_row)| {
            *a_lagrg_i = inner_product_sparse_f_dense_f(&a_row, dense_vec);
        });

    cfg_iter_mut!(b_lagrg[..num_constraints])
        .zip(cfg_iter!(&matrices.b))
        .for_each(|(b_lagrg_i, b_row)| {
            *b_lagrg_i = inner_product_sparse_f_dense_f(&b_row, dense_vec);
        });

    cfg_iter_mut!(c_lagrg[..num_constraints])
        .zip(cfg_iter!(&matrices.c))
        .for_each(|(c_lagrg_i, c_row)| {
            *c_lagrg_i = inner_product_sparse_f_dense_f(&c_row, dense_vec);
        });

    Ok((a_lagrg, b_lagrg, c_lagrg))
}

/// A matrix by columns: for each of `num_columns` variables `j`, the pairs
/// `(row, matrix[row][j])`.
pub fn matrix_columns<F: PrimeField>(matrix: &Matrix<F>, num_columns: usize) -> Vec<Vec<(usize, F)>> {
    let mut cols: Vec<Vec<(usize, F)>> = vec![vec![]; num_columns];
    matrix.iter().enumerate().for_each(|(i, row)| {
        row.iter().for_each(|(val, j)| cols[*j].push((i, *val)));
    });
    cols
}

/// The Lagrange coefficients of `A * z` and `B * z` for a sparse `z`, given as
/// `(variable, value)` pairs, as sorted `(row, value)` lists; `a_cols` and
/// `b_cols` are `A` and `B` by columns (see [`matrix_columns`]). The cost
/// depends only on the columns of the variables in `z`.
///
/// With `z` a change to the assignment, these are the changes to the Lagrange
/// coefficients (the paper's `Delta^A` and `Delta^B`). The dense counterpart
/// is [`lagrange_of_dense_vec`].
pub fn lagrange_of_sparse_vec<F: PrimeField>(
    a_cols: &[Vec<(usize, F)>],
    b_cols: &[Vec<(usize, F)>],
    sparse_vec: &[(usize, F)],
) -> Result<(Vec<(usize, F)>, Vec<(usize, F)>), SynthesisError> {
    let mut a_pairs: Vec<(usize, F)> = sparse_vec
        .par_iter()
        .map(|(j, vl)| {
            a_cols[*j]
                .par_iter()
                .map(|(i, coef)| (*i, *coef * *vl))
                .collect::<Vec<(usize, F)>>()
        })
        .collect::<Vec<Vec<(usize, F)>>>()
        .into_par_iter()
        .flatten()
        .collect::<Vec<(usize, F)>>();

    let mut b_pairs: Vec<(usize, F)> = sparse_vec
        .par_iter()
        .map(|(j, vl)| {
            b_cols[*j]
                .par_iter()
                .map(|(i, coef)| (*i, *coef * *vl))
                .collect::<Vec<(usize, F)>>()
        })
        .collect::<Vec<Vec<(usize, F)>>>()
        .into_par_iter()
        .flatten()
        .collect::<Vec<(usize, F)>>();

    let mut a_lagrange_sparse: Vec<(usize, F)> = Vec::new();
    let mut b_lagrange_sparse: Vec<(usize, F)> = Vec::new();

    a_pairs.sort_by_key(|k| k.0);
    b_pairs.sort_by_key(|k| k.0);

    for (i, vl) in a_pairs {
        if let Some((last_i, sum)) = a_lagrange_sparse.last_mut() {
            if *last_i == i {
                *sum += vl; // accumulate
                continue;
            }
        }
        a_lagrange_sparse.push((i, vl)); // start new group
    }
    for (i, vl) in b_pairs {
        if let Some((last_i, sum)) = b_lagrange_sparse.last_mut() {
            if *last_i == i {
                *sum += vl; // accumulate
                continue;
            }
        }
        b_lagrange_sparse.push((i, vl)); // start new group
    }

    Ok((a_lagrange_sparse, b_lagrange_sparse))
}

/// Inner product of one sparse R1CS row with the assignment.
pub fn inner_product_sparse_f_dense_f<F: PrimeField>(terms: &[(F, usize)], assignment: &[F]) -> F {
    #[cfg(feature = "parallel")]
    if terms.len() < 5 {
        serial_inner_product_sparse_f_dense_f(terms, assignment)
    } else {
        terms
            .par_iter()
            .map(|(coeff, index)| {
                let val = assignment[*index];
                if coeff.is_one() {
                    val
                } else {
                    val * coeff
                }
            })
            .sum()
    }
    #[cfg(not(feature = "parallel"))]
    serial_inner_product_sparse_f_dense_f(terms, assignment)
}

fn serial_inner_product_sparse_f_dense_f<F: PrimeField>(
    terms: &[(F, usize)],
    assignment: &[F],
) -> F {
    let mut sum = F::zero();
    // Process elements in chunks for better CPU vectorization
    for chunk in terms.chunks(4) {
        let chunk_sum = chunk
            .iter()
            .map(|(coeff, index)| {
                let val = assignment[*index];
                if coeff.is_one() {
                    val
                } else {
                    val * coeff
                }
            })
            .sum::<F>();
        sum += chunk_sum;
    }
    sum
}

/// The coefficients of the quotient `h = (a * b - c) / Z_H`, given the Lagrange
/// coefficients `a`, `b`, `c` of `A * z`, `B * z`, `C * z`: the 6 DFTs of
/// [`quotient_on_coset`], then a 7th to go back to coefficients.
#[inline]
pub fn compute_quotient<F: PrimeField, D: EvaluationDomain<F>>(
    domain: &D,
    a: Vec<F>,
    b: Vec<F>,
    c: Vec<F>,
) -> Result<Vec<F>, SynthesisError> {
    let mut h = quotient_on_coset(domain, a, b, c)?;
    let coset_domain = domain.get_coset(F::GENERATOR).unwrap();
    coset_domain.ifft_in_place(&mut h);
    Ok(h)
}

/// The evaluations of the quotient `h = (a * b - c) / Z_H` on the coset
/// `g * H`, with `g = F::GENERATOR`, given the Lagrange coefficients `a`, `b`,
/// `c` of `A * z`, `B * z`, `C * z`. Takes 6 DFTs: `a`, `b`, `c` to
/// coefficients, then to evaluations on the coset, where `Z_H` is the nonzero
/// constant `g^n - 1`.
#[inline]
pub fn quotient_on_coset<F: PrimeField, D: EvaluationDomain<F>>(
    domain: &D,
    mut a: Vec<F>,
    mut b: Vec<F>,
    mut c: Vec<F>,
) -> Result<Vec<F>, SynthesisError> {
    domain.ifft_in_place(&mut a);
    domain.ifft_in_place(&mut b);

    let coset_domain = domain.get_coset(F::GENERATOR).unwrap();

    coset_domain.fft_in_place(&mut a);
    coset_domain.fft_in_place(&mut b);

    let mut ab = domain.mul_polynomials_in_evaluation_domain(&a, &b);
    drop(a);
    drop(b);

    domain.ifft_in_place(&mut c);
    coset_domain.fft_in_place(&mut c);

    let vanishing_polynomial_over_coset = domain
        .evaluate_vanishing_polynomial(F::GENERATOR)
        .inverse()
        .unwrap();
    cfg_iter_mut!(ab).zip(c).for_each(|(ab_i, c_i)| {
        *ab_i -= &c_i;
        *ab_i *= &vanishing_polynomial_over_coset;
    });

    Ok(ab)
}

/// The scalars `Z_H(tau) * tau^i / delta` for `i < max_power`, from which the generator
/// computes the prover-key elements used to commit to the quotient.
pub fn tau_powers_field_lifted<F: PrimeField, D: EvaluationDomain<F>>(
    max_power: usize,
    tau: F,
    vanish_tau: F,
    delta_inverse: F,
) -> Result<Vec<F>, SynthesisError> {
    let scalars = cfg_into_iter!(0..max_power)
        .map(|i| vanish_tau * &delta_inverse * &tau.pow([i as u64]))
        .collect::<Vec<_>>();
    Ok(scalars)
}

/// The scalars `tau^i / delta` for `i < max_power`.
pub fn tau_powers_field<F: PrimeField, D: EvaluationDomain<F>>(
    max_power: usize,
    tau: F,
    delta_inverse: F,
) -> Result<Vec<F>, SynthesisError> {
    let scalars = cfg_into_iter!(0..max_power)
        .map(|i| delta_inverse * &tau.pow([i as u64]))
        .collect::<Vec<_>>();
    Ok(scalars)
}

/// The QAP instance map: `u_j(tau)`, `v_j(tau)`, `w_j(tau)` for every variable
/// `j`, together with `Z_H(tau)`.
#[inline]
#[allow(clippy::type_complexity)]
pub fn compute_u_v_w<F: PrimeField, D: EvaluationDomain<F>>(
    domain: &D,
    num_constraints: usize,
    num_instance_variables: usize,
    num_witness_variables: usize,
    matrices: &ConstraintMatrices<F>,
    tau: &F,
) -> Result<(Vec<F>, Vec<F>, Vec<F>, F), SynthesisError> {
    debug_assert!(matrices.num_constraints == num_constraints);
    debug_assert!(matrices.num_instance_variables == num_instance_variables);
    debug_assert!(matrices.num_witness_variables == num_witness_variables);

    let vanish_tau = domain.evaluate_vanishing_polynomial(*tau);

    // Evaluate all Lagrange polynomials
    let coefficients_time = start_timer!(|| "Evaluate Lagrange coefficients");
    let lagrange_vec_tau: Vec<F> = domain.evaluate_all_lagrange_coefficients(*tau);
    end_timer!(coefficients_time);

    let mut u: Vec<F> = vec![F::zero(); num_instance_variables + num_witness_variables];
    let mut v: Vec<F> = vec![F::zero(); num_instance_variables + num_witness_variables];
    let mut w: Vec<F> = vec![F::zero(); num_instance_variables + num_witness_variables];

    for (i, lagrange_i_tau) in lagrange_vec_tau.iter().enumerate().take(num_constraints) {
        for &(ref a_i_j, j) in &matrices.a[i] {
            u[j] += &(*lagrange_i_tau * a_i_j);
        }
        for &(ref b_i_j, j) in &matrices.b[i] {
            v[j] += &(*lagrange_i_tau * b_i_j);
        }
        for &(ref c_i_j, j) in &matrices.c[i] {
            w[j] += &(*lagrange_i_tau * c_i_j);
        }
    }
    Ok((u, v, w, vanish_tau))
}
