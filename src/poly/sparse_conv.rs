//! Sparse convolution with the fixed vector `w` (Protocol "Sparse
//! Convolution" in the paper's appendix), in `O(m log^2 m)` field operations
//! for `m` nonzero entries.
//!
//! The sums it needs, `S_k = sum_{l != k} c_l / (x_k - x_l)`, are computed with
//! the subproduct tree of the points: with `V(X) = prod_l (X - x_l)` and
//! `N(X) = sum_l c_l * V(X) / (X - x_l)`,
//! `S_k = (N'(x_k) - c_k * V''(x_k) / 2) / V'(x_k)`.
use ark_ff::{batch_inversion, FftField, Field, PrimeField};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::vec::Vec;
use rayon::prelude::*;

/// Below this many coefficients, multiply schoolbook.
pub const NAIVE_MUL: usize = 64;
/// Below this many points, evaluate a remainder by Horner.
pub const EVAL_LEAF: usize = 32;

// Polynomial arithmetic on coefficient vectors, lowest degree first.

/// The product of two polynomials.
pub fn mul<F: FftField>(a: &[F], b: &[F]) -> Vec<F> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let out_len = a.len() + b.len() - 1;
    if a.len().min(b.len()) <= NAIVE_MUL {
        let mut r = vec![F::zero(); out_len];
        for (i, x) in a.iter().enumerate() {
            for (j, y) in b.iter().enumerate() {
                r[i + j] += *x * y;
            }
        }
        return r;
    }
    let d = GeneralEvaluationDomain::<F>::new(out_len).unwrap();
    let mut fa = a.to_vec();
    let mut fb = b.to_vec();
    fa.resize(d.size(), F::zero());
    fb.resize(d.size(), F::zero());
    d.fft_in_place(&mut fa);
    d.fft_in_place(&mut fb);
    fa.par_iter_mut().zip(fb.par_iter()).for_each(|(x, y)| *x *= y);
    d.ifft_in_place(&mut fa);
    fa.truncate(out_len);
    fa
}

fn add_into<F: Field>(acc: &mut Vec<F>, b: &[F]) {
    if acc.len() < b.len() {
        acc.resize(b.len(), F::zero());
    }
    acc.iter_mut().zip(b).for_each(|(x, y)| *x += y);
}

/// `g^{-1} mod X^k`, for `g[0] != 0`, by Newton iteration.
pub fn inv_series<F: FftField>(g: &[F], k: usize) -> Vec<F> {
    let mut h = vec![g[0].inverse().unwrap()];
    let mut l = 1;
    while l < k {
        l = (2 * l).min(k);
        let mut e = mul(&g[..g.len().min(l)], &h);
        e.resize(l, F::zero());
        e.iter_mut().for_each(|x| *x = -*x);
        e[0] += F::from(2u64);
        h = mul(&h, &e);
        h.truncate(l);
    }
    h.truncate(k);
    h
}

/// `f mod g` for monic `g`, given `g_rev_inv = rev(g)^{-1} mod X^{deg f - deg g + 1}`
/// (or longer).
fn rem<F: FftField>(f: &[F], g: &[F], g_rev_inv: &[F]) -> Vec<F> {
    let dg = g.len() - 1;
    if f.len() <= dg {
        return f.to_vec();
    }
    let k = f.len() - dg; // number of quotient coefficients
    if k.min(dg) <= NAIVE_MUL {
        // schoolbook division by a monic divisor
        let mut r = f.to_vec();
        for i in (dg..f.len()).rev() {
            let q = r[i];
            if q.is_zero() {
                continue;
            }
            for (j, gj) in g.iter().enumerate().take(dg) {
                r[i - dg + j] -= q * gj;
            }
            r[i] = F::zero();
        }
        r.truncate(dg);
        return r;
    }
    let f_rev: Vec<F> = f.iter().rev().take(k).copied().collect();
    let mut q_rev = mul(&f_rev, &g_rev_inv[..k]);
    q_rev.resize(k, F::zero());
    q_rev.reverse();
    let qg = mul(&q_rev, g);
    let mut r: Vec<F> = f[..dg].to_vec();
    r.iter_mut().zip(&qg).for_each(|(x, y)| *x -= y);
    r
}

fn deriv<F: Field>(p: &[F]) -> Vec<F> {
    p.iter()
        .enumerate()
        .skip(1)
        .map(|(i, c)| F::from(i as u64) * c)
        .collect()
}

fn horner<F: Field>(p: &[F], x: F) -> F {
    p.iter().rev().fold(F::zero(), |acc, c| acc * x + c)
}

// Subproduct tree.

/// A node over the points `xs[lo..hi]`, holding `V(X) = prod (X - x)`, which is monic.
pub struct Node<F> {
    lo: usize,
    hi: usize,
    poly: Vec<F>,
    /// `rev(poly)^-1`, to the precision needed when reducing the parent's
    /// remainder.
    rev_inv: Vec<F>,
    children: Option<Box<(Node<F>, Node<F>)>>,
}

/// The subproduct tree of the distinct points `xs`.
pub fn build_tree<F: FftField>(xs: &[F]) -> Node<F> {
    let mut root = build(xs, 0, xs.len());
    root.rev_inv = vec![];
    root
}

fn build<F: FftField>(xs: &[F], lo: usize, hi: usize) -> Node<F> {
    if hi - lo == 1 {
        return Node {
            lo,
            hi,
            poly: vec![-xs[lo], F::one()],
            rev_inv: vec![],
            children: None,
        };
    }
    let mid = (lo + hi) / 2;
    let (mut l, mut r) = rayon::join(|| build(xs, lo, mid), || build(xs, mid, hi));
    let poly = mul(&l.poly, &r.poly);
    // A remainder reduced by the parent has degree < deg(l) + deg(r), so
    // reducing it by one child needs `deg(sibling)` quotient coefficients.
    if hi - lo > EVAL_LEAF {
        let (dl, dr) = (mid - lo, hi - mid);
        let (il, ir) = rayon::join(
            || inv_series(&rev(&l.poly), dr),
            || inv_series(&rev(&r.poly), dl),
        );
        l.rev_inv = il;
        r.rev_inv = ir;
    }
    Node {
        lo,
        hi,
        poly,
        rev_inv: vec![],
        children: Some(Box::new((l, r))),
    }
}

fn rev<F: Copy>(p: &[F]) -> Vec<F> {
    p.iter().rev().copied().collect()
}

/// Evaluates `f` (of degree < number of points under `node`) at those points.
pub fn eval_tree<F: FftField>(node: &Node<F>, xs: &[F], f: &[F]) -> Vec<F> {
    if node.hi - node.lo <= EVAL_LEAF || node.children.is_none() {
        return (node.lo..node.hi).map(|i| horner(f, xs[i])).collect();
    }
    let (l, r) = node.children.as_deref().unwrap();
    let (fl, fr) = rayon::join(|| rem(f, &l.poly, &l.rev_inv), || rem(f, &r.poly, &r.rev_inv));
    let (mut vl, vr) = rayon::join(|| eval_tree(l, xs, &fl), || eval_tree(r, xs, &fr));
    vl.extend(vr);
    vl
}

/// `N(X) = sum_k c_k * V(X) / (X - x_k)` over the points under `node`.
pub fn lin_comb<F: FftField>(node: &Node<F>, c: &[F]) -> Vec<F> {
    match node.children.as_deref() {
        None => vec![c[node.lo]],
        Some((l, r)) => {
            let (nl, nr) = rayon::join(|| lin_comb(l, c), || lin_comb(r, c));
            let (mut a, b) = rayon::join(|| mul(&nl, &r.poly), || mul(&nr, &l.poly));
            add_into(&mut a, &b);
            a
        },
    }
}

// Cauchy sums.

/// For each input vector `c`, the vector `S_k = sum_{l != k} c_l / (x_k - x_l)`
/// over the distinct points `xs`.
pub fn cauchy_sums<F: FftField>(xs: &[F], cs: &[&[F]]) -> Vec<Vec<F>> {
    let m = xs.len();
    if m == 0 {
        return cs.iter().map(|_| vec![]).collect();
    }
    let tree = build_tree(xs);
    let v1 = deriv(&tree.poly);
    let v2 = deriv(&v1);
    let (mut v1_at, v2_at) = rayon::join(|| eval_tree(&tree, xs, &v1), || eval_tree(&tree, xs, &v2));
    batch_inversion(&mut v1_at); // 1 / V'(x_k)
    let half = F::from(2u64).inverse().unwrap();
    cs.par_iter()
        .map(|c| {
            let n1 = deriv(&lin_comb(&tree, c));
            let n1_at = eval_tree(&tree, xs, &n1);
            (0..m)
                .map(|k| (n1_at[k] - c[k] * v2_at[k] * half) * v1_at[k])
                .collect()
        })
        .collect()
}

/// The inner sums of the cross term, via sparse convolution: for each `(i, _)`
/// in `a`, `sum_j w[(j - i) mod n] * b_j`, and for each `(j, _)` in `b`,
/// `sum_i w[(i - j) mod n] * a_i`. Indices must be distinct within each list.
///
/// Since `w[j - i] = omega^j / (n * (omega^i - omega^j))`, both are Cauchy
/// sums over the points `omega^k` for `k` in the union of the indices.
pub fn cross_sums_sparse_conv<F: PrimeField>(
    omega: F,
    n: usize,
    a: &[(usize, F)],
    b: &[(usize, F)],
) -> (Vec<F>, Vec<F>) {
    let mut idx: Vec<usize> = a.iter().chain(b).map(|(i, _)| *i).collect();
    idx.sort_unstable();
    idx.dedup();
    let pos = |i: usize| idx.binary_search(&i).unwrap();
    let xs: Vec<F> = idx.par_iter().map(|i| omega.pow([*i as u64])).collect();
    let n_inv = F::from(n as u64).inverse().unwrap();
    let mut ca = vec![F::zero(); idx.len()];
    let mut cb = vec![F::zero(); idx.len()];
    for (i, v) in a {
        let k = pos(*i);
        ca[k] += *v * xs[k] * n_inv;
    }
    for (j, v) in b {
        let k = pos(*j);
        cb[k] += *v * xs[k] * n_inv;
    }
    let s = cauchy_sums(&xs, &[&cb, &ca]);
    let s_b = a.iter().map(|(i, _)| s[0][pos(*i)]).collect();
    let s_a = b.iter().map(|(j, _)| s[1][pos(*j)]).collect();
    (s_b, s_a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_poly::{
        univariate::{DenseOrSparsePolynomial, DensePolynomial},
        DenseUVPolynomial,
    };
    use ark_std::{test_rng, UniformRand, Zero};

    #[test]
    fn series_inverse() {
        let mut rng = test_rng();
        for k in [1, 2, 3, 65, 200, 513] {
            let g: Vec<Fr> = (0..k + 3).map(|_| Fr::rand(&mut rng)).collect();
            let gh = mul(&g, &inv_series(&g, k));
            assert_eq!(gh[0], Fr::from(1u64));
            assert!(gh[1..k].iter().all(|x| x.is_zero()), "k = {k}");
        }
    }

    #[test]
    fn remainder_matches_long_division() {
        let mut rng = test_rng();
        for (df, dg) in [(10, 3), (129, 64), (300, 150), (1000, 700)] {
            let f: Vec<Fr> = (0..=df).map(|_| Fr::rand(&mut rng)).collect();
            let mut g: Vec<Fr> = (0..dg).map(|_| Fr::rand(&mut rng)).collect();
            g.push(Fr::from(1u64));
            let mut r = rem(&f, &g, &inv_series(&rev(&g), df - dg + 1));
            let (_, mut expected) = DenseOrSparsePolynomial::from(DensePolynomial::from_coefficients_vec(f))
                .divide_with_q_and_r(&DensePolynomial::from_coefficients_vec(g).into())
                .unwrap();
            r.resize(dg, Fr::zero());
            expected.coeffs.resize(dg, Fr::zero());
            assert_eq!(r, expected.coeffs, "deg f = {df}, deg g = {dg}");
        }
    }

    #[test]
    fn cauchy_sums_match_direct() {
        let mut rng = test_rng();
        for m in [1, 2, 3, 7, 33, 64, 65, 100, 257] {
            let xs: Vec<Fr> = (0..m).map(|_| Fr::rand(&mut rng)).collect();
            let c: Vec<Fr> = (0..m).map(|_| Fr::rand(&mut rng)).collect();
            let direct: Vec<Fr> = (0..m)
                .map(|k| {
                    (0..m)
                        .filter(|l| *l != k)
                        .map(|l| c[l] * (xs[k] - xs[l]).inverse().unwrap())
                        .sum()
                })
                .collect();
            assert_eq!(cauchy_sums(&xs, &[&c])[0], direct, "m = {m}");
        }
    }
}
