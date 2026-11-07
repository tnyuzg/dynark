//! The cross term of an update, with its inner sums computed pairwise, by
//! sparse convolution, or by whichever `CrossTermMethod::Auto` picks.
//!
//! Two shapes: `semi` changes the same `d` rows in `A * z` and `B * z`, as one
//! SemiDynark update does; `fully` crosses `acc` accumulated rows with `d` new
//! ones, as FullyDynark does. Uses the first `--log-n` (default 15).
mod common;

use ark_bls12_381::Bls12_381;
use ark_ec::pairing::Pairing;
use ark_std::{
    rand::{rngs::StdRng, seq::SliceRandom, SeedableRng},
    UniformRand,
};
use common::{time, Arg, Options, Recorder};
use dynark::{
    decomposition::cross::{cross_term_with, CrossTermMethod},
    groth16,
    synthetic_r1cs::generate_matrices,
    QuotientBasis,
};

type E = Bls12_381;
type F = <E as Pairing>::ScalarField;

fn sparse(rng: &mut StdRng, rows: &[usize]) -> Vec<(usize, F)> {
    let mut v: Vec<(usize, F)> = rows.iter().map(|i| (*i, F::rand(rng))).collect();
    v.sort_by_key(|x| x.0);
    v
}

fn main() {
    let opts = Options::parse(
        "cross_term",
        &[15],
        3,
        &[("--log-d", Arg::Value), ("--log-acc", Arg::Value)],
    );
    let log_n = opts.log_n[0];
    let n = 1usize << log_n;
    let log_d = opts.list("--log-d", &(0..=13.min(log_n - 2)).collect::<Vec<_>>());
    let log_acc = opts.list("--log-acc", &[11, 13]);

    let mut rng = StdRng::seed_from_u64(7);
    let (matrices, ..) = generate_matrices::<E>(n, 1, n - 1, &mut rng);
    let (pk, _, domain) = groth16::setup::<E, _>(matrices, &mut rng).unwrap();
    let basis = QuotientBasis::<E>::new(&domain, &pk.h_query);
    let mut rows: Vec<usize> = (0..n).collect();

    let mut shapes: Vec<(&str, usize, usize)> = vec![];
    for &ld in &log_d {
        shapes.push(("semi", 1 << ld, 1 << ld));
    }
    for &la in &log_acc {
        for &ld in log_d.iter().filter(|&&ld| ld <= la) {
            shapes.push(("fully", 1 << la, 1 << ld));
        }
    }

    let mut rec = Recorder::new(
        &opts,
        &["shape", "rows_a", "rows_b", "method", "log_n", "threads"],
    );
    for (shape, rows_a, rows_b) in shapes {
        if rows_a > n {
            continue;
        }
        rows.shuffle(&mut rng);
        let a = sparse(&mut rng, &rows[..rows_a]);
        let b = sparse(&mut rng, &rows[..rows_b]);
        for run in 0..opts.repeat {
            for (name, method) in [
                ("pairwise", CrossTermMethod::Pairwise),
                ("sparse_conv", CrossTermMethod::SparseConv),
                ("auto", CrossTermMethod::Auto),
            ] {
                let (_, t) = time(|| cross_term_with(&basis, &a, &b, method));
                let row = vec![
                    shape.into(),
                    rows_a.to_string(),
                    rows_b.to_string(),
                    name.into(),
                    log_n.to_string(),
                    opts.threads.to_string(),
                ];
                rec.record(row, run, t);
            }
        }
        rec.print_progress();
    }
}
