//! The two ways of computing the cross term agree exactly.
use ark_bls12_381::Bls12_381;
use ark_ec::pairing::Pairing;
use ark_std::{
    rand::{rngs::StdRng, seq::SliceRandom, SeedableRng},
    UniformRand,
};
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

#[test]
fn pairwise_matches_sparse_conv() {
    let mut rng = StdRng::seed_from_u64(3);
    let n = 1 << 12;
    let (matrices, ..) = generate_matrices::<E>(n, 1, n - 1, &mut rng);
    let (pk, _, domain) = groth16::setup::<E, _>(matrices, &mut rng).unwrap();
    let basis = QuotientBasis::<E>::new(&domain, &pk.h_query);
    let mut rows: Vec<usize> = (0..n).collect();

    // (|da|, |db|, whether db uses the same rows as da)
    for (da, db, same_rows) in [
        (0, 0, true),
        (1, 1, true),
        (1, 0, false),
        (5, 3, false),
        (64, 64, true),
        (300, 7, true),
        (7, 300, false),
        (500, 500, false),
        (2000, 2000, true),
    ] {
        rows.shuffle(&mut rng);
        let a = sparse(&mut rng, &rows[..da]);
        let b = if same_rows {
            sparse(&mut rng, &rows[..db])
        } else {
            sparse(&mut rng, &rows[da..da + db])
        };
        let pairwise = cross_term_with(&basis, &a, &b, CrossTermMethod::Pairwise);
        let sparse_conv = cross_term_with(&basis, &a, &b, CrossTermMethod::SparseConv);
        assert_eq!(pairwise, sparse_conv, "|da| = {da}, |db| = {db}");
    }
}

#[test]
fn auto_picks_by_cost() {
    let rows = |k: usize| (0..k).map(|i| (i, F::from(1u64))).collect::<Vec<_>>();
    assert_eq!(CrossTermMethod::Auto.resolve(&rows(16), &rows(16)), CrossTermMethod::Pairwise);
    assert_eq!(CrossTermMethod::Auto.resolve(&rows(8192), &rows(8192)), CrossTermMethod::SparseConv);
    // Many accumulated rows against a small update: pairwise is cheaper.
    assert_eq!(CrossTermMethod::Auto.resolve(&rows(16384), &rows(64)), CrossTermMethod::Pairwise);
}
