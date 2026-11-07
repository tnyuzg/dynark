//! The three ways of precomputing the projection data agree exactly: in one
//! go, stepwise, and with the Feist-Khovratovich baseline.
use ark_bls12_381::Bls12_381;
use ark_ec::pairing::Pairing;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::{
    rand::{rngs::StdRng, SeedableRng},
    UniformRand,
};
use dynark::{
    decomposition::projection::{
        precompute_projection, precompute_projection_fk, FkPrecomputation, StepwiseProjection,
    },
    groth16,
    synthetic_r1cs::generate_matrices,
    QuotientBasis,
};

type E = Bls12_381;
type F = <E as Pairing>::ScalarField;

fn setup(log_n: usize) -> (GeneralEvaluationDomain<F>, dynark::ProvingKey<E>, Vec<F>, Vec<F>) {
    let mut rng = StdRng::seed_from_u64(log_n as u64);
    let n = 1 << log_n;
    let (matrices, ..) = generate_matrices::<E>(n, 1, n - 1, &mut rng);
    let (pk, _, domain) = groth16::setup::<E, _>(matrices, &mut rng).unwrap();
    let a = (0..n).map(|_| F::rand(&mut rng)).collect();
    let b = (0..n).map(|_| F::rand(&mut rng)).collect();
    (domain, pk, a, b)
}

#[test]
fn one_shot_matches_stepwise() {
    // The stepwise DFT splits the domain into sqrt(n) x sqrt(n), so log n is even.
    for log_n in [10, 12] {
        let (domain, pk, a, b) = setup(log_n);
        let basis = QuotientBasis::<E>::new(&domain, &pk.h_query);
        let expected = precompute_projection(&basis, domain, &a, &b);

        let n = domain.size();
        let mut stepwise = StepwiseProjection::<E>::new(n);
        let (mut q_a, mut q_b) = (vec![Default::default(); n], vec![Default::default(); n]);
        let mut steps = 0;
        while !stepwise.step(&basis, &a, &b, &mut q_a, &mut q_b, 3).0 {
            steps += 1;
            assert!(steps < 1_000_000, "stepwise projection did not finish");
        }
        assert_eq!((q_a, q_b), expected, "log n = {log_n}");
    }
}

#[test]
fn feist_khovratovich_matches_ours() {
    for log_n in [4, 9, 10] {
        let (domain, pk, a, b) = setup(log_n);
        let basis = QuotientBasis::<E>::new(&domain, &pk.h_query);
        let fk = FkPrecomputation::<E>::new(&domain, &pk.h_query);
        assert_eq!(
            precompute_projection_fk(&fk, domain, &a, &b),
            precompute_projection(&basis, domain, &a, &b),
            "log n = {log_n}"
        );
    }
}
