//! All the Groth16 provers produce the same proof for the same randomness.
use ark_bls12_381::Bls12_381;
use ark_ec::pairing::Pairing;
use ark_ff::{One, Zero};
use ark_relations::r1cs::ConstraintMatrices;
use ark_std::{
    rand::{rngs::StdRng, SeedableRng},
};
use dynark::{
    decomposition::cross::CrossTermMethod,
    groth16,
    FasterGroth16, FasterGroth16Key, SparseGroth16, SparseGroth16Key,
    prepare_verifying_key,
    synthetic_r1cs::{generate_instance_witness, generate_matrices, generate_sparse_r1cs},
};

type E = Bls12_381;
type F = <E as Pairing>::ScalarField;

fn synthetic(n: usize, rng: &mut StdRng) -> (ConstraintMatrices<F>, Vec<F>, Vec<F>) {
    let (matrices, ka, kb, ca, cb) = generate_matrices::<E>(n, 1, n - 1, rng);
    let (instance, witness) = generate_instance_witness::<E>(1, n - 1, rng, &ka, &kb, &ca, &cb);
    (matrices, instance, witness)
}

/// Checks that `prove_6dft`, `FasterGroth16` and `SparseGroth16` give the
/// same proof as `groth16::prove`.
fn check(matrices: ConstraintMatrices<F>, instance: Vec<F>, witness: Vec<F>, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let (pk, vk, domain) = groth16::setup::<E, _>(matrices.clone(), &mut rng).unwrap();
    let faster_key = FasterGroth16Key::new(&domain, &pk);
    let sparse_key = SparseGroth16Key::new(&matrices, &domain, &pk);
    let pvk = prepare_verifying_key(&vk);

    let prove_seed = seed + 1;
    let expected = groth16::prove(&pk, &matrices, &instance, &witness, &mut StdRng::seed_from_u64(prove_seed)).unwrap();
    assert!(groth16::verify(&pvk, &expected.0, &instance).unwrap());

    let coset_basis = groth16::CosetBasis::new(&domain, &pk.h_query);
    let six = groth16::prove_6dft(&pk, &coset_basis, &matrices, &instance, &witness, &mut StdRng::seed_from_u64(prove_seed)).unwrap();
    assert_eq!(six, expected, "prove_6dft");

    let fast = FasterGroth16::prove(&pk, &faster_key, &matrices, &instance, &witness, &mut StdRng::seed_from_u64(prove_seed)).unwrap();
    assert_eq!(fast, expected, "FasterGroth16");

    let assignment: Vec<(usize, F)> = instance
        .iter()
        .chain(&witness)
        .enumerate()
        .filter(|(_, z)| !z.is_zero())
        .map(|(j, z)| (j, *z))
        .collect();
    let sparse = SparseGroth16::prove(&pk, &sparse_key, instance.len(), &assignment, &mut StdRng::seed_from_u64(prove_seed)).unwrap();
    assert_eq!(sparse, expected, "SparseGroth16");
}

#[test]
fn provers_agree_on_synthetic_workload() {
    for log_n in [10, 12] {
        let mut rng = StdRng::seed_from_u64(log_n as u64);
        let (matrices, instance, witness) = synthetic(1 << log_n, &mut rng);
        check(matrices, instance, witness, 10 * log_n as u64);
    }
}

#[test]
fn provers_agree_on_sparse_circuit() {
    let mut rng = StdRng::seed_from_u64(5);
    let (matrices, instance, witness) = generate_sparse_r1cs::<E>(1 << 10, 256, 8, &mut rng);
    check(matrices, instance, witness, 50);
}

#[test]
fn sparse_groth16_agrees_when_it_uses_sparse_convolution() {
    let mut rng = StdRng::seed_from_u64(6);
    // About half of the 2^13 rows are nonzero in A * z and B * z.
    let (matrices, instance, witness) = synthetic(1 << 13, &mut rng);
    let rows: Vec<(usize, F)> = (0..1usize << 12).map(|i| (i, F::one())).collect();
    assert_eq!(CrossTermMethod::Auto.resolve(&rows, &rows), CrossTermMethod::SparseConv);
    check(matrices, instance, witness, 60);
}
