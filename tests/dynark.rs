//! End-to-end checks of the semi-dynamic and fully-dynamic update algorithms
//! on the synthetic workload.
use ark_bls12_381::Bls12_381;
use ark_ec::pairing::Pairing;
use ark_ff::{One, UniformRand, Zero};
use dynark::{
    prepare_verifying_key,
    synthetic_r1cs::{
        generate_instance_witness, generate_matrices, generate_sparse_r1cs, generate_update,
        generate_update_once,
    },
    FullyCheckpoint, FullyDynark, SemiDynark,
};
use ark_std::rand::{rngs::StdRng, SeedableRng};

type E = Bls12_381;
type F = <E as Pairing>::ScalarField;

const LOG_N: usize = 10;

#[test]
fn semi_dynamic_update_verifies() {
    let mut rng = StdRng::seed_from_u64(1);
    let n = 1 << LOG_N;
    let (num_instance, num_witness) = (1, n - 1);
    let (matrices, ka, kb, ca, cb) = generate_matrices::<E>(n, num_instance, num_witness, &mut rng);
    let (instance, witness) =
        generate_instance_witness::<E>(num_instance, num_witness, &mut rng, &ka, &kb, &ca, &cb);

    let (pk, vk, domain) = SemiDynark::<E>::setup(matrices.clone(), &mut rng).unwrap();
    let uk = SemiDynark::<E>::generate_updating_key(&matrices, &domain, &pk).unwrap();
    let pvk = prepare_verifying_key(&vk);

    let (proof, mut data) =
        SemiDynark::<E>::prove(&pk, &matrices, &instance, &witness, &mut rng).unwrap();
    assert!(SemiDynark::<E>::verify(&pvk, &proof, &instance).unwrap());
    SemiDynark::<E>::preprocess(&uk, &matrices, &instance, &witness, &mut data).unwrap();

    let (instance_update, mut witness_update) = generate_update_once::<E>(
        n, 16, num_instance, &ka, &kb, &ca, &cb, &instance, &witness, &mut rng,
    );
    let updated = SemiDynark::<E>::update(&uk, &instance_update, &witness_update, &data, &mut rng).unwrap();
    assert!(SemiDynark::<E>::verify(&pvk, &updated, &instance).unwrap());

    // An update that breaks a constraint must not produce a valid proof.
    witness_update[0].1 += F::one();
    let tampered = SemiDynark::<E>::update(&uk, &instance_update, &witness_update, &data, &mut rng).unwrap();
    assert!(!SemiDynark::<E>::verify(&pvk, &tampered, &instance).unwrap());
}

#[test]
fn fully_dynamic_updates_verify() {
    let mut rng = StdRng::seed_from_u64(2);
    let n = 1 << LOG_N;
    let (num_instance, num_witness) = (1, n - 1);
    let (matrices, ka, kb, ca, cb) = generate_matrices::<E>(n, num_instance, num_witness, &mut rng);
    let (mut instance, mut witness) =
        generate_instance_witness::<E>(num_instance, num_witness, &mut rng, &ka, &kb, &ca, &cb);

    let (pk, vk, domain) = SemiDynark::<E>::setup(matrices.clone(), &mut rng).unwrap();
    let uk = SemiDynark::<E>::generate_updating_key(&matrices, &domain, &pk).unwrap();
    let pvk = prepare_verifying_key(&vk);

    let checkpoint = FullyCheckpoint::<E>::new(n, &uk, &matrices, &instance, &witness);
    let mut prover = FullyDynark::new(checkpoint);

    let num_update = 16;
    let mut rebuilds = 0;
    for _ in 0..40 {
        let (instance_update, witness_update) = generate_update::<E>(
            n, num_update, num_instance, &ka, &kb, &ca, &cb, &mut instance, &mut witness, &mut rng,
        );
        rebuilds += prover.advance_rebuild(&uk, num_update).rebuilds_completed;
        let proof = prover.apply_update(&uk, &instance_update, &witness_update, &mut rng);
        assert!(SemiDynark::<E>::verify(&pvk, &proof, &instance).unwrap());
    }
    // Enough updates for the rebuilt checkpoint to take over at least twice.
    assert!(rebuilds >= 2, "only {rebuilds} rebuilds happened");

    // An update that breaks a constraint must not produce a valid proof.
    let (instance_update, mut witness_update) = generate_update::<E>(
        n, num_update, num_instance, &ka, &kb, &ca, &cb, &mut instance, &mut witness, &mut rng,
    );
    witness_update[0].1 += F::one();
    let tampered = prover.update(&uk, &instance_update, &witness_update, num_update, &mut rng);
    assert!(!SemiDynark::<E>::verify(&pvk, &tampered, &instance).unwrap());
}

#[test]
fn updated_proofs_are_freshly_blinded() {
    let mut rng = StdRng::seed_from_u64(3);
    let n = 1 << LOG_N;
    let (num_instance, num_witness) = (1, n - 1);
    let (matrices, ka, kb, ca, cb) = generate_matrices::<E>(n, num_instance, num_witness, &mut rng);
    let (instance, witness) =
        generate_instance_witness::<E>(num_instance, num_witness, &mut rng, &ka, &kb, &ca, &cb);
    let (pk, vk, domain) = SemiDynark::<E>::setup(matrices.clone(), &mut rng).unwrap();
    let uk = SemiDynark::<E>::generate_updating_key(&matrices, &domain, &pk).unwrap();
    let pvk = prepare_verifying_key(&vk);
    let (_, mut data) = SemiDynark::<E>::prove(&pk, &matrices, &instance, &witness, &mut rng).unwrap();
    SemiDynark::<E>::preprocess(&uk, &matrices, &instance, &witness, &mut data).unwrap();

    // The same update twice: both proofs verify, but they differ.
    let (instance_update, witness_update) =
        generate_update_once::<E>(n, 4, num_instance, &ka, &kb, &ca, &cb, &instance, &witness, &mut rng);
    let first = SemiDynark::<E>::update(&uk, &instance_update, &witness_update, &data, &mut rng).unwrap();
    let second = SemiDynark::<E>::update(&uk, &instance_update, &witness_update, &data, &mut rng).unwrap();
    assert!(SemiDynark::<E>::verify(&pvk, &first, &instance).unwrap());
    assert!(SemiDynark::<E>::verify(&pvk, &second, &instance).unwrap());
    assert_ne!(first, second);

    // Likewise for FullyDynark, from two copies of the same prover.
    let mut prover = FullyDynark::new(FullyCheckpoint::<E>::new(n, &uk, &matrices, &instance, &witness));
    let mut copy = prover.clone();
    let first = prover.update(&uk, &instance_update, &witness_update, 4, &mut rng);
    let second = copy.update(&uk, &instance_update, &witness_update, 4, &mut rng);
    assert!(SemiDynark::<E>::verify(&pvk, &first, &instance).unwrap());
    assert!(SemiDynark::<E>::verify(&pvk, &second, &instance).unwrap());
    assert_ne!(first, second);
}

#[test]
fn semi_dynamic_update_with_more_variables_than_constraints() {
    let mut rng = StdRng::seed_from_u64(4);
    let (n, num_inputs) = (1 << 8, 64);
    let (matrices, instance, witness) = generate_sparse_r1cs::<E>(n, num_inputs, 16, &mut rng);
    assert!(matrices.num_instance_variables + matrices.num_witness_variables > n);

    let (pk, vk, domain) = SemiDynark::<E>::setup(matrices.clone(), &mut rng).unwrap();
    let uk = SemiDynark::<E>::generate_updating_key(&matrices, &domain, &pk).unwrap();
    let pvk = prepare_verifying_key(&vk);
    let (_, mut data) = SemiDynark::<E>::prove(&pk, &matrices, &instance, &witness, &mut rng).unwrap();
    SemiDynark::<E>::preprocess(&uk, &matrices, &instance, &witness, &mut data).unwrap();

    // Change one nonzero input, and the outputs that depend on it: each row r
    // is z_x * z_y = z_out with out = 1 + num_inputs + r.
    let z: Vec<F> = [&instance[..], &witness[..]].concat();
    let mut z_new = z.clone();
    let input = (1..=num_inputs).find(|&j| !z[j].is_zero()).unwrap();
    z_new[input] += F::rand(&mut rng);
    for row in 0..n {
        let (x, y) = (matrices.a[row][0].1, matrices.b[row][0].1);
        z_new[1 + num_inputs + row] = z_new[x] * z_new[y];
    }
    let witness_update: Vec<(usize, F)> = (1..z.len())
        .filter(|&j| z_new[j] != z[j])
        .map(|j| (j, z_new[j] - z[j]))
        .collect();

    let updated = SemiDynark::<E>::update(&uk, &[], &witness_update, &data, &mut rng).unwrap();
    assert!(SemiDynark::<E>::verify(&pvk, &updated, &instance).unwrap());
}
