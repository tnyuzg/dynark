//! FullyDynark on a small synthetic circuit: a stream of updates, each applied
//! to the current witness, while the preprocessed data is rebuilt in the
//! background a few steps per update.
//!
//! Run with `cargo run --release --example fully_dynamic`. For timings at
//! larger sizes, see `benches/`.
use ark_bls12_381::Bls12_381;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use dynark::{
    prepare_verifying_key,
    synthetic_r1cs::{generate_instance_witness, generate_matrices, generate_update},
    FullyCheckpoint, FullyDynark, SemiDynark,
};
use std::time::{Duration, Instant};

type E = Bls12_381;

fn main() {
    let mut rng = StdRng::seed_from_u64(0);
    // The stepwise DFT splits the domain into sqrt(n) x sqrt(n), so log n is even.
    let n = 1 << 12;
    let num_instance = 1;
    let (matrices, ka, kb, ca, cb) = generate_matrices::<E>(n, num_instance, n - num_instance, &mut rng);
    let (mut instance, mut witness) =
        generate_instance_witness::<E>(num_instance, n - num_instance, &mut rng, &ka, &kb, &ca, &cb);

    let (pk, vk, domain) = SemiDynark::<E>::setup(matrices.clone(), &mut rng).unwrap();
    let uk = SemiDynark::<E>::generate_updating_key(&matrices, &domain, &pk).unwrap();
    let pvk = prepare_verifying_key(&vk);

    let start = Instant::now();
    let mut prover = FullyDynark::new(FullyCheckpoint::<E>::new(n, &uk, &matrices, &instance, &witness));
    println!("n = {n}: initial checkpoint in {:?}", start.elapsed());

    let d = 16;
    let updates = 200;
    let (mut total, mut rebuilds) = (Duration::ZERO, 0);
    for i in 0..updates {
        // Change d entries of the current witness, and keep the witness in sync.
        let (instance_update, witness_update) = generate_update::<E>(
            n, d, num_instance, &ka, &kb, &ca, &cb, &mut instance, &mut witness, &mut rng,
        );
        let start = Instant::now();
        let progress = prover.advance_rebuild(&uk, d);
        let proof = prover.apply_update(&uk, &instance_update, &witness_update, &mut rng);
        total += start.elapsed();
        if progress.rebuilds_completed > 0 {
            rebuilds += progress.rebuilds_completed;
            println!("update {i:>3}: a rebuild finished, the checkpoints swapped");
        }
        assert!(SemiDynark::<E>::verify(&pvk, &proof, &instance).unwrap());
    }
    println!(
        "{updates} updates of d = {d}: {:?} on average, {rebuilds} rebuilds; every proof verified",
        total / updates
    );
}
