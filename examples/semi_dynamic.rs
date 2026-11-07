//! SemiDynark on a small synthetic circuit: prove once, preprocess the
//! witness, then update the proof after some witness entries change.
//!
//! Run with `cargo run --release --example semi_dynamic`. For timings at
//! larger sizes, see `benches/`.
use ark_bls12_381::Bls12_381;
use ark_serialize::{CanonicalSerialize, Compress};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use dynark::{
    prepare_verifying_key,
    synthetic_r1cs::{generate_instance_witness, generate_matrices, generate_update_once},
    SemiDynark,
};
use std::time::Instant;

type E = Bls12_381;

fn main() {
    let mut rng = StdRng::seed_from_u64(0);
    let n = 1 << 12;
    let num_instance = 1;
    let (matrices, ka, kb, ca, cb) =
        generate_matrices::<E>(n, num_instance, n - num_instance, &mut rng);
    let (instance, witness) = generate_instance_witness::<E>(
        num_instance,
        n - num_instance,
        &mut rng,
        &ka,
        &kb,
        &ca,
        &cb,
    );
    println!("R1CS with n = {n} constraints");

    // G: the Groth16 setup, then the updating key derived from the prover key.
    let start = Instant::now();
    let (pk, vk, domain) = SemiDynark::<E>::setup(matrices.clone(), &mut rng).unwrap();
    let uk = SemiDynark::<E>::generate_updating_key(&matrices, &domain, &pk).unwrap();
    let pvk = prepare_verifying_key(&vk);
    println!("setup:      {:?}", start.elapsed());

    // P: a Groth16 proof, keeping what an update needs.
    let start = Instant::now();
    let (proof, mut data) =
        SemiDynark::<E>::prove(&pk, &matrices, &instance, &witness, &mut rng).unwrap();
    println!("prove:      {:?}", start.elapsed());
    assert!(SemiDynark::<E>::verify(&pvk, &proof, &instance).unwrap());

    // K: preprocess the base witness, once.
    let start = Instant::now();
    SemiDynark::<E>::preprocess(&uk, &matrices, &instance, &witness, &mut data).unwrap();
    println!("preprocess: {:?}", start.elapsed());

    // U: update the proof after d entries of the witness change.
    for d in [1, 16, 256] {
        let (instance_update, witness_update) = generate_update_once::<E>(
            n,
            d,
            num_instance,
            &ka,
            &kb,
            &ca,
            &cb,
            &instance,
            &witness,
            &mut rng,
        );
        let start = Instant::now();
        let updated =
            SemiDynark::<E>::update(&uk, &instance_update, &witness_update, &data, &mut rng)
                .unwrap();
        let elapsed = start.elapsed();
        let ok = SemiDynark::<E>::verify(&pvk, &updated, &instance).unwrap();
        println!("update d = {d:<4} {elapsed:?}, verifies: {ok}");
        assert!(ok);
    }

    println!("proof size: {} bytes", proof.serialized_size(Compress::Yes));
}
