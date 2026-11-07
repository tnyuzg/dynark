//! The Groth16 provers: `groth16_6dft` (6 DFTs, as in snarkjs: the baseline),
//! `groth16_7dft` (7 DFTs, as in arkworks), `faster_groth16` (FasterGroth16,
//! 4 DFTs) and `sparse_groth16` (SparseGroth16).
//!
//! Phase `one_time` is the per-circuit precomputation a prover needs beyond
//! the prover key (`CosetBasis`, `FasterGroth16Key` and `SparseGroth16Key`;
//! `groth16_7dft` needs none). Phase `prove`
//! is one proof.
//!
//! Two workloads: `dense`, the synthetic R1CS of the evaluation, and
//! `sparse`, where only `--nonzero` inputs (default 64) are nonzero.
mod common;

use ark_bls12_381::Bls12_381;
use ark_ec::pairing::Pairing;
use ark_ff::Zero;
use ark_relations::r1cs::ConstraintMatrices;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use common::{single_threaded, time, Arg, Options, Recorder};
use dynark::{
    groth16,
    FasterGroth16, FasterGroth16Key, SparseGroth16, SparseGroth16Key,
    synthetic_r1cs::{generate_instance_witness, generate_matrices, generate_sparse_r1cs},
};

type E = Bls12_381;
type F = <E as Pairing>::ScalarField;

fn main() {
    let opts = Options::parse(
        "provers",
        &[12, 14, 16],
        3,
        &[
            ("--workloads", Arg::Value),
            ("--provers", Arg::Value),
            ("--nonzero", Arg::Value),
            ("--serial", Arg::Switch),
        ],
    );
    let workloads = opts.text("--workloads", "dense,sparse");
    let mut rec = Recorder::new(&opts, &["workload", "prover", "phase", "log_n", "threads"]);
    for &log_n in &opts.log_n {
        let n = 1 << log_n;
        let mut rng = StdRng::seed_from_u64(log_n as u64);
        if workloads.contains("dense") {
            let (matrices, ka, kb, ca, cb) = generate_matrices::<E>(n, 1, n - 1, &mut rng);
            let (instance, witness) = generate_instance_witness::<E>(1, n - 1, &mut rng, &ka, &kb, &ca, &cb);
            run(&opts, &mut rec, log_n, "dense", matrices, instance, witness);
        }
        if workloads.contains("sparse") {
            let nonzero = opts.usize("--nonzero", 64);
            let (matrices, instance, witness) = generate_sparse_r1cs::<E>(n, n / 4, nonzero, &mut rng);
            run(&opts, &mut rec, log_n, "sparse", matrices, instance, witness);
        }
    }
}

fn run(
    opts: &Options,
    rec: &mut Recorder,
    log_n: usize,
    workload: &str,
    matrices: ConstraintMatrices<F>,
    instance: Vec<F>,
    witness: Vec<F>,
) {
    let provers = opts.text("--provers", "groth16_6dft,groth16_7dft,faster_groth16,sparse_groth16");
    let mut rng = StdRng::seed_from_u64(1);
    let (pk, _, domain) = groth16::setup::<E, _>(matrices.clone(), &mut rng).unwrap();
    let faster_key = FasterGroth16Key::<E>::new(&domain, &pk);
    let sparse_key = SparseGroth16Key::<E>::new(&matrices, &domain, &pk);
    let coset_basis = groth16::CosetBasis::new(&domain, &pk.h_query);
    let assignment: Vec<(usize, F)> = instance
        .iter()
        .chain(&witness)
        .enumerate()
        .filter(|(_, z)| !z.is_zero())
        .map(|(j, z)| (j, *z))
        .collect();

    let mut thread_counts = vec![opts.threads];
    if opts.switch("--serial") && opts.threads > 1 {
        thread_counts.push(1);
    }
    for &threads in &thread_counts {
        for run in 0..opts.repeat {
            for prover in provers.split(',') {
                let one_time = || match prover {
                    "groth16_6dft" => drop(groth16::CosetBasis::<E>::new(&domain, &pk.h_query)),
                    "faster_groth16" => drop(FasterGroth16Key::<E>::new(&domain, &pk)),
                    "sparse_groth16" => drop(SparseGroth16Key::<E>::new(&matrices, &domain, &pk)),
                    _ => {},
                };
                if prover != "groth16_7dft" {
                    let t = if threads == 1 { single_threaded(|| time(one_time).1) } else { time(one_time).1 };
                    rec.record(
                        vec![workload.into(), prover.into(), "one_time".into(), log_n.to_string(), threads.to_string()],
                        run,
                        t,
                    );
                }
                let prove = || {
                    let mut rng = StdRng::seed_from_u64(run as u64);
                    match prover {
                        "groth16_6dft" => groth16::prove_6dft(&pk, &coset_basis, &matrices, &instance, &witness, &mut rng),
                        "groth16_7dft" => groth16::prove(&pk, &matrices, &instance, &witness, &mut rng),
                        "faster_groth16" => FasterGroth16::prove(&pk, &faster_key, &matrices, &instance, &witness, &mut rng),
                        "sparse_groth16" => SparseGroth16::prove(&pk, &sparse_key, instance.len(), &assignment, &mut rng),
                        other => panic!("unknown prover {other}"),
                    }
                    .unwrap()
                };
                let t = if threads == 1 { single_threaded(|| time(prove).1) } else { time(prove).1 };
                rec.record(
                    vec![workload.into(), prover.into(), "prove".into(), log_n.to_string(), threads.to_string()],
                    run,
                    t,
                );
            }
        }
        rec.print_progress();
    }
}
