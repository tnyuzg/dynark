//! Preprocessing `K`: the projection data of a witness, precomputed our way
//! (circular convolutions) and with the Feist-Khovratovich baseline (linear
//! convolutions), in two phases:
//!
//! - `one_time`: depends only on the circuit (ours: the quotient basis; FK:
//!   the DFT of the padded `h_query`).
//! - `per_witness`: the Lagrange coefficients of `A * z` and `B * z` and their
//!   projection data, i.e. `K` itself.
mod common;

use ark_bls12_381::Bls12_381;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use common::{single_threaded, time, Arg, Options, Recorder};
use dynark::{
    decomposition::projection::{precompute_projection, precompute_projection_fk, FkPrecomputation},
    groth16::{self, qap::lagrange_of_dense_vec},
    synthetic_r1cs::{generate_instance_witness, generate_matrices},
    QuotientBasis,
};

type E = Bls12_381;

fn main() {
    let opts = Options::parse(
        "preprocessing",
        &[12, 14, 16],
        3,
        &[("--serial-max-log-n", Arg::Value), ("--methods", Arg::Value)],
    );
    // Also measure single-threaded, up to this size (0: never).
    let serial_max_log_n = opts.usize("--serial-max-log-n", 0);
    let methods = opts.text("--methods", "ours,fk");
    let mut rec = Recorder::new(&opts, &["method", "phase", "log_n", "threads"]);

    for &log_n in &opts.log_n {
        let n = 1 << log_n;
        let mut rng = StdRng::seed_from_u64(log_n as u64);
        let (matrices, ka, kb, ca, cb) = generate_matrices::<E>(n, 1, n - 1, &mut rng);
        let (instance, witness) = generate_instance_witness::<E>(1, n - 1, &mut rng, &ka, &kb, &ca, &cb);
        let z = [&instance[..], &witness[..]].concat();
        let (pk, _, domain) = groth16::setup::<E, _>(matrices.clone(), &mut rng).unwrap();
        let basis = QuotientBasis::<E>::new(&domain, &pk.h_query);
        let fk = FkPrecomputation::<E>::new(&domain, &pk.h_query);

        let ours = || {
            let (a, b, _) = lagrange_of_dense_vec(&matrices, &z).unwrap();
            precompute_projection(&basis, domain, &a, &b)
        };
        let feist_khovratovich = || {
            let (a, b, _) = lagrange_of_dense_vec(&matrices, &z).unwrap();
            precompute_projection_fk(&fk, domain, &a, &b)
        };
        assert_eq!(ours(), feist_khovratovich(), "the two methods disagree");

        let mut thread_counts = vec![opts.threads];
        if log_n <= serial_max_log_n && opts.threads > 1 {
            thread_counts.push(1);
        }
        for &threads in &thread_counts {
            let row = |method: &str, phase: &str| {
                vec![method.into(), phase.into(), log_n.to_string(), threads.to_string()]
            };
            let measure = |f: &(dyn Fn() + Sync)| {
                if threads == 1 {
                    single_threaded(|| time(f).1)
                } else {
                    time(f).1
                }
            };
            for run in 0..opts.repeat {
                if methods.contains("ours") {
                    let t = measure(&|| drop(QuotientBasis::<E>::new(&domain, &pk.h_query)));
                    rec.record(row("ours", "one_time"), run, t);
                    let t = measure(&|| drop(ours()));
                    rec.record(row("ours", "per_witness"), run, t);
                }
                if methods.contains("fk") {
                    let t = measure(&|| drop(FkPrecomputation::<E>::new(&domain, &pk.h_query)));
                    rec.record(row("fk", "one_time"), run, t);
                    let t = measure(&|| drop(feist_khovratovich()));
                    rec.record(row("fk", "per_witness"), run, t);
                }
            }
            rec.print_progress();
        }
    }
}
