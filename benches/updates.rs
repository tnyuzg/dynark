//! Update time: SemiDynark and FullyDynark updates of `d` entries, against
//! proving from scratch with Groth16, over a grid of `n` and `d`.
//!
//! - `groth16`: proving the whole witness from scratch with the 6-DFT prover,
//!   `groth16::prove_6dft` (phase `prove`), and verifying the proof (phase
//!   `verify`). Its one-time `CosetBasis` is not timed here; see `provers`.
//! - `semi`: one update of `d` entries from the base witness, `--repeat` times.
//! - `fully`: consecutive updates of `d` entries. By default the number of
//!   updates is about one rebuild cycle, `max(repeat, 2^((log n + 6) / 2) / d)`,
//!   so the mean includes the cost of rebuilding. Each update is recorded as
//!   `update`, and split into `rebuild` (the background rebuild steps) and
//!   `apply` (the rest). Needs even `log n`.
//!
//! With `--serial`, `groth16` and `semi` are also measured single-threaded,
//! and for `fully` one random update in every `--serial-every` (default 32)
//! runs single-threaded.
mod common;

use ark_bls12_381::Bls12_381;
use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
use common::{single_threaded, time, Arg, Options, Recorder};
use dynark::{
    groth16, prepare_verifying_key,
    synthetic_r1cs::{generate_instance_witness, generate_matrices, generate_update, generate_update_once},
    FullyCheckpoint, FullyDynark, SemiDynark,
};

type E = Bls12_381;

fn main() {
    let opts = Options::parse(
        "updates",
        &[10, 12, 14, 16],
        3,
        &[
            ("--log-d", Arg::Value),
            ("--schemes", Arg::Value),
            ("--fully-runs", Arg::Value),
            ("--half-budget", Arg::Switch),
            ("--serial", Arg::Switch),
            ("--serial-every", Arg::Value),
        ],
    );
    let log_d = opts.list("--log-d", &[0]);
    let schemes = opts.text("--schemes", "groth16,semi,fully");
    let serial = opts.switch("--serial") && opts.threads > 1;
    let serial_every = opts.usize("--serial-every", 32).max(1);
    let threads = opts.threads.to_string();
    let mut rec = Recorder::new(&opts, &["scheme", "log_n", "d", "threads", "phase"]);

    for &log_n in &opts.log_n {
        let n = 1 << log_n;
        let num_instance = 1;
        let mut rng = StdRng::seed_from_u64(log_n as u64);
        let (matrices, ka, kb, ca, cb) = generate_matrices::<E>(n, num_instance, n - num_instance, &mut rng);
        let (mut instance, mut witness) =
            generate_instance_witness::<E>(num_instance, n - num_instance, &mut rng, &ka, &kb, &ca, &cb);
        let (pk, vk, domain) = SemiDynark::<E>::setup(matrices.clone(), &mut rng).unwrap();
        let pvk = prepare_verifying_key(&vk);
        let uk = SemiDynark::<E>::generate_updating_key(&matrices, &domain, &pk).unwrap();
        let coset_basis = groth16::CosetBasis::new(&domain, &pk.h_query);
        let (_, mut data) = SemiDynark::<E>::prove(&pk, &matrices, &instance, &witness, &mut rng).unwrap();
        SemiDynark::<E>::preprocess(&uk, &matrices, &instance, &witness, &mut data).unwrap();
        let row = |scheme: &str, d: &str, threads: &str, phase: &str| {
            vec![scheme.into(), log_n.to_string(), d.into(), threads.into(), phase.into()]
        };

        if schemes.contains("groth16") {
            for run in 0..opts.repeat {
                let prove = || {
                    let mut rng = StdRng::seed_from_u64(run as u64);
                    groth16::prove_6dft(&pk, &coset_basis, &matrices, &instance, &witness, &mut rng).unwrap()
                };
                let ((proof, _), t) = time(prove);
                rec.record(row("groth16", "", &threads, "prove"), run, t);
                let (ok, t) = time(|| groth16::verify(&pvk, &proof, &instance).unwrap());
                assert!(ok, "Groth16 proof rejected");
                rec.record(row("groth16", "", &threads, "verify"), run, t);
                if serial {
                    let (_, t) = single_threaded(|| time(prove));
                    rec.record(row("groth16", "", "1", "prove"), run, t);
                }
            }
            rec.print_progress();
        }

        let mut fully = (schemes.contains("fully") && log_n % 2 == 0).then(|| {
            FullyDynark::new(FullyCheckpoint::<E>::new_with_cached_quotient(
                n, &uk, &matrices, &instance, &witness, &data.q_a, &data.q_b,
            ))
        });
        if schemes.contains("fully") && fully.is_none() {
            println!("skipping fully for log n = {log_n}: the stepwise DFT needs even log n");
        }

        for &ld in &log_d {
            let d = 1usize << ld;
            if d >= n / 2 {
                println!("skipping d = {d} for log n = {log_n}: d must be less than n / 2");
                continue;
            }
            let d_col = d.to_string();

            if schemes.contains("semi") {
                for run in 0..opts.repeat {
                    let (instance_update, witness_update) = generate_update_once::<E>(
                        n, d, num_instance, &ka, &kb, &ca, &cb, &instance, &witness, &mut rng,
                    );
                    let update = || {
                        let mut rng = StdRng::seed_from_u64(run as u64);
                        SemiDynark::<E>::update(&uk, &instance_update, &witness_update, &data, &mut rng).unwrap()
                    };
                    let (proof, t) = time(update);
                    assert!(SemiDynark::<E>::verify(&pvk, &proof, &instance).unwrap(), "semi proof rejected");
                    rec.record(row("semi", &d_col, &threads, "update"), run, t);
                    if serial {
                        let (_, t) = single_threaded(|| time(update));
                        rec.record(row("semi", &d_col, "1", "update"), run, t);
                    }
                }
                rec.print_progress();
            }

            if let Some(prover) = fully.as_mut() {
                let runs = opts.usize("--fully-runs", opts.repeat.max((1 << ((log_n + 6) / 2)) / d));
                let budget = if opts.switch("--half-budget") { (d + 1) / 2 } else { d };
                let mut serial_run = rng.gen_range(0..serial_every);
                for run in 0..runs {
                    if run % serial_every == 0 && run > 0 {
                        serial_run = run + rng.gen_range(0..serial_every);
                    }
                    let (instance_update, witness_update) = generate_update::<E>(
                        n, d, num_instance, &ka, &kb, &ca, &cb, &mut instance, &mut witness, &mut rng,
                    );
                    let single = serial && run == serial_run;
                    let mut update = || {
                        let (_, rebuild) = time(|| prover.advance_rebuild(&uk, budget));
                        let mut blind_rng = StdRng::seed_from_u64(run as u64);
                        let (proof, apply) = time(|| prover.apply_update(&uk, &instance_update, &witness_update, &mut blind_rng));
                        (proof, rebuild, apply)
                    };
                    let (proof, rebuild, apply) = if single { single_threaded(update) } else { update() };
                    assert!(SemiDynark::<E>::verify(&pvk, &proof, &instance).unwrap(), "fully proof rejected");
                    let threads_col = if single { "1" } else { threads.as_str() };
                    rec.record(row("fully", &d_col, threads_col, "update"), run, rebuild + apply);
                    rec.record(row("fully", &d_col, threads_col, "rebuild"), run, rebuild);
                    rec.record(row("fully", &d_col, threads_col, "apply"), run, apply);
                }
                rec.print_progress();
            }
        }
    }
}
