//! Setup: the Groth16 setup `G`, and deriving the updating key from the
//! prover key.
mod common;

use ark_bls12_381::Bls12_381;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use common::{time, Options, Recorder};
use dynark::{groth16, synthetic_r1cs::generate_matrices, SemiDynark};

type E = Bls12_381;

fn main() {
    let opts = Options::parse("setup", &[12, 14, 16], 3, &[]);
    let mut rec = Recorder::new(&opts, &["log_n", "phase"]);
    for &log_n in &opts.log_n {
        let n = 1 << log_n;
        let mut rng = StdRng::seed_from_u64(log_n as u64);
        let (matrices, ..) = generate_matrices::<E>(n, 1, n - 1, &mut rng);
        for run in 0..opts.repeat {
            let ((pk, _, domain), t) =
                time(|| groth16::setup::<E, _>(matrices.clone(), &mut rng).unwrap());
            rec.record(vec![log_n.to_string(), "groth16_setup".into()], run, t);
            let (_, t) =
                time(|| SemiDynark::<E>::generate_updating_key(&matrices, &domain, &pk).unwrap());
            rec.record(vec![log_n.to_string(), "updating_key".into()], run, t);
        }
        rec.print_progress();
    }
}
