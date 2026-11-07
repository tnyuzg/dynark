//! The quotient decomposition behind the quotient basis (appendix "Quotient
//! Decomposition" of the paper), checked on the group elements the code uses.
//!
//! With `l_i` the Lagrange polynomials of the domain, `lbar_i` the truncated
//! ones and `Z` the vanishing polynomial, for `i != j`
//!
//! ```text
//! l_i * l_j = Z * phi_ij,   phi_ij = w[j - i] * lbar_i + w[i - j] * lbar_j
//! ```
//!
//! and `l_i^2 - l_i = Z * phi_ii`. The basis holds `ell_bar[i] = Z(tau) * lbar_i(tau) / delta`
//! and `diag[i] = Z(tau) * phi_ii(tau) / delta` in G1, so with `g` the
//! generator of G1:
//!
//! ```text
//! g * (l_i(tau) * l_j(tau) / delta) = w[j - i] * ell_bar[i] + w[i - j] * ell_bar[j]
//! g * ((l_i(tau)^2 - l_i(tau)) / delta) = diag[i]
//! ```
use ark_bls12_381::Bls12_381;
use ark_ec::{pairing::Pairing, CurveGroup};
use ark_ff::{Field, UniformRand};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use dynark::{groth16, synthetic_r1cs::generate_matrices, QuotientBasis};

type E = Bls12_381;
type F = <E as Pairing>::ScalarField;
type G1 = <E as Pairing>::G1;
type G2 = <E as Pairing>::G2;

#[test]
fn quotient_decomposition_holds() {
    let mut rng = StdRng::seed_from_u64(0);
    for log_n in [2, 3, 6] {
        let n = 1usize << log_n;
        let (matrices, ..) = generate_matrices::<E>(n, 1, n - 1, &mut rng);
        let domain = GeneralEvaluationDomain::<F>::new(n).unwrap();

        // A prover key from known trapdoors.
        let tau = domain.sample_element_outside_domain(&mut rng);
        let delta = F::rand(&mut rng);
        let (alpha, beta, gamma) = (F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng));
        let (g1, g2) = (G1::rand(&mut rng), G2::rand(&mut rng));
        let pk = groth16::generate_prover_key_with_matrices::<E>(
            domain,
            &matrices,
            (
                tau,
                alpha,
                beta,
                gamma,
                delta,
                g1.into_affine(),
                g2.into_affine(),
            ),
        )
        .unwrap();
        let basis = QuotientBasis::<E>::new(&domain, &pk.h_query);

        // l_i(tau) for every i: the inverse DFT of the powers of tau.
        let powers: Vec<F> = (0..n).map(|k| tau.pow([k as u64])).collect();
        let l = domain.ifft(&powers);
        let delta_inv = delta.inverse().unwrap();

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    let expected = g1 * ((l[i] * l[i] - l[i]) * delta_inv);
                    assert_eq!(basis.diag[i], expected, "n = {n}, i = {i}");
                } else {
                    let expected = g1 * (l[i] * l[j] * delta_inv);
                    let decomposition = basis.ell_bar[i] * basis.w[(n + j - i) % n]
                        + basis.ell_bar[j] * basis.w[(n + i - j) % n];
                    assert_eq!(decomposition, expected, "n = {n}, i = {i}, j = {j}");
                }
            }
        }
    }
}
