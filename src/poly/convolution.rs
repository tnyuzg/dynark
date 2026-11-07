//! Circular convolutions and elementwise operations over field and `G1`
//! vectors, computed with the arkworks DFT.
use ark_ec::pairing::Pairing;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::SynthesisError;
use ark_std::vec::Vec;
use rayon::prelude::*;

/// Circular convolution of a field vector with a `G1` vector of the same length.
#[inline]
pub fn convolution_f_g<E: Pairing>(a: &[E::ScalarField], b: &[E::G1]) -> Vec<<E as Pairing>::G1> {
    let a_len = a.len();
    let b_len = b.len();
    debug_assert_eq!(a_len, b_len);
    let domain = GeneralEvaluationDomain::<E::ScalarField>::new(a_len)
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)
        .unwrap();
    let a_fft = domain.fft(a);
    let mut b_fft: Vec<E::G1> = domain.fft(b);
    b_fft
        .par_iter_mut()
        .zip(a_fft.par_iter())
        .for_each(|(x_b, x_a)| {
            *x_b *= *x_a;
        });
    domain.ifft_in_place(&mut b_fft);
    b_fft
}

/// Circular convolution of two field vectors of the same length.
pub fn convolution_f_f<E: Pairing>(
    a: &[E::ScalarField],
    b: &[E::ScalarField],
) -> Vec<<E as Pairing>::ScalarField> {
    let a_len = a.len();
    let b_len = b.len();
    debug_assert_eq!(a_len, b_len);
    let domain = GeneralEvaluationDomain::<E::ScalarField>::new(a.len())
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)
        .unwrap();
    let mut a_fft = domain.fft(a);
    let b_fft = domain.fft(b);
    a_fft
        .par_iter_mut()
        .zip(b_fft.par_iter())
        .for_each(|(x_a, x_b)| {
            *x_a *= *x_b;
        });
    domain.ifft_in_place(&mut a_fft);
    a_fft
}

/// Elementwise product of a field vector with a `G1` vector.
#[inline]
pub fn elementwise_product_f_g<E: Pairing>(
    a: &[E::ScalarField],
    b: &[E::G1],
) -> Vec<<E as Pairing>::G1> {
    a.par_iter()
        .zip(b.par_iter())
        .map(|(a, b)| *b * a)
        .collect()
}

/// Elementwise sum of two `G1` vectors.
#[inline]
pub fn elementwise_add_g_g<E: Pairing>(a: &[E::G1], b: &[E::G1]) -> Vec<<E as Pairing>::G1> {
    a.par_iter()
        .zip(b.par_iter())
        .map(|(a, b)| *a + b)
        .collect()
}

/// Circular convolution of a fixed field vector, given by its DFT `w_dft`
/// over `domain`, with a `G1` vector.
pub fn convolution_by_dft_f_g<E: Pairing>(
    domain: &GeneralEvaluationDomain<E::ScalarField>,
    w_dft: &[E::ScalarField],
    b: &[E::G1],
) -> Vec<E::G1> {
    let mut b_fft: Vec<E::G1> = domain.fft(b);
    b_fft
        .par_iter_mut()
        .zip(w_dft.par_iter())
        .for_each(|(x_b, x_w)| *x_b *= *x_w);
    domain.ifft_in_place(&mut b_fft);
    b_fft
}

/// Circular convolution of a fixed field vector, given by its DFT `w_dft`
/// over `domain`, with another field vector.
pub fn convolution_by_dft_f_f<E: Pairing>(
    domain: &GeneralEvaluationDomain<E::ScalarField>,
    w_dft: &[E::ScalarField],
    a: &[E::ScalarField],
) -> Vec<E::ScalarField> {
    let mut a_fft = domain.fft(a);
    a_fft
        .par_iter_mut()
        .zip(w_dft.par_iter())
        .for_each(|(x_a, x_w)| *x_a *= *x_w);
    domain.ifft_in_place(&mut a_fft);
    a_fft
}
