//! The projection term of an update (Section "Projection Terms" of the
//! paper), and the per-witness data it needs.
//!
//! The projection term pairs a change on one side with the other side's
//! projection data (`q_a`, `q_b`; the paper's projection quotients), at one
//! group operation per change: [`projection_term`]. Precomputing the
//! projection data is the preprocessing `K`:
//!
//! - [`precompute_projection`]: in one go, with circular convolutions.
//! - [`StepwiseProjection`]: the same, a few steps at a time, so that
//!   FullyDynark can spread the cost across updates (appendix "Stepwise DFT
//!   and Stepwise Preprocessing").
//! - [`precompute_projection_fk`]: the same result with the method of Feist
//!   and Khovratovich, as a baseline.
use crate::{
    decomposition::QuotientBasis,
    poly::{
        convolution::{
            convolution_by_dft_f_f, convolution_by_dft_f_g, elementwise_add_g_g,
            elementwise_product_f_g,
        },
        stepwise_dft::{sqrt_fft_fw_field, sqrt_fft_fw_g1, CustomParaFFT, SqrtFftStatus},
    },
};
use ark_ec::{pairing::Pairing, AffineRepr};
use ark_ff::Zero;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::SynthesisError;
use ark_std::vec::Vec;
use rayon::{join, prelude::*};

/// The projection data `(q_a, q_b)` for the Lagrange coefficients `a` and `b`
/// of `A * z` and `B * z`, in one go. With `conv` the circular
/// convolution and `.*` the elementwise product:
///
/// ```text
/// q_a = conv(w, a .* ell_bar) + conv(w_prime, a) .* ell_bar + a .* diag
/// ```
///
/// and likewise for `q_b`.
#[inline]
pub fn precompute_projection<E: Pairing>(
    basis: &QuotientBasis<E>,
    domain: GeneralEvaluationDomain<E::ScalarField>,
    a: &[E::ScalarField],
    b: &[E::ScalarField],
) -> (Vec<E::G1>, Vec<E::G1>) {
    let n = domain.size();
    debug_assert_eq!(a.len(), n);
    debug_assert_eq!(b.len(), n);
    let quotients = |x: &[E::ScalarField]| {
        let first = convolution_by_dft_f_g::<E>(
            &domain,
            &basis.w_dft,
            &elementwise_product_f_g::<E>(x, &basis.ell_bar),
        );
        let second = elementwise_product_f_g::<E>(
            &convolution_by_dft_f_f::<E>(&domain, &basis.w_prime_dft, x),
            &basis.ell_bar,
        );
        let third = elementwise_product_f_g::<E>(x, &basis.diag);
        elementwise_add_g_g::<E>(&elementwise_add_g_g::<E>(&first, &second), &third)
    };
    (quotients(a), quotients(b))
}

/// A projection term: `sum_i delta_i * q[i]` over a sparse change `delta` to
/// the Lagrange coefficients of one side, with `q` the projection data of the
/// other side. One group operation per change.
#[inline]
pub fn projection_term<E: Pairing>(q: &[E::G1], delta: &[(usize, E::ScalarField)]) -> E::G1 {
    let (sparse_vec, dense_vec) = (delta, q);
    if sparse_vec.len() < 5 {
        serial_projection_term::<E>(sparse_vec, dense_vec)
    } else {
        sparse_vec
            .par_iter()
            .map(|(idx, coeff)| {
                let val = dense_vec[*idx];
                val * coeff
            })
            .reduce(|| E::G1::zero(), |a, b| a + b)
    }
}

#[inline]
fn serial_projection_term<E: Pairing>(
    sparse_vec: &[(usize, E::ScalarField)],
    dense_vec: &[E::G1],
) -> E::G1 {
    let mut sum = E::G1::zero();
    for (idx, coeff) in sparse_vec {
        let val = dense_vec[*idx];
        sum += val * coeff;
    }
    sum
}

/// One-time phase of the Feist-Khovratovich baseline (`feist2023fast` in the
/// paper): the DFT, over a domain of size `2n`, of the prover key's `h_query`
/// reversed and zero-padded. It depends only on the circuit, like the
/// [`QuotientBasis`] that [`precompute_projection`] uses.
#[derive(Clone, Debug, PartialEq)]
pub struct FkPrecomputation<E: Pairing> {
    /// The domain of size `2n`.
    domain2: GeneralEvaluationDomain<E::ScalarField>,
    /// The DFT of `h_query` reversed and zero-padded to `2n`.
    h_rev_dft: Vec<E::G1>,
}

impl<E: Pairing> FkPrecomputation<E> {
    /// Precomputes for a domain of size `n` from `h_query`, the elements
    /// `tau^i * Z_H(tau) / delta` in G1 for `i < n - 1`.
    pub fn new(
        domain: &GeneralEvaluationDomain<E::ScalarField>,
        h_query: &[E::G1Affine],
    ) -> Self {
        let n = domain.size();
        let domain2 = GeneralEvaluationDomain::<E::ScalarField>::new(2 * n)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)
            .unwrap();
        let mut h_rev: Vec<E::G1> = h_query.iter().rev().map(|h| h.into_group()).collect();
        h_rev.resize(2 * n, E::G1::zero());
        domain2.fft_in_place(&mut h_rev);
        Self { domain2, h_rev_dft: h_rev }
    }
}

/// The projection data `(q_a, q_b)`, as [`precompute_projection`] computes it,
/// but with the method of Feist and Khovratovich: the witness-dependent
/// phase of the baseline.
///
/// Entry `i` of `q_a` is `omega^i / n` times the KZG
/// opening quotient `(A(X) - A(omega^i)) / (X - omega^i)` in G1. All `n`
/// openings are one Toeplitz matrix-vector product with `h_query`, computed as
/// a linear convolution of size `2n`, followed by a DFT of size `n`.
pub fn precompute_projection_fk<E: Pairing>(
    pre: &FkPrecomputation<E>,
    domain: GeneralEvaluationDomain<E::ScalarField>,
    a: &[E::ScalarField],
    b: &[E::ScalarField],
) -> (Vec<E::G1>, Vec<E::G1>) {
    (fk_quotients(pre, &domain, a), fk_quotients(pre, &domain, b))
}

fn fk_quotients<E: Pairing>(
    pre: &FkPrecomputation<E>,
    domain: &GeneralEvaluationDomain<E::ScalarField>,
    lagrange: &[E::ScalarField],
) -> Vec<E::G1> {
    let n = domain.size();
    // Coefficients c of A(X) in the monomial basis.
    let mut coeffs = domain.ifft(lagrange);
    coeffs.resize(2 * n, E::ScalarField::zero());
    pre.domain2.fft_in_place(&mut coeffs);
    // H_t = sum_k c_{t+k+1} * h_k is entry t + n - 1 of the linear
    // convolution of c with h reversed.
    let mut conv: Vec<E::G1> = pre
        .h_rev_dft
        .par_iter()
        .zip(coeffs.par_iter())
        .map(|(h, c)| *h * c)
        .collect();
    pre.domain2.ifft_in_place(&mut conv);
    let mut openings: Vec<E::G1> = (0..n)
        .map(|t| if t + 1 < n { conv[t + n - 1] } else { E::G1::zero() })
        .collect();
    // The opening quotient at omega^i is the DFT of H at omega^i.
    domain.fft_in_place(&mut openings);
    let n_inv = domain.size_inv();
    openings
        .par_iter_mut()
        .zip(domain.elements().collect::<Vec<_>>().par_iter())
        .for_each(|(q, omega_i)| *q *= *omega_i * n_inv);
    openings
}

///base batch size
pub const BASE_BATCH_SIZE: usize = 1<<10;
///Group operation batch size
pub const GROUP_OP_BATCH_SIZE: usize = BASE_BATCH_SIZE << 1;

/// Phase of Reconstruction
#[derive(Clone, Debug, PartialEq)]
pub enum ReconstructPhase{
    /// Prepare Vector Space
    VecPrepare(usize),
    /// DFT for Convolution 
    Dft(SqrtFftStatus),
    /// Pointwise Production
    PointwiseProduct(usize),
    /// iDFT for Convolution
    IDft(SqrtFftStatus),
    /// VectorIntegrate
    VecIntegrate(usize),
}

/// The state of a resumable computation of the projection data, using
/// the stepwise DFT.
#[derive(Clone, Debug, PartialEq)]
pub struct StepwiseProjection<E: Pairing> {
    /// Domain size.
    n: usize,
    /// The domain of size about `sqrt(n)` that the stepwise DFT is built from.
    haf_domain: GeneralEvaluationDomain<E::ScalarField>,
    /// `a .* ell_bar`, transformed in place by the DFT, the product with `w`
    /// and the iDFT.
    fl_a: Vec<E::G1>,
    /// `b .* ell_bar`, likewise.
    fl_b: Vec<E::G1>,
    /// `a`, transformed in place by the DFT, the product with `w_prime` and the
    /// iDFT.
    fw_a: Vec<E::ScalarField>,
    /// `b`, likewise.
    fw_b: Vec<E::ScalarField>,
    /// Scratch space for the transposes of `fl_a`.
    mem_fl_a: Vec<E::G1>,
    /// Scratch space for the transposes of `fl_b`.
    mem_fl_b: Vec<E::G1>,
    /// Scratch space for the transposes of `fw_a`.
    mem_fw_a: Vec<E::ScalarField>,
    /// Scratch space for the transposes of `fw_b`.
    mem_fw_b: Vec<E::ScalarField>,
    /// The current phase.
    status: ReconstructPhase,
    /// DFT tables for size `n`.
    fft_info: CustomParaFFT<E>,
}

impl<E: Pairing> StepwiseProjection<E> {
    /// A computation over a domain of size `n`, at its first step.
    pub fn new(n: usize) -> Self {
        let domain = GeneralEvaluationDomain::<E::ScalarField>::new(n)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)
            .unwrap();
        let haf_n: usize = 1 << (n.trailing_zeros() >> 1);
        let haf_domain = GeneralEvaluationDomain::<E::ScalarField>::new(haf_n)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)
            .unwrap();
        Self {
            n,
            haf_domain,
            fl_a: vec![E::G1::zero(); n],
            fl_b: vec![E::G1::zero(); n],
            fw_a: vec![E::ScalarField::zero(); n],
            fw_b: vec![E::ScalarField::zero(); n],
            mem_fl_a: vec![E::G1::zero(); n],
            mem_fl_b: vec![E::G1::zero(); n],
            mem_fw_a: vec![E::ScalarField::zero(); n],
            mem_fw_b: vec![E::ScalarField::zero(); n],
            status: ReconstructPhase::VecPrepare(0),
            fft_info: CustomParaFFT::<E>::new(n, domain.group_gen()),
        }
    }

    /// Starts over from the first step.
    pub fn restart(&mut self) {
        self.status = ReconstructPhase::VecPrepare(0);
    }

    /// Runs up to `try_steps` steps of computing the projection data of
    /// `a` and `b` into `q_a` and `q_b`, which the computation overwrites.
    /// Returns whether it finished, and how many steps are left over.
    pub fn step(
        &mut self,
        basis: &QuotientBasis<E>,
        a: &[E::ScalarField],
        b: &[E::ScalarField],
        q_a: &mut [E::G1],
        q_b: &mut [E::G1],
        try_steps: usize,
    ) -> (bool, usize) {
        let mut rem_stp = try_steps;
        while rem_stp > 0{
            let unt_stp = std::cmp::min(self.n, GROUP_OP_BATCH_SIZE);
            match self.status{
                ReconstructPhase::VecPrepare(ref mut id)=>{
                    let cnt_stp = std::cmp::min(rem_stp, (self.n-*id)/unt_stp);
                    let mx_stp = cnt_stp * unt_stp;
                    let res_a = &mut q_a[*id..(*id+mx_stp)];
                    let res_b = &mut q_b[*id..(*id+mx_stp)];
                    let f_a_slice = &a[*id..(*id+mx_stp)];
                    let f_b_slice = &b[*id..(*id+mx_stp)];
                    let fl_a_slice = &mut self.fl_a[*id..(*id+mx_stp)];
                    let fl_b_slice = &mut self.fl_b[*id..(*id+mx_stp)];
                    let fw_a_slice = &mut self.fw_a[*id..(*id+mx_stp)];
                    let fw_b_slice = &mut self.fw_b[*id..(*id+mx_stp)];
                    let lagrange = &basis.ell_bar[*id..(*id+mx_stp)];
                    let p = &basis.diag[*id..(*id+mx_stp)];
                    f_a_slice.par_iter().zip(f_b_slice.par_iter()).zip(
                    res_a.par_iter_mut().zip(res_b.par_iter_mut()).zip(
                        lagrange.par_iter().zip(p.par_iter()).zip(
                            fl_a_slice.par_iter_mut().zip(fl_b_slice.par_iter_mut())
                                .zip(fw_a_slice.par_iter_mut().zip(fw_b_slice.par_iter_mut()))
                            )
                        )
                    ).for_each(|((f_a,f_b),((qa,qb),((li, pi),((fl_a,fl_b),(fw_a,fw_b)))))|{
                        *fl_a = *li * f_a;
                        *fl_b = *li * f_b;
                        *fw_a = *f_a;
                        *fw_b = *f_b;
                        *qa = *pi * f_a;
                        *qb = *pi * f_b;
                    });
                    *id += mx_stp;
                    rem_stp -= cnt_stp;
                    if *id >= self.n{
                        *id = 0;
                        self.status = ReconstructPhase::Dft(SqrtFftStatus::new(false));
                    }
                }
                ReconstructPhase::Dft(ref mut fft_status)=>{
                    let mut fft_status_1a = fft_status.clone();
                    let mut fft_status_1b = fft_status.clone();
                    let mut fft_status_2a = fft_status.clone();
                    let mut fft_status_2b = fft_status.clone();
                    let (mut finish1a, mut finish1b, mut finish2a, mut finish2b) = ((false, 0), (false, 0), (false, 0), (false, 0));
                    {
                        let (mut mem_fl_a, mut mem_fl_b, mut mem_fw_a, mut mem_fw_b) = (&mut self.mem_fl_a, &mut self.mem_fl_b, &mut self.mem_fw_a, &mut self.mem_fw_b);
                        let (mut fl_a, mut fl_b, mut fw_a, mut fw_b) = (&mut self.fl_a, &mut self.fl_b, &mut self.fw_a, &mut self.fw_b);
                        join(
                            || {join(
                                || {finish1a = sqrt_fft_fw_g1(&self.haf_domain,&self.fft_info,&mut fl_a,&mut fft_status_1a, &mut mem_fl_a, rem_stp)},
                                || {finish1b = sqrt_fft_fw_g1(&self.haf_domain,&self.fft_info,&mut fl_b,&mut fft_status_1b, &mut mem_fl_b, rem_stp)},
                            )},
                            || {join(
                                || {finish2a = sqrt_fft_fw_field(&self.haf_domain,&self.fft_info,&mut fw_a,&mut fft_status_2a, &mut mem_fw_a, rem_stp)},
                                || {finish2b = sqrt_fft_fw_field(&self.haf_domain,&self.fft_info,&mut fw_b,&mut fft_status_2b, &mut mem_fw_b, rem_stp)},
                            )},
                        );
                    }
                    if finish1a != finish1b || finish1b != finish2a || finish2a !=finish2b{
                        println!("{:?}   {:?}   {:?}   {:?}", finish1a,finish1b,finish2a,finish2b);
                        println!("   {:?}\n   {:?}\n   {:?}\n   {:?}", fft_status_1a,fft_status_1b,fft_status_2a,fft_status_2b);
                        panic!("Not Syncronized");
                    }
                    rem_stp = finish1a.1;
                    if finish1a.0{
                        self.status = ReconstructPhase::PointwiseProduct(0);
                    }
                    else{
                        *fft_status = fft_status_1a;
                    }
                }
                ReconstructPhase::PointwiseProduct(ref mut id)=>{
                    let cnt_stp = std::cmp::min(rem_stp, (self.n-*id)/unt_stp);
                    let mx_stp = cnt_stp * unt_stp;
                    let fl_a_slice = &mut self.fl_a[*id..(*id+mx_stp)];
                    let fl_b_slice = &mut self.fl_b[*id..(*id+mx_stp)];
                    let fw_a_slice = &mut self.fw_a[*id..(*id+mx_stp)];
                    let fw_b_slice = &mut self.fw_b[*id..(*id+mx_stp)];
                    let w_slice = &basis.w_dft[*id..(*id+mx_stp)];
                    let w_rev_slice = &basis.w_prime_dft[*id..(*id+mx_stp)];
                    w_slice.par_iter().zip(w_rev_slice.par_iter()).zip(
                    fl_a_slice.par_iter_mut().zip(fl_b_slice.par_iter_mut()).zip(
                        fw_a_slice.par_iter_mut().zip(fw_b_slice.par_iter_mut())
                        )
                    ).for_each(|((w,w_rev),((fl_a,fl_b),(fw_a,fw_b)))|{
                        *fl_a *= *w;
                        *fl_b *= *w;
                        *fw_a *= *w_rev;
                        *fw_b *= *w_rev;
                    });
                    *id += mx_stp;
                    rem_stp -= cnt_stp;
                    if *id >= self.n{
                        *id = 0;
                        self.status = ReconstructPhase::IDft(SqrtFftStatus::new(true));
                    }
                }
                ReconstructPhase::IDft(ref mut fft_status)=>{
                    let mut fft_status_1a = fft_status.clone();
                    let mut fft_status_1b = fft_status.clone();
                    let mut fft_status_2a = fft_status.clone();
                    let mut fft_status_2b = fft_status.clone();
                    let (mut finish1a, mut finish1b, mut finish2a, mut finish2b) = ((false, 0), (false, 0), (false, 0), (false, 0));
                    {
                        let (mut mem_fl_a, mut mem_fl_b, mut mem_fw_a, mut mem_fw_b) = (&mut self.mem_fl_a, &mut self.mem_fl_b, &mut self.mem_fw_a, &mut self.mem_fw_b);
                        let (mut fl_a, mut fl_b, mut fw_a, mut fw_b) = (&mut self.fl_a, &mut self.fl_b, &mut self.fw_a, &mut self.fw_b);
                        join(
                            || {join(
                                || {finish1a = sqrt_fft_fw_g1(&self.haf_domain,&self.fft_info,&mut fl_a,&mut fft_status_1a, &mut mem_fl_a, rem_stp)},
                                || {finish1b = sqrt_fft_fw_g1(&self.haf_domain,&self.fft_info,&mut fl_b,&mut fft_status_1b, &mut mem_fl_b, rem_stp)},
                            )},|| {join(
                                || {finish2a = sqrt_fft_fw_field(&self.haf_domain,&self.fft_info,&mut fw_a,&mut fft_status_2a, &mut mem_fw_a, rem_stp)},
                                || {finish2b = sqrt_fft_fw_field(&self.haf_domain,&self.fft_info,&mut fw_b,&mut fft_status_2b, &mut mem_fw_b, rem_stp)},
                            )},
                        );
                    }
                    // let finish1a = sqrt_fft_fw_g1(&self.haf_domain,&self.fft_info,&mut self.fl_a,&mut fft_status_1a, &mut self.mem_fl_a);
                    // let finish1b = sqrt_fft_fw_g1(&self.haf_domain,&self.fft_info,&mut self.fl_b,&mut fft_status_1b, &mut self.mem_fl_b);
                    // let finish2a = sqrt_fft_fw_field(&self.haf_domain,&self.fft_info,&mut self.fw_a,&mut fft_status_2a, &mut self.mem_fw_a);
                    // let finish2b = sqrt_fft_fw_field(&self.haf_domain,&self.fft_info,&mut self.fw_b,&mut fft_status_2b, &mut self.mem_fw_b);
                    if finish1a != finish1b || finish1b != finish2a || finish2a !=finish2b{
                        panic!("Not Syncronized");
                    }
                    rem_stp = finish1a.1;
                    if finish1a.0{
                        self.status = ReconstructPhase::VecIntegrate(0);
                    }
                    else{
                        *fft_status = fft_status_1a;
                    }
                }
                ReconstructPhase::VecIntegrate(ref mut id)=>{
                    let cnt_stp = std::cmp::min(rem_stp, (self.n-*id)/unt_stp);
                    let mx_stp = cnt_stp * unt_stp;
                    let res_a = &mut q_a[*id..(*id+mx_stp)];
                    let res_b = &mut q_b[*id..(*id+mx_stp)];
                    let fl_a_slice = &mut self.fl_a[*id..(*id+mx_stp)];
                    let fl_b_slice = &mut self.fl_b[*id..(*id+mx_stp)];
                    let fw_a_slice = &mut self.fw_a[*id..(*id+mx_stp)];
                    let fw_b_slice = &mut self.fw_b[*id..(*id+mx_stp)];
                    let lagrange = &basis.ell_bar[*id..(*id+mx_stp)];
                    res_a.par_iter_mut().zip(res_b.par_iter_mut()).zip(
                    lagrange.par_iter().zip(
                        fl_a_slice.par_iter_mut().zip(fl_b_slice.par_iter_mut())
                            .zip(fw_a_slice.par_iter_mut().zip(fw_b_slice.par_iter_mut()))
                        )
                    ).for_each(|((qa,qb),(li,((fl_a,fl_b),(fw_a,fw_b))))|{
                        *qa += *fl_a + *li * fw_a;
                        *qb += *fl_b + *li * fw_b;
                    });
                    *id += mx_stp;
                    rem_stp -= cnt_stp;
                    if *id >= self.n{
                        *id = 0;
                        self.status = ReconstructPhase::VecPrepare(0);
                        return (true, rem_stp);
                    }
                }
            }
        };
        (false, rem_stp)
    }
}
