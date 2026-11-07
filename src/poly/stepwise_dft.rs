//! Stepwise DFT (appendix "Stepwise DFT and Stepwise Preprocessing" of the
//! paper): a DFT of size `n` computed as `sqrt(n) x sqrt(n)` sub-DFTs and transposes,
//! which can be paused after any number of steps. The fully-dynamic prover
//! uses it to rebuild its preprocessed data a few steps per update.
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::SynthesisError;
use ark_serialize::*;
use ark_std::{log2, vec::Vec};
use rayon::prelude::*;
use std::fmt::Debug;

/// Number of Threads
pub const N_THREADS: usize = 1;
/// Size of Step of Sqrt FFT
pub const STEP_SQRT_FFT: usize = 1;
/// Size of Step of Transpose
pub const STEP_SQRT_TRANSPOSE: usize = 2;
/// Size of Step of Transpose
pub const STEP_SQRT_COPY: usize = 32;

#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize, Default)]
/// Stores info of FFT, including powers of generators and bit-reversion table
pub struct CustomParaFFT<E: Pairing> {
    /// size of a FFT array
    pub log_n: usize,
    /// size of a FFT array
    pub n: usize,
    /// power of roots of unity
    pub omegas: Vec<E::ScalarField>,
    /// power of inversion of roots of unity
    pub omega_invs: Vec<E::ScalarField>,
    /// list of bit-reversed match
    pub bit_rev_table: Vec<usize>,
    /// inverse of N
    pub bit_rev_task: Vec<usize>,
    /// Need Swap
    pub inv_n: E::ScalarField,
    /// covering_length per Thread
    pub len_thread: usize,
}

impl<E: Pairing> CustomParaFFT<E> {
    /// Create new FFT parameter header
    pub fn new(g_size: usize, g_gen: E::ScalarField) -> Self {
        let n = g_size;
        // let n = domain.size();
        if !n.is_power_of_two() {
            // panic!("Get N = {:?} not Power of Two", n);
        }
        if !N_THREADS.is_power_of_two() {
            panic!("Number of Threads = {:?} not Power of Two", N_THREADS);
        }
        if (N_THREADS << 1) > n {
            panic!(
                "Number of Threads = {:?}, its twice even more than N( {:?} )",
                N_THREADS, n
            );
        }

        let omega = g_gen;
        // let omega = domain.group_gen();
        // let omega = F::get_root_of_unity(n);

        let omega_inv = omega.inverse().unwrap();

        let log_n = log2(n) as usize;
        let f_one = <E::ScalarField as Field>::ONE;
        let f_two = f_one + f_one;

        let mut omegas = vec![f_one; n];
        let mut omega_invs = vec![f_one; n];

        for i in 1..n {
            omegas[i] = omegas[i - 1] * omega;
            omega_invs[i] = omega_invs[i - 1] * omega_inv;
        }

        let mut bit_rev_table = vec![0; n];
        let mut bit_rev_task: Vec<usize> = Vec::new();
        for i in 0..n {
            bit_rev_table[i] = (bit_rev_table[i >> 1] >> 1) | ((i & 1) << (log_n - 1));
            if i < bit_rev_table[i] {
                bit_rev_task.push(i);
                bit_rev_task.push(bit_rev_table[i]);
            }
        }

        if (bit_rev_task.len() & ((N_THREADS << 1) - 1)) > 0 {
            // panic!("Number of Threads = {:?}, Not Dividing N_bit_rev( {:?}
            // )", (N_THREADS<<1),bit_rev_task.len());
        }

        let mut inv_n = f_one;
        for _ in 0..log_n {
            inv_n = inv_n * f_two.inverse().unwrap();
        }
        let len_thread = n / N_THREADS;
        Self {
            log_n,
            n,
            omegas,
            omega_invs,
            bit_rev_table,
            bit_rev_task,
            inv_n,
            len_thread,
        }
    }
}

/// sqrt FFT
pub fn sqrt_fft<E: Pairing>(fft_info: &CustomParaFFT<E>, x: &mut FFTVectorTypes<E>, is_rev: bool) {
    // println!("It's really happening!");
    // if fft_info.log_n & 1 != 0 {
    //     panic!("FFT log_n Not Even");
    // }
    let rows_log = fft_info.log_n >> 1;
    let cols_log = fft_info.log_n - rows_log;
    let rows_cnt:usize = 1 << rows_log;
    let cols_cnt:usize = 1 << cols_log;
    type D<F> = GeneralEvaluationDomain<F>;
    let rows_domain: GeneralEvaluationDomain<<E as Pairing>::ScalarField> = D::new(rows_cnt)
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)
        .unwrap();

    let cols_domain: GeneralEvaluationDomain<<E as Pairing>::ScalarField> = D::new(cols_cnt)
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)
        .unwrap();
    match x {
        FFTVectorTypes::XF(ref mut xf) => {
            // First Transpose
            let copy = xf.clone();
            let cover_copy: Vec<E::ScalarField> = (0..fft_info.n)
                .into_par_iter()
                .map(|i: usize| {
                    let r = i >> rows_log;
                    let c = i & ((1 << rows_log) - 1);
                    copy[(c << cols_log) | r]
                })
                .collect();
            xf.copy_from_slice(&cover_copy);
            // Column FFT
            xf.par_chunks_mut(rows_cnt).for_each(|row| {
                let mut row_copy = row.to_vec();
                if is_rev {
                    rows_domain.ifft_in_place(&mut row_copy);
                } else {
                    rows_domain.fft_in_place(&mut row_copy);
                }
                row.copy_from_slice(&row_copy);
            });
            // Transpose with Variable Twiddle Factors
            let copy: Vec<E::ScalarField> = xf.clone();
            let cover_copy: Vec<E::ScalarField> = (0..fft_info.n)
                .into_par_iter()
                .map(|i| {
                    let r = i >> cols_log;
                    let c = i & ((1 << cols_log) - 1);
                    copy[(c << rows_log) | r]
                        * if is_rev {
                            fft_info.omega_invs[r * c]
                        } else {
                            fft_info.omegas[r * c]
                        }
                })
                .collect();
            xf.copy_from_slice(&cover_copy);
            // Row FFT
            xf.par_chunks_mut(1 << cols_log).for_each(|row| {
                let mut row_copy = row.to_vec();
                if is_rev {
                    cols_domain.ifft_in_place(&mut row_copy);
                } else {
                    cols_domain.fft_in_place(&mut row_copy);
                }
                row.copy_from_slice(&row_copy);
            });
            // Final Transpose
            let copy: Vec<E::ScalarField> = xf.clone();
            let cover_copy: Vec<E::ScalarField> = (0..fft_info.n)
                .into_par_iter()
                .map(|i| {
                    let r = i >> rows_log;
                    let c = i & ((1 << rows_log) - 1);
                    copy[(c << cols_log) | r]
                })
                .collect();
            xf.copy_from_slice(&cover_copy);
        },
        FFTVectorTypes::XG1(ref mut xg1) => {
            // First Transpose
            let copy = xg1.clone();
            let cover_copy: Vec<E::G1> = (0..fft_info.n)
                .into_par_iter()
                .map(|i: usize| {
                    let r = i >> rows_log;
                    let c = i & ((1 << rows_log) - 1);
                    copy[(c << cols_log) | r]
                })
                .collect();
            xg1.copy_from_slice(&cover_copy);
            // Column FFT
            xg1.par_chunks_mut(rows_cnt).for_each(|row| {
                let mut row_copy = row.to_vec();
                if is_rev {
                    rows_domain.ifft_in_place(&mut row_copy);
                } else {
                    rows_domain.fft_in_place(&mut row_copy);
                }
                row.copy_from_slice(&row_copy);
            });
            // Transpose with Variable Twiddle Factors
            let copy: Vec<E::G1> = xg1.clone();
            let cover_copy: Vec<E::G1> = (0..fft_info.n)
                .into_par_iter()
                .map(|i| {
                    let r = i >> cols_log;
                    let c = i & ((1 << cols_log) - 1);
                    copy[(c << rows_log) | r]
                        * if is_rev {
                            fft_info.omega_invs[r * c]
                        } else {
                            fft_info.omegas[r * c]
                        }
                })
                .collect();
            xg1.copy_from_slice(&cover_copy);
            // Row FFT
            xg1.par_chunks_mut(1 << cols_log).for_each(|row| {
                let mut row_copy = row.to_vec();
                if is_rev {
                    cols_domain.ifft_in_place(&mut row_copy);
                } else {
                    cols_domain.fft_in_place(&mut row_copy);
                }
                row.copy_from_slice(&row_copy);
            });
            // Final Transpose
            let copy: Vec<E::G1> = xg1.clone();
            let cover_copy: Vec<E::G1> = (0..fft_info.n)
                .into_par_iter()
                .map(|i| {
                    let r = i >> rows_log;
                    let c = i & ((1 << rows_log) - 1);
                    copy[(c << cols_log) | r]
                })
                .collect();
            xg1.copy_from_slice(&cover_copy);
        },
        FFTVectorTypes::XG2(ref mut xg2) => {
            // First Transpose
            let copy = xg2.clone();
            let cover_copy: Vec<E::G2> = (0..fft_info.n)
                .into_par_iter()
                .map(|i: usize| {
                    let r = i >> rows_log;
                    let c = i & ((1 << rows_log) - 1);
                    copy[(c << cols_log) | r]
                })
                .collect();
            xg2.copy_from_slice(&cover_copy);
            // Column FFT
            xg2.par_chunks_mut(rows_cnt).for_each(|row| {
                let mut row_copy = row.to_vec();
                if is_rev {
                    rows_domain.ifft_in_place(&mut row_copy);
                } else {
                    rows_domain.fft_in_place(&mut row_copy);
                }
                row.copy_from_slice(&row_copy);
            });
            // Transpose with Variable Twiddle Factors
            let copy: Vec<E::G2> = xg2.clone();
            let cover_copy: Vec<E::G2> = (0..fft_info.n)
                .into_par_iter()
                .map(|i| {
                    let r = i >> cols_log;
                    let c = i & ((1 << cols_log) - 1);
                    copy[(c << rows_log) | r]
                        * if is_rev {
                            fft_info.omega_invs[r * c]
                        } else {
                            fft_info.omegas[r * c]
                        }
                })
                .collect();
            xg2.copy_from_slice(&cover_copy);
            // Row FFT
            xg2.par_chunks_mut(1 << cols_log).for_each(|row| {
                let mut row_copy = row.to_vec();
                if is_rev {
                    cols_domain.ifft_in_place(&mut row_copy);
                } else {
                    cols_domain.fft_in_place(&mut row_copy);
                }
                row.copy_from_slice(&row_copy);
            });
            // Final Transpose
            let copy: Vec<E::G2> = xg2.clone();
            let cover_copy: Vec<E::G2> = (0..fft_info.n)
                .into_par_iter()
                .map(|i| {
                    let r = i >> rows_log;
                    let c = i & ((1 << rows_log) - 1);
                    copy[(c << cols_log) | r]
                })
                .collect();
            xg2.copy_from_slice(&cover_copy);
        },
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
/// Describe the phase of sqrt FFT process
enum FftPhase {
    /// first transpose
    PreTranspose,
    /// phase1
    ColFFT,
    /// phase2
    TRANSPOSE,
    /// phase3
    RowFFT,
    /// final transpose
    SufTranspose
}

#[derive(Clone, Debug, PartialEq)]
/// Describe the status of sqrt FFT process
pub struct SqrtFftStatus {
    /// Whether is reverse of fft
    is_rev: bool,
    /// Current Phase
    phase: FftPhase,
    /// Current Level of phase
    lev: usize,
}
impl SqrtFftStatus{
    /// The status at the start of a DFT, or of an inverse DFT if `is_rev`.
    pub fn new(is_rev: bool)->Self{
        Self{
            is_rev,
            phase:FftPhase::PreTranspose,
            lev:0
        }
    }
}

/// Different types of Element doing fft
#[derive(Clone, Debug, PartialEq)]
pub enum FFTVectorTypes<E: Pairing> {
    /// Field elements.
    XF(Vec<E::ScalarField>),
    /// `G1` elements.
    XG1(Vec<E::G1>),
    /// `G2` elements.
    XG2(Vec<E::G2>),
}

/// a step of sbs FFT for field
pub fn sqrt_fft_fw_field<E: Pairing>(
    sub_domain: &GeneralEvaluationDomain<<E as Pairing>::ScalarField>,
    sqrt_fft_info: &CustomParaFFT<E>,
    xf: &mut Vec<E::ScalarField>,
    status: &mut SqrtFftStatus,
    mem_for_transpose: &mut Vec<E::ScalarField>,
    steps: usize,
) -> (bool, usize) {
    let is_rev = status.is_rev;
    let haf_log = sqrt_fft_info.log_n >> 1;
    let mx_step: usize = 1 << haf_log;
    let mut steps_rem = steps;
    while steps_rem >0{
        match status.phase {
            FftPhase::PreTranspose | FftPhase::SufTranspose =>{
                let unit_stp = std::cmp::min(mx_step, STEP_SQRT_COPY);
                if status.lev < (1<< haf_log){
                    let cnt_stp = std::cmp::min(steps_rem, (mx_step -status.lev) /unit_stp);
                    let stp = cnt_stp * unit_stp;
                    steps_rem -= cnt_stp;
                    let l = status.lev  << haf_log;
                    let r = (status.lev+ stp) << haf_log;
                    (l..r).into_par_iter().zip(mem_for_transpose[l..r].par_iter_mut()).for_each(|(i,v)|{
                        let r = i >> haf_log;
                        let c = i&((1<<haf_log)-1);
                        *v = xf[(c << haf_log) | r];
                    });
                    status.lev += stp;
                }
                else{
                    let cnt_stp = std::cmp::min(steps_rem, ((mx_step<<1) -status.lev) /unit_stp);
                    let stp = cnt_stp * unit_stp;
                    steps_rem -= cnt_stp;
                    let l = (status.lev-(1<<haf_log))  << haf_log;
                    let r = (status.lev+stp-(1<<haf_log))  << haf_log;
                    xf[l..r].copy_from_slice(&mem_for_transpose[l..r]);
                    status.lev += stp;
                }
                if status.lev >= 2<<haf_log{
                    if status.phase == FftPhase::PreTranspose{
                        status.lev = 0;
                        status.phase = FftPhase::ColFFT;
                    }
                    else{
                        status.lev = 0;
                        status.phase = FftPhase::PreTranspose;
                        return (true, steps_rem);
                    }
                }
            }
            FftPhase::ColFFT => {
                let unit_stp = std::cmp::min(mx_step, STEP_SQRT_FFT);
                let cnt_stp = std::cmp::min(steps_rem, (mx_step-status.lev)/unit_stp);
                let stp = cnt_stp * unit_stp;
                steps_rem -= cnt_stp;
                let haf_info = CustomParaFFT::<E>::new(1 << haf_log, sub_domain.group_gen());
                xf[(status.lev << haf_log)..((status.lev + stp) << haf_log)]
                    .par_chunks_mut(1 << haf_log)
                    .for_each(|row| {
                        let mut xf = FFTVectorTypes::XF(row.to_vec());
                        sqrt_fft(&haf_info, &mut xf, is_rev);
                        if let FFTVectorTypes::XF(v) = xf {
                            row.copy_from_slice(&v);
                        }
                    });
                status.lev += stp;
                if status.lev >= 1 << haf_log {
                    status.phase = FftPhase::TRANSPOSE;
                    status.lev = 0;
                }
            },
            FftPhase::TRANSPOSE => {
                let unit_stp = std::cmp::min(mx_step, STEP_SQRT_TRANSPOSE);
                if status.lev < (1<< haf_log){
                    let cnt_stp = std::cmp::min(steps_rem, (mx_step-status.lev)/unit_stp);
                    let stp = cnt_stp * unit_stp;
                    steps_rem -= cnt_stp;
                    let l = status.lev  << haf_log;
                    let r = (status.lev+ stp) << haf_log;
                    (l..r).into_par_iter().zip(mem_for_transpose[l..r].par_iter_mut()).for_each(|(i,v)|{
                        let r = i >> haf_log;
                        let c = i&((1<<haf_log)-1);
                        *v = xf[(c << haf_log) | r] * 
                            if status.is_rev {
                                sqrt_fft_info.omega_invs[r * c]
                            } else {
                                sqrt_fft_info.omegas[r * c]
                            }
                    });
                    status.lev += stp;
                }
                else{
                    let cnt_stp = std::cmp::min(steps_rem, ((mx_step<<1)-status.lev)/unit_stp);
                    let stp = cnt_stp * unit_stp;
                    steps_rem -= cnt_stp;
                    let l = (status.lev-(1<<haf_log))  << haf_log;
                    let r = (status.lev+stp-(1<<haf_log))  << haf_log;
                    xf[l..r].copy_from_slice(&mem_for_transpose[l..r]);
                    status.lev += stp;
                    if status.lev >= 2 << haf_log {
                        status.phase = FftPhase::RowFFT;
                        status.lev = 0;
                    }
                }
            },
            FftPhase::RowFFT => {
                let unit_stp = std::cmp::min(mx_step, STEP_SQRT_FFT);
                let cnt_stp = std::cmp::min(steps_rem, (mx_step - status.lev)/unit_stp);
                let stp = cnt_stp * unit_stp;
                steps_rem -= cnt_stp;
                let haf_info = CustomParaFFT::<E>::new(1 << haf_log, sub_domain.group_gen());
                xf[(status.lev << haf_log)..((status.lev + stp) << haf_log)]
                    .par_chunks_mut(1 << haf_log)
                    .for_each(|row| {
                        let mut xf = FFTVectorTypes::XF(row.to_vec());
                        sqrt_fft(&haf_info, &mut xf, is_rev);
                        if let FFTVectorTypes::XF(v) = xf {
                            row.copy_from_slice(&v);
                        }
                    });
                status.lev += stp;
                if status.lev >= 1 << haf_log {
                    status.phase = FftPhase::SufTranspose;
                    status.lev = 0;
                }
            },
        }
    }
    return (false, 0);
}


/// a step of sbs FFT for g1
pub fn sqrt_fft_fw_g1<E: Pairing>(
    sub_domain: &GeneralEvaluationDomain<<E as Pairing>::ScalarField>,
    sqrt_fft_info: &CustomParaFFT<E>,
    xg1: &mut Vec<E::G1>,
    status: &mut SqrtFftStatus,
    mem_for_transpose: &mut Vec<E::G1>,
    steps: usize
) -> (bool, usize) {
    let is_rev = status.is_rev;
    let haf_log = sqrt_fft_info.log_n >> 1;
    let mx_step: usize = 1<<haf_log;
    let mut steps_rem = steps;
    while steps_rem >0{
        match status.phase {
            FftPhase::PreTranspose | FftPhase::SufTranspose =>{
                let unit_stp = std::cmp::min(mx_step, STEP_SQRT_COPY);
                if status.lev < (1<< haf_log){
                    let cnt_stp = std::cmp::min(steps_rem, (mx_step -status.lev) /unit_stp);
                    let stp = cnt_stp * unit_stp;
                    steps_rem -= cnt_stp;
                    let l = status.lev  << haf_log;
                    let r = (status.lev+ stp) << haf_log;
                    (l..r).into_par_iter().zip(mem_for_transpose[l..r].par_iter_mut()).for_each(|(i,v)|{
                        let r = i >> haf_log;
                        let c = i&((1<<haf_log)-1);
                        *v = xg1[(c << haf_log) | r];
                    });
                    status.lev += stp;
                }
                else{
                    let cnt_stp = std::cmp::min(steps_rem, ((mx_step<<1) -status.lev) /unit_stp);
                    let stp = cnt_stp * unit_stp;
                    steps_rem -= cnt_stp;
                    let l = (status.lev-(1<<haf_log))  << haf_log;
                    let r = (status.lev+stp-(1<<haf_log))  << haf_log;
                    xg1[l..r].copy_from_slice(&mem_for_transpose[l..r]);
                    status.lev += stp;
                }
                if status.lev >= 2<<haf_log{
                    if status.phase == FftPhase::PreTranspose{
                        status.lev = 0;
                        status.phase = FftPhase::ColFFT;
                    }
                    else{
                        status.lev = 0;
                        status.phase = FftPhase::PreTranspose;
                        return (true, steps_rem);
                    }
                }
            }
            FftPhase::ColFFT => {
                let unit_stp = std::cmp::min(mx_step, STEP_SQRT_FFT);
                let cnt_stp = std::cmp::min(steps_rem, (mx_step - status.lev)/unit_stp);
                let stp = cnt_stp * unit_stp;
                steps_rem -= cnt_stp;
                let haf_info = CustomParaFFT::<E>::new(1 << haf_log, sub_domain.group_gen());
                xg1[(status.lev << haf_log)..((status.lev + stp) << haf_log)]
                    .par_chunks_mut(1 << haf_log)
                    .for_each(|row| {
                        let mut xf = FFTVectorTypes::XG1(row.to_vec());
                        sqrt_fft(&haf_info, &mut xf, is_rev);
                        if let FFTVectorTypes::XG1(v) = xf {
                            row.copy_from_slice(&v);
                        }
                    });
                status.lev += stp;
                if status.lev >= 1 << haf_log {
                    status.phase = FftPhase::TRANSPOSE;
                    status.lev = 0;
                }
            },
            FftPhase::TRANSPOSE => {
                let unit_stp = std::cmp::min(mx_step, STEP_SQRT_TRANSPOSE);
                if status.lev < (1<< haf_log){
                    let cnt_stp = std::cmp::min(steps_rem, (mx_step-status.lev)/unit_stp);
                    let stp = cnt_stp * unit_stp;
                    steps_rem -= cnt_stp;
                    let l = status.lev  << haf_log;
                    let r = (status.lev+ stp) << haf_log;
                    (l..r).into_par_iter().zip(mem_for_transpose[l..r].par_iter_mut()).for_each(|(i,v)|{
                        let r = i >> haf_log;
                        let c = i&((1<<haf_log)-1);
                        *v = xg1[(c << haf_log) | r] * 
                            if status.is_rev {
                                sqrt_fft_info.omega_invs[r * c]
                            } else {
                                sqrt_fft_info.omegas[r * c]
                            }
                    });
                    status.lev += stp;
                }
                else{
                    let cnt_stp = std::cmp::min(steps_rem, ((mx_step<<1)-status.lev)/unit_stp);
                    let stp = cnt_stp * unit_stp;
                    steps_rem -= cnt_stp;
                    let l = (status.lev-(1<<haf_log))  << haf_log;
                    let r = (status.lev+stp-(1<<haf_log))  << haf_log;
                    xg1[l..r].copy_from_slice(&mem_for_transpose[l..r]);
                    status.lev += stp;
                    if status.lev >= 2 << haf_log {
                        status.phase = FftPhase::RowFFT;
                        status.lev = 0;
                    }
                }
            },
            FftPhase::RowFFT => {
                let unit_stp = std::cmp::min(mx_step, STEP_SQRT_FFT);
                let cnt_stp = std::cmp::min(steps_rem, (mx_step-status.lev)/unit_stp);
                let stp = cnt_stp * unit_stp;
                steps_rem -= cnt_stp;
                let haf_info = CustomParaFFT::<E>::new(1 << haf_log, sub_domain.group_gen());
                xg1[(status.lev << haf_log)..((status.lev + stp) << haf_log)]
                    .par_chunks_mut(1 << haf_log)
                    .for_each(|row| {
                        let mut xf = FFTVectorTypes::XG1(row.to_vec());
                        sqrt_fft(&haf_info, &mut xf, is_rev);
                        if let FFTVectorTypes::XG1(v) = xf {
                            row.copy_from_slice(&v);
                        }
                    });
                status.lev += stp;
                if status.lev >= 1 << haf_log {
                    status.phase = FftPhase::SufTranspose;
                    status.lev = 0;
                }
            },
        }
    }
    return (false, 0);
}

