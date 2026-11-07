//! The synthetic R1CS workloads used in the evaluation (Section
//! "Evaluations" of the paper).
//!
//! With `n` constraints and `m = n / 2`, row `i` for `0 < i < m` enforces
//! `(k_a * z_i + c_a) * (k_b * z_i + c_b) = z_{i+m}` with random `k_a, k_b, c_a, c_b`.
//! An update to `z_i` therefore also changes `z_{i+m}`.
use ark_ec::pairing::Pairing;
use ark_ff::{One, UniformRand, Zero};
use ark_relations::r1cs::{ConstraintMatrices, Matrix};
use ark_std::rand::{seq::SliceRandom, Rng};
use rand::seq::index::sample;

/// The constraint matrices, and the coefficients `(k_a, k_b, c_a, c_b)` of each row.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn generate_matrices<E: Pairing>(
    num_constraint: usize,
    num_instance: usize,
    num_witness: usize,
    rng: &mut impl Rng,
) -> (
    ConstraintMatrices<E::ScalarField>,
    Vec<E::ScalarField>,
    Vec<E::ScalarField>,
    Vec<E::ScalarField>,
    Vec<E::ScalarField>,
) {
    assert_eq!(
        num_constraint,
        num_instance + num_witness,
        "num_constraint must equal num_instance + num_witness"
    );

    let m = num_constraint >> 1;
    let mut a: Matrix<E::ScalarField> = vec![Vec::new(); num_constraint];
    let mut b: Matrix<E::ScalarField> = vec![Vec::new(); num_constraint];
    let mut c = a.clone();
    let mut ka: Vec<E::ScalarField> = Vec::new();
    let mut kb: Vec<E::ScalarField> = Vec::new();
    let mut ca: Vec<E::ScalarField> = Vec::new();
    let mut cb: Vec<E::ScalarField> = Vec::new();
    (1..m).for_each(|i| {
        let k_a: <E as Pairing>::ScalarField = E::ScalarField::rand(rng);
        let k_b: <E as Pairing>::ScalarField = E::ScalarField::rand(rng);
        let c_a: <E as Pairing>::ScalarField = E::ScalarField::rand(rng);
        let c_b: <E as Pairing>::ScalarField = E::ScalarField::rand(rng);
        // let k_a: <E as Pairing>::ScalarField = E::ScalarField::one();
        // let k_b: <E as Pairing>::ScalarField = E::ScalarField::zero();
        // let c_a: <E as Pairing>::ScalarField = E::ScalarField::zero();
        // let c_b: <E as Pairing>::ScalarField = E::ScalarField::one();
        a[i].push((k_a, i));
        a[i].push((c_a, 0));
        b[i].push((k_b, i));
        b[i].push((c_b, 0));
        c[i].push((E::ScalarField::one(), i + m));
        ka.push(k_a);
        kb.push(k_b);
        ca.push(c_a);
        cb.push(c_b);
    });

    (
        ConstraintMatrices {
            num_instance_variables: num_instance,
            num_witness_variables: num_witness,
            num_constraints: num_constraint,
            a_num_non_zero: num_constraint,
            b_num_non_zero: num_constraint,
            c_num_non_zero: num_constraint,
            a,
            b,
            c,
        },
        ka,
        kb,
        ca,
        cb,
    )
}

/// A satisfying assignment with random `z_i` for `0 < i < m`.
pub fn generate_instance_witness<E: Pairing>(
    num_instance: usize,
    num_witness: usize,
    rng: &mut impl Rng,
    ka: &[E::ScalarField],
    kb: &[E::ScalarField],
    ca: &[E::ScalarField],
    cb: &[E::ScalarField],
) -> (Vec<E::ScalarField>, Vec<E::ScalarField>) {
    let mut instance = vec![E::ScalarField::zero(); num_instance];
    instance[0] = E::ScalarField::one(); // first instance variable is 1
    let mut witness = vec![E::ScalarField::zero(); num_witness];
    let m = (num_witness + num_instance) >> 1;

    for i in 1..m {
        let vl = E::ScalarField::rand(rng);
        // let vl = E::ScalarField::one();
        let tar = (vl * ka[i - 1] + ca[i - 1]) * (vl * kb[i - 1] + cb[i - 1]);
        if i < num_instance {
            instance[i] = vl;
        } else {
            witness[i - num_instance] = vl;
        }
        if i + m < num_instance {
            instance[i + m] = tar;
        } else {
            witness[i + m - num_instance] = tar;
        }
    }
    // println!("{:?}", instance);
    // println!("{:?}", witness);
    (instance, witness)
}

/// A random update to `num_update` of the `z_i`, applied to `instance` and
/// `witness` in place. Returns the sorted changes to the instance and to the
/// witness, as `(variable index, delta)`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn generate_update<E: Pairing>(
    num_constraint: usize,
    num_update: usize,
    num_instance: usize,
    ka: &[E::ScalarField],
    kb: &[E::ScalarField],
    ca: &[E::ScalarField],
    cb: &[E::ScalarField],
    instance: &mut [E::ScalarField],
    witness: &mut [E::ScalarField],
    rng: &mut impl Rng,
) -> (Vec<(usize, E::ScalarField)>, Vec<(usize, E::ScalarField)>) {
    let mut update_instance = Vec::new();
    let mut update_witness = Vec::new();
    let m = num_constraint >> 1;
    if num_update >= m {
        panic!("Too Much Update");
    }
    let mut sample_rng = rand::rng();
    let indices: Vec<usize> = sample(&mut sample_rng, m - 1, num_update)
        .into_iter()
        .map(|x| x + 1)
        .collect();
    for i in indices {
        let dt_z = E::ScalarField::rand(rng);
        let vl = if i < num_instance {
            update_instance.push((i, dt_z));
            &mut instance[i]
        } else {
            update_witness.push((i, dt_z));
            &mut witness[i - num_instance]
        };
        let prev_val = (*vl * ka[i - 1] + ca[i - 1]) * (*vl * kb[i - 1] + cb[i - 1]);
        *vl += dt_z;
        let next_val = (*vl * ka[i - 1] + ca[i - 1]) * (*vl * kb[i - 1] + cb[i - 1]);
        let delta = next_val - prev_val;
        if i + m < num_instance {
            instance[i + m] = next_val;
            update_instance.push((i + m, delta));
        } else {
            witness[i + m - num_instance] = next_val;
            update_witness.push((i + m, delta));
        }
    }
    update_instance.sort_by_key(|x| x.0);
    update_witness.sort_by_key(|y| y.0);
    // This need to be sorted,
    (update_instance, update_witness)
}

/// Like [`generate_update`], but leaves `instance` and `witness` unchanged.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn generate_update_once<E: Pairing>(
    num_constraint: usize,
    num_update: usize,
    num_instance: usize,
    ka: &[E::ScalarField],
    kb: &[E::ScalarField],
    ca: &[E::ScalarField],
    cb: &[E::ScalarField],
    instance: &[E::ScalarField],
    witness: &[E::ScalarField],
    rng: &mut impl Rng,
) -> (Vec<(usize, E::ScalarField)>, Vec<(usize, E::ScalarField)>) {
    let mut update_instance = Vec::new();
    let mut update_witness = Vec::new();
    let m = num_constraint >> 1;
    if num_update >= m {
        panic!("Too Much Update");
    }
    let mut sample_rng = rand::rng();
    let indices: Vec<usize> = sample(&mut sample_rng, m - 1, num_update)
        .into_iter()
        .map(|x| x + 1)
        .collect();
    for i in indices {
        let dt_z = E::ScalarField::rand(rng);
        let vl = if i < num_instance {
            update_instance.push((i, dt_z));
            instance[i]
        } else {
            update_witness.push((i, dt_z));
            witness[i - num_instance]
        };
        let prev_val = (vl * ka[i - 1] + ca[i - 1]) * (vl * kb[i - 1] + cb[i - 1]);
        let next_val =
            ((vl + dt_z) * ka[i - 1] + ca[i - 1]) * ((vl + dt_z) * kb[i - 1] + cb[i - 1]);
        let delta = next_val - prev_val;
        if i + m < num_instance {
            update_instance.push((i + m, delta));
        } else {
            update_witness.push((i + m, delta));
        }
    }
    update_instance.sort_by_key(|x| x.0);
    update_witness.sort_by_key(|y| y.0);
    // This need to be sorted,
    (update_instance, update_witness)
}

/// A sparse workload: `n` rows `z_x * z_y = z_out` over `num_inputs` inputs, of
/// which only `nonzero` are nonzero, so `z`, `A * z` and `B * z` are mostly
/// zero. Returns the matrices, the instance (just the constant one) and the
/// witness.
#[allow(clippy::type_complexity)]
pub fn generate_sparse_r1cs<E: Pairing>(
    n: usize,
    num_inputs: usize,
    nonzero: usize,
    rng: &mut impl Rng,
) -> (
    ConstraintMatrices<E::ScalarField>,
    Vec<E::ScalarField>,
    Vec<E::ScalarField>,
) {
    type F<E> = <E as Pairing>::ScalarField;
    let num_variables = 1 + num_inputs + n;
    let mut z = vec![F::<E>::zero(); num_variables];
    z[0] = F::<E>::one();
    let mut inputs: Vec<usize> = (1..=num_inputs).collect();
    inputs.shuffle(rng);
    for i in &inputs[..nonzero] {
        z[*i] = F::<E>::rand(rng);
    }
    let (mut a, mut b, mut c): (Matrix<F<E>>, Matrix<F<E>>, Matrix<F<E>>) =
        (vec![], vec![], vec![]);
    for row in 0..n {
        let (x, y, out) = (
            1 + rng.gen_range(0..num_inputs),
            1 + rng.gen_range(0..num_inputs),
            1 + num_inputs + row,
        );
        z[out] = z[x] * z[y];
        a.push(vec![(F::<E>::one(), x)]);
        b.push(vec![(F::<E>::one(), y)]);
        c.push(vec![(F::<E>::one(), out)]);
    }
    let matrices = ConstraintMatrices {
        num_instance_variables: 1,
        num_witness_variables: num_variables - 1,
        num_constraints: n,
        a_num_non_zero: n,
        b_num_non_zero: n,
        c_num_non_zero: n,
        a,
        b,
        c,
    };
    (matrices, vec![F::<E>::one()], z[1..].to_vec())
}
