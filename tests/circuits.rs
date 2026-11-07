//! Fresh proofs on circuits written against `ark-relations`, checking that
//! Dynark's QAP reduction works beyond the synthetic workload.
use ark_crypto_primitives::crh::{
    sha256::constraints::{Sha256Gadget, UnitVar},
    CRHSchemeGadget,
};
use ark_ec::pairing::Pairing;
use ark_ff::{Field, One, PrimeField};
use ark_r1cs_std::prelude::*;
use ark_relations::{
    lc, ns,
    r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError, Variable},
};
use ark_std::{
    marker::PhantomData,
    rand::{rngs::StdRng, Rng, RngCore, SeedableRng},
    UniformRand,
};
use dynark::{circuit::synthesize, groth16, prepare_verifying_key};
use sha2::{Digest, Sha256};

/// Proves `circuit` from scratch and checks the proof against
/// `public_inputs`, and that a wrong public input is rejected.
fn prove_and_verify<E: Pairing, C: ConstraintSynthesizer<E::ScalarField>>(
    circuit: C,
    public_inputs: &[E::ScalarField],
    rng: &mut impl Rng,
) {
    let (matrices, instance, witness) = synthesize(circuit).unwrap();
    assert_eq!(instance[0], E::ScalarField::one());
    assert_eq!(&instance[1..], public_inputs);

    let (pk, vk, _) = groth16::setup::<E, _>(matrices.clone(), rng).unwrap();
    let pvk = prepare_verifying_key(&vk);
    let (proof, _) = groth16::prove(&pk, &matrices, &instance, &witness, rng).unwrap();
    assert!(groth16::verify(&pvk, &proof, &instance).unwrap());

    if instance.len() > 1 {
        let mut wrong = instance.clone();
        wrong[1] += E::ScalarField::one();
        assert!(!groth16::verify(&pvk, &proof, &wrong).unwrap());
    }
}

// --- a * b = c, repeated, on several curves ---------------------------------

struct MySillyCircuit<F: Field> {
    a: Option<F>,
    b: Option<F>,
}

impl<ConstraintF: Field> ConstraintSynthesizer<ConstraintF> for MySillyCircuit<ConstraintF> {
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<ConstraintF>,
    ) -> Result<(), SynthesisError> {
        let a = cs.new_witness_variable(|| self.a.ok_or(SynthesisError::AssignmentMissing))?;
        let b = cs.new_witness_variable(|| self.b.ok_or(SynthesisError::AssignmentMissing))?;
        let c = cs.new_input_variable(|| {
            let mut a = self.a.ok_or(SynthesisError::AssignmentMissing)?;
            let b = self.b.ok_or(SynthesisError::AssignmentMissing)?;

            a *= &b;
            Ok(a)
        })?;

        cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + c)?;
        cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + c)?;
        cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + c)?;
        cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + c)?;
        cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + c)?;
        cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + c)?;

        Ok(())
    }
}

fn silly_circuit_on<E: Pairing>() {
    let mut rng = StdRng::seed_from_u64(0);
    for _ in 0..5 {
        let a = E::ScalarField::rand(&mut rng);
        let b = E::ScalarField::rand(&mut rng);
        let circuit = MySillyCircuit {
            a: Some(a),
            b: Some(b),
        };
        prove_and_verify::<E, _>(circuit, &[a * b], &mut rng);
    }
}

#[test]
fn silly_circuit_bls12_377() {
    silly_circuit_on::<ark_bls12_377::Bls12_377>();
}

#[test]
fn silly_circuit_bn254() {
    silly_circuit_on::<ark_bn254::Bn254>();
}

#[test]
fn silly_circuit_bw6_761() {
    silly_circuit_on::<ark_bw6_761::BW6_761>();
}

// --- x^3 + x + 1 = y ---------------------------------------------------------

/// Define a simple circuit that computes x^3 + x + 1 = y
struct CubicPlusLinearCircuit<F: Field> {
    x: Option<F>,
}

impl<ConstraintF: Field> ConstraintSynthesizer<ConstraintF>
    for CubicPlusLinearCircuit<ConstraintF>
{
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<ConstraintF>,
    ) -> Result<(), SynthesisError> {
        // Define variables x, x^2, x^3 and y (i.e. x^3 + x + 1)
        let x = cs.new_witness_variable(|| self.x.ok_or(SynthesisError::AssignmentMissing))?;
        let x_squared = cs.new_witness_variable(|| {
            let mut x_val = self.x.ok_or(SynthesisError::AssignmentMissing)?;
            x_val.square_in_place(); // Calculate x^2
            Ok(x_val)
        })?;
        let x_cubed = cs.new_witness_variable(|| {
            let mut x_val = self.x.ok_or(SynthesisError::AssignmentMissing)?;
            x_val.square_in_place(); // Calculate x^2
            x_val *= &self.x.ok_or(SynthesisError::AssignmentMissing)?; // Calculate x^3
            Ok(x_val)
        })?;
        let y = cs.new_input_variable(|| {
            let mut x_cubed_val = self.x.ok_or(SynthesisError::AssignmentMissing)?;
            x_cubed_val.square_in_place(); // Calculate x^2
            x_cubed_val *= &self.x.ok_or(SynthesisError::AssignmentMissing)?; // Calculate x^3

            let mut result = x_cubed_val; // Initial value is x^3
            result += &self.x.ok_or(SynthesisError::AssignmentMissing)?; // Add x
            result += ConstraintF::one(); // Add 1
            Ok(result) // Return y
        })?;

        // Add constraints: x * x = x^2, x^2 * x = x^3, and x^3 + x + 1 = y
        cs.enforce_constraint(lc!() + x, lc!() + x, lc!() + x_squared)?; // x * x = x^2
        cs.enforce_constraint(lc!() + x_squared, lc!() + x, lc!() + x_cubed)?; // x^2 * x = x^3
        cs.enforce_constraint(
            lc!() + x_cubed + x + (ConstraintF::one(), Variable::One), // x^3 + x + 1
            lc!() + Variable::One,                                     /* 1 (no multiplication
                                                                        * needed) */
            lc!() + y,
        )?; // y = x^3 + x + 1

        Ok(())
    }
}

#[test]
fn cubic_expression() {
    type E = ark_bls12_381::Bls12_381;
    type Fr = <E as Pairing>::ScalarField;
    let mut rng = StdRng::seed_from_u64(0);
    let x = Fr::rand(&mut rng);
    let y = x * x * x + x + Fr::one();
    prove_and_verify::<E, _>(CubicPlusLinearCircuit { x: Some(x) }, &[y], &mut rng);
}

// --- 0 <= x < 2^32 -----------------------------------------------------------

/// Define a circuit to prove x is in range [0, 2^32]
struct RangeProofCircuit<F: Field> {
    x: Option<F>,
}

impl<ConstraintF: PrimeField> ConstraintSynthesizer<ConstraintF>
    for RangeProofCircuit<ConstraintF>
{
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<ConstraintF>,
    ) -> Result<(), SynthesisError> {
        // Declare input variable x
        let x = cs.new_input_variable(|| self.x.ok_or(SynthesisError::AssignmentMissing))?;

        // Decompose x into 32 binary bits
        let mut bits = Vec::new();
        for i in 0..32 {
            let bit = cs.new_witness_variable(|| {
                let x_val = self.x.ok_or(SynthesisError::AssignmentMissing)?;
                let x_bigint = x_val.into_bigint();
                let bit_val = (x_bigint.as_ref()[0] >> i) & 1 == 1;
                Ok(if bit_val {
                    ConstraintF::one()
                } else {
                    ConstraintF::zero()
                })
            })?;

            // Constrain each bit to be 0 or 1: bit * (1 - bit) = 0
            cs.enforce_constraint(
                lc!() + bit,
                lc!() + (ConstraintF::one(), Variable::One) - bit,
                lc!(),
            )?;

            bits.push(bit);
        }

        // Constrain the combination of bits equals x
        let mut lc = lc!();
        let mut coeff = ConstraintF::one();
        for bit in bits.iter() {
            lc = lc + (coeff, *bit);
            coeff = coeff.double();
        }

        // Ensure the combination of bits equals x
        cs.enforce_constraint(
            lc!() + lc,
            lc!() + (ConstraintF::one(), Variable::One),
            lc!() + x,
        )?;

        Ok(())
    }
}

#[test]
fn range_proof() {
    type E = ark_bls12_381::Bls12_381;
    type Fr = <E as Pairing>::ScalarField;
    let mut rng = StdRng::seed_from_u64(0);
    let x = Fr::from(rng.next_u32() as u64);
    prove_and_verify::<E, _>(RangeProofCircuit { x: Some(x) }, &[x], &mut rng);
}

// --- MiMC preimage -----------------------------------------------------------

const MIMC_ROUNDS: usize = 322;

/// This is an implementation of MiMC, specifically a
/// variant named `LongsightF322p3` for BLS12-377.
/// See http://eprint.iacr.org/2016/492 for more
/// information about this construction.
///
/// ```
/// function LongsightF322p3(xL ⦂ Fp, xR ⦂ Fp) {
///     for i from 0 up to 321 {
///         xL, xR := xR + (xL + Ci)^3, xL
///     }
///     return xL
/// }
/// ```
fn mimc<F: Field>(mut xl: F, mut xr: F, constants: &[F]) -> F {
    assert_eq!(constants.len(), MIMC_ROUNDS);

    for i in 0..MIMC_ROUNDS {
        let mut tmp1 = xl;
        tmp1.add_assign(&constants[i]);
        let mut tmp2 = tmp1;
        tmp2.square_in_place();
        tmp2.mul_assign(&tmp1);
        tmp2.add_assign(&xr);
        xr = xl;
        xl = tmp2;
    }

    xl
}

/// This is our demo circuit for proving knowledge of the
/// preimage of a MiMC hash invocation.
struct MiMCDemo<'a, F: Field> {
    xl: Option<F>,
    xr: Option<F>,
    constants: &'a [F],
}

/// Our demo circuit implements this `Circuit` trait which
/// is used during paramgen and proving in order to
/// synthesize the constraint system.
impl<'a, F: Field> ConstraintSynthesizer<F> for MiMCDemo<'a, F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        assert_eq!(self.constants.len(), MIMC_ROUNDS);

        // Allocate the first component of the preimage.
        let mut xl_value = self.xl;
        let mut xl =
            cs.new_witness_variable(|| xl_value.ok_or(SynthesisError::AssignmentMissing))?;

        // Allocate the second component of the preimage.
        let mut xr_value = self.xr;
        let mut xr =
            cs.new_witness_variable(|| xr_value.ok_or(SynthesisError::AssignmentMissing))?;

        for i in 0..MIMC_ROUNDS {
            // xL, xR := xR + (xL + Ci)^3, xL
            let ns = ns!(cs, "round");
            let cs = ns.cs();

            // tmp = (xL + Ci)^2
            let tmp_value = xl_value.map(|mut e| {
                e.add_assign(&self.constants[i]);
                e.square_in_place();
                e
            });
            let tmp =
                cs.new_witness_variable(|| tmp_value.ok_or(SynthesisError::AssignmentMissing))?;

            cs.enforce_constraint(
                lc!() + xl + (self.constants[i], Variable::One),
                lc!() + xl + (self.constants[i], Variable::One),
                lc!() + tmp,
            )?;

            // new_xL = xR + (xL + Ci)^3
            // new_xL = xR + tmp * (xL + Ci)
            // new_xL - xR = tmp * (xL + Ci)
            let new_xl_value = xl_value.map(|mut e| {
                e.add_assign(&self.constants[i]);
                e.mul_assign(&tmp_value.unwrap());
                e.add_assign(&xr_value.unwrap());
                e
            });

            let new_xl = if i == (MIMC_ROUNDS - 1) {
                // This is the last round, xL is our image and so
                // we allocate a public input.
                cs.new_input_variable(|| new_xl_value.ok_or(SynthesisError::AssignmentMissing))?
            } else {
                cs.new_witness_variable(|| new_xl_value.ok_or(SynthesisError::AssignmentMissing))?
            };

            cs.enforce_constraint(
                lc!() + tmp,
                lc!() + xl + (self.constants[i], Variable::One),
                lc!() + new_xl - xr,
            )?;

            // xR = xL
            xr = xl;
            xr_value = xl_value;

            // xL = new_xL
            xl = new_xl;
            xl_value = new_xl_value;
        }

        Ok(())
    }
}

#[test]
fn mimc_preimage() {
    use ark_bls12_377::{Bls12_377, Fr};
    let mut rng = StdRng::seed_from_u64(0);
    let constants = (0..MIMC_ROUNDS).map(|_| rng.gen()).collect::<Vec<Fr>>();
    let xl = rng.gen();
    let xr = rng.gen();
    let image = mimc(xl, xr, &constants);
    let circuit = MiMCDemo {
        xl: Some(xl),
        xr: Some(xr),
        constants: &constants,
    };
    prove_and_verify::<Bls12_377, _>(circuit, &[image], &mut rng);
}

// --- SHA-256 preimage, with the digest as a witness --------------------------

struct Sha256Circuit<ConstraintF: Field> {
    preimage: Option<Vec<u8>>, // Input to be hashed
    hash: Option<Vec<u8>>,     // Expected hash value to verify
    _phantom: PhantomData<ConstraintF>,
}

impl<ConstraintF: PrimeField> ConstraintSynthesizer<ConstraintF> for Sha256Circuit<ConstraintF> {
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<ConstraintF>,
    ) -> Result<(), SynthesisError> {
        // Modification: Make hash value a private input instead of public input
        let preimage_var = UInt8::new_witness_vec(
            ark_relations::ns!(cs, "preimage"),
            self.preimage.as_deref().unwrap_or(&[]),
        )?;

        let hash_var = UInt8::new_witness_vec(
            // Changed to witness instead of input
            ark_relations::ns!(cs, "hash"),
            self.hash.as_deref().unwrap_or(&[]),
        )?;

        // Use SHA256 gadget to compute hash
        let computed_hash =
            Sha256Gadget::<ConstraintF>::evaluate(&UnitVar::default(), &preimage_var)?;

        // Add constraint: computed hash must equal input hash
        for (computed_byte, expected_byte) in computed_hash.0.iter().zip(hash_var.iter()) {
            computed_byte.enforce_equal(expected_byte)?;
        }

        Ok(())
    }
}

#[test]
fn sha256_preimage() {
    type E = ark_bls12_381::Bls12_381;
    let mut rng = StdRng::seed_from_u64(0);
    let preimage = b"Hello, World!".to_vec();
    let hash = Sha256::digest(&preimage).to_vec();
    let circuit = Sha256Circuit {
        preimage: Some(preimage),
        hash: Some(hash),
        _phantom: PhantomData,
    };
    prove_and_verify::<E, _>(circuit, &[], &mut rng);
}
