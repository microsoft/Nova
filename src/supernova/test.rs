use crate::gadgets::utils::alloc_zero;
use crate::provider::poseidon::PoseidonConstantsCircuit;
use crate::provider::Bn256EngineIPA;
use crate::provider::Bn256EngineKZG;
use crate::provider::GrumpkinEngine;
use crate::provider::PallasEngine;
use crate::provider::Secp256k1Engine;
use crate::provider::Secq256k1Engine;
use crate::provider::VestaEngine;
use crate::supernova::circuit::{
  EnforcingStepCircuit, StepCircuit, TrivialSecondaryCircuit, TrivialTestCircuit,
};
use crate::traits::snark::default_ck_hint;
use crate::{bellpepper::test_shape_cs::TestShapeCS, gadgets::utils::alloc_one};
use bellpepper_core::num::AllocatedNum;
use bellpepper_core::{ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use ff::Field;
use ff::PrimeField;
use std::fmt::Write;
use tap::TapOptional;

use super::{utils::get_selector_vec_from_index, *};

#[derive(Clone, Debug, Default)]
struct CubicCircuit<F: PrimeField> {
  _p: PhantomData<F>,
  circuit_index: usize,
  rom_size: usize,
}

impl<F> CubicCircuit<F>
where
  F: PrimeField,
{
  fn new(circuit_index: usize, rom_size: usize) -> Self {
    CubicCircuit {
      circuit_index,
      rom_size,
      _p: PhantomData,
    }
  }
}

fn next_rom_index_and_pc<F: PrimeField, CS: ConstraintSystem<F>>(
  cs: &mut CS,
  rom_index: &AllocatedNum<F>,
  allocated_rom: &[AllocatedNum<F>],
  pc: &AllocatedNum<F>,
) -> Result<(AllocatedNum<F>, AllocatedNum<F>), SynthesisError> {
  // Compute a selector for the current rom_index in allocated_rom
  let current_rom_selector = get_selector_vec_from_index(
    cs.namespace(|| "rom selector"),
    rom_index,
    allocated_rom.len(),
  )?;

  // Enforce that allocated_rom[rom_index] = pc
  for (rom, bit) in allocated_rom.iter().zip_eq(current_rom_selector.iter()) {
    // if bit = 1, then rom = pc
    // bit * (rom - pc) = 0
    cs.enforce(
      || "enforce bit = 1 => rom = pc",
      |lc| lc + &bit.lc(CS::one(), F::ONE),
      |lc| lc + rom.get_variable() - pc.get_variable(),
      |lc| lc,
    );
  }

  // Get the index of the current rom, or the index of the invalid rom if no match
  let current_rom_index = current_rom_selector
    .iter()
    .position(|bit| bit.get_value().is_some_and(|v| v))
    .unwrap_or_default();
  let next_rom_index = current_rom_index + 1;

  let rom_index_next = AllocatedNum::alloc_infallible(cs.namespace(|| "next rom index"), || {
    F::from(next_rom_index as u64)
  });
  cs.enforce(
    || " rom_index + 1 - next_rom_index_num = 0",
    |lc| lc,
    |lc| lc,
    |lc| lc + rom_index.get_variable() + CS::one() - rom_index_next.get_variable(),
  );

  // Allocate the next pc without checking.
  // The next iteration will check whether the next pc is valid.
  let pc_next = AllocatedNum::alloc_infallible(cs.namespace(|| "next pc"), || {
    allocated_rom
      .get(next_rom_index)
      .and_then(|v| v.get_value())
      .unwrap_or(-F::ONE)
  });

  Ok((rom_index_next, pc_next))
}

impl<F> StepCircuit<F> for CubicCircuit<F>
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    2 + self.rom_size // value + rom_pc + rom[].len()
  }

  fn circuit_index(&self) -> usize {
    self.circuit_index
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    pc: Option<&AllocatedNum<F>>,
    z: &[AllocatedNum<F>],
  ) -> Result<(Option<AllocatedNum<F>>, Vec<AllocatedNum<F>>), SynthesisError> {
    let rom_index = &z[1];
    let allocated_rom = &z[2..];

    let (rom_index_next, pc_next) = next_rom_index_and_pc(
      &mut cs.namespace(|| "next and rom_index and pc"),
      rom_index,
      allocated_rom,
      pc.ok_or(SynthesisError::AssignmentMissing)?,
    )?;

    // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
    let x = &z[0];
    let x_sq = x.square(cs.namespace(|| "x_sq"))?;
    let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), x)?;
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + F::from(5u64))
    })?;

    cs.enforce(
      || "y = x^3 + x + 5",
      |lc| {
        lc + x_cu.get_variable()
          + x.get_variable()
          + CS::one()
          + CS::one()
          + CS::one()
          + CS::one()
          + CS::one()
      },
      |lc| lc + CS::one(),
      |lc| lc + y.get_variable(),
    );

    let mut z_next = vec![y];
    z_next.push(rom_index_next);
    z_next.extend(z[2..].iter().cloned());
    Ok((Some(pc_next), z_next))
  }
}

#[derive(Clone, Debug, Default)]
struct SquareCircuit<F: PrimeField> {
  _p: PhantomData<F>,
  circuit_index: usize,
  rom_size: usize,
}

impl<F> SquareCircuit<F>
where
  F: PrimeField,
{
  fn new(circuit_index: usize, rom_size: usize) -> Self {
    SquareCircuit {
      circuit_index,
      rom_size,
      _p: PhantomData,
    }
  }
}

impl<F> StepCircuit<F> for SquareCircuit<F>
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    2 + self.rom_size // value + rom_pc + rom[].len()
  }

  fn circuit_index(&self) -> usize {
    self.circuit_index
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    pc: Option<&AllocatedNum<F>>,
    z: &[AllocatedNum<F>],
  ) -> Result<(Option<AllocatedNum<F>>, Vec<AllocatedNum<F>>), SynthesisError> {
    let rom_index = &z[1];
    let allocated_rom = &z[2..];

    let (rom_index_next, pc_next) = next_rom_index_and_pc(
      &mut cs.namespace(|| "next and rom_index and pc"),
      rom_index,
      allocated_rom,
      pc.ok_or(SynthesisError::AssignmentMissing)?,
    )?;

    // Consider an equation: `x^2 + x + 5 = y`, where `x` and `y` are respectively the input and output.
    let x = &z[0];
    let x_sq = x.square(cs.namespace(|| "x_sq"))?;
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      Ok(x_sq.get_value().unwrap() + x.get_value().unwrap() + F::from(5u64))
    })?;

    cs.enforce(
      || "y = x^2 + x + 5",
      |lc| {
        lc + x_sq.get_variable()
          + x.get_variable()
          + CS::one()
          + CS::one()
          + CS::one()
          + CS::one()
          + CS::one()
      },
      |lc| lc + CS::one(),
      |lc| lc + y.get_variable(),
    );

    let mut z_next = vec![y];
    z_next.push(rom_index_next);
    z_next.extend(z[2..].iter().cloned());
    Ok((Some(pc_next), z_next))
  }
}

fn print_constraints_name_on_error_index<E1, E2, C1, C2>(
  err: &SuperNovaError,
  pp: &PublicParams<E1, E2, C1, C2>,
  c_primary: &C1,
  c_secondary: &C2,
  num_augmented_circuits: usize,
) where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C1: EnforcingStepCircuit<E1::Scalar>,
  C2: EnforcingStepCircuit<E2::Scalar>,
{
  match err {
    SuperNovaError::UnSatIndex(msg, index) if *msg == "r_primary" => {
      let circuit_primary: SuperNovaAugmentedCircuit<'_, E2, C1> = SuperNovaAugmentedCircuit::new(
        &pp.augmented_circuit_params_primary,
        None,
        c_primary,
        pp.ro_consts_circuit_primary.clone(),
        num_augmented_circuits,
      );
      let mut cs: TestShapeCS<E1> = TestShapeCS::new();
      let _ = circuit_primary.synthesize(&mut cs);
      cs.constraints
        .get(*index)
        .tap_some(|constraint| debug!("{msg} failed at constraint {}", constraint.3));
    }
    SuperNovaError::UnSatIndex(msg, index) if *msg == "r_secondary" || *msg == "l_secondary" => {
      let circuit_secondary: SuperNovaAugmentedCircuit<'_, E1, C2> = SuperNovaAugmentedCircuit::new(
        &pp.augmented_circuit_params_secondary,
        None,
        c_secondary,
        pp.ro_consts_circuit_secondary.clone(),
        num_augmented_circuits,
      );
      let mut cs: TestShapeCS<E2> = TestShapeCS::new();
      let _ = circuit_secondary.synthesize(&mut cs);
      cs.constraints
        .get(*index)
        .tap_some(|constraint| debug!("{msg} failed at constraint {}", constraint.3));
    }
    _ => (),
  }
}

const OPCODE_0: usize = 0;
const OPCODE_1: usize = 1;

struct TestROM<E1, E2, S>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  S: EnforcingStepCircuit<E2::Scalar> + Default,
{
  rom: Vec<usize>,
  _p: PhantomData<(E1, E2, S)>,
}

#[derive(Debug, Clone)]
enum TestROMCircuit<F: PrimeField> {
  Cubic(CubicCircuit<F>),
  Square(SquareCircuit<F>),
}

impl<F: PrimeField> StepCircuit<F> for TestROMCircuit<F> {
  fn arity(&self) -> usize {
    match self {
      Self::Cubic(x) => x.arity(),
      Self::Square(x) => x.arity(),
    }
  }

  fn circuit_index(&self) -> usize {
    match self {
      Self::Cubic(x) => x.circuit_index(),
      Self::Square(x) => x.circuit_index(),
    }
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    pc: Option<&AllocatedNum<F>>,
    z: &[AllocatedNum<F>],
  ) -> Result<(Option<AllocatedNum<F>>, Vec<AllocatedNum<F>>), SynthesisError> {
    match self {
      Self::Cubic(x) => x.synthesize(cs, pc, z),
      Self::Square(x) => x.synthesize(cs, pc, z),
    }
  }
}

impl<E1, E2>
  NonUniformCircuit<E1, E2, TestROMCircuit<E1::Scalar>, TrivialSecondaryCircuit<E2::Scalar>>
  for TestROM<E1, E2, TrivialSecondaryCircuit<E2::Scalar>>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
{
  fn num_circuits(&self) -> usize {
    2
  }

  fn primary_circuit(&self, circuit_index: usize) -> TestROMCircuit<E1::Scalar> {
    match circuit_index {
      0 => TestROMCircuit::Cubic(CubicCircuit::new(circuit_index, self.rom.len())),
      1 => TestROMCircuit::Square(SquareCircuit::new(circuit_index, self.rom.len())),
      _ => panic!("unsupported primary circuit index"),
    }
  }

  fn secondary_circuit(&self) -> TrivialSecondaryCircuit<E2::Scalar> {
    Default::default()
  }

  fn initial_circuit_index(&self) -> usize {
    self.rom[0]
  }
}

impl<E1, E2, S> TestROM<E1, E2, S>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  S: EnforcingStepCircuit<E2::Scalar> + Default,
{
  fn new(rom: Vec<usize>) -> Self {
    Self {
      rom,
      _p: Default::default(),
    }
  }
}

fn test_trivial_nivc_with<E1, E2>()
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
{
  // Here demo a simple RAM machine
  // - with 2 argumented circuit
  // - each argumented circuit contains primary and secondary circuit
  // - a memory commmitment via a public IO `rom` (like a program) to constraint the sequence execution

  // This test also ready to add more argumented circuit and ROM can be arbitrary length

  // ROM is for constraints the sequence of execution order for opcode

  // TODO: replace with memory commitment along with suggestion from Supernova 4.4 optimisations

  // This is mostly done with the existing Nova code. With additions of U_i[] and program_counter checks
  // in the augmented circuit.

  let rom = vec![
    OPCODE_1, OPCODE_1, OPCODE_0, OPCODE_0, OPCODE_1, OPCODE_1, OPCODE_0, OPCODE_0, OPCODE_1,
    OPCODE_1,
  ]; // Rom can be arbitrary length.

  let test_rom = TestROM::<E1, E2, TrivialSecondaryCircuit<E2::Scalar>>::new(rom);

  let pp = PublicParams::setup(&test_rom, &*default_ck_hint(), &*default_ck_hint());

  // extend z0_primary/secondary with rom content
  let mut z0_primary = vec![<E1 as Engine>::Scalar::ONE];
  z0_primary.push(<E1 as Engine>::Scalar::ZERO); // rom_index = 0
  z0_primary.extend(
    test_rom
      .rom
      .iter()
      .map(|opcode| <E1 as Engine>::Scalar::from(*opcode as u64)),
  );
  let z0_secondary = vec![<E2 as Engine>::Scalar::ONE];

  let mut recursive_snark_option: Option<RecursiveSNARK<E1, E2>> = None;

  for &op_code in test_rom.rom.iter() {
    let circuit_primary = test_rom.primary_circuit(op_code);
    let circuit_secondary = test_rom.secondary_circuit();

    let mut recursive_snark = recursive_snark_option.unwrap_or_else(|| {
      RecursiveSNARK::new(
        &pp,
        &test_rom,
        &circuit_primary,
        &circuit_secondary,
        &z0_primary,
        &z0_secondary,
      )
      .unwrap()
    });

    recursive_snark
      .prove_step(&pp, &circuit_primary, &circuit_secondary)
      .unwrap();
    recursive_snark
      .verify(&pp, &z0_primary, &z0_secondary)
      .map_err(|err| {
        print_constraints_name_on_error_index(
          &err,
          &pp,
          &circuit_primary,
          &circuit_secondary,
          test_rom.num_circuits(),
        )
      })
      .unwrap();

    recursive_snark_option = Some(recursive_snark)
  }

  assert!(recursive_snark_option.is_some());

  // Now you can handle the Result using if let
  let RecursiveSNARK {
    zi_primary,
    zi_secondary,
    program_counter,
    ..
  } = &recursive_snark_option.unwrap();

  println!("zi_primary: {:?}", zi_primary);
  println!("zi_secondary: {:?}", zi_secondary);
  println!("final program_counter: {:?}", program_counter);

  // The final program counter should be -1
  assert_eq!(*program_counter, -<E1 as Engine>::Scalar::ONE);
}

#[test]
fn test_trivial_nivc() {
  // Experimenting with selecting the running claims for nifs
  test_trivial_nivc_with::<PallasEngine, VestaEngine>();
}

// In the following we use 1 to refer to the primary, and 2 to refer to the secondary circuit
fn test_recursive_circuit_with<E1, E2>(
  primary_params: &SuperNovaAugmentedCircuitParams,
  secondary_params: &SuperNovaAugmentedCircuitParams,
  ro_consts1: ROConstantsCircuit<E2>,
  ro_consts2: ROConstantsCircuit<E1>,
  num_constraints_primary: usize,
  num_constraints_secondary: usize,
) where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
{
  // Initialize the shape and ck for the primary
  let step_circuit1 = TrivialTestCircuit::default();
  let arity1 = step_circuit1.arity();
  let circuit1: SuperNovaAugmentedCircuit<'_, E2, TrivialTestCircuit<<E2 as Engine>::Base>> =
    SuperNovaAugmentedCircuit::new(primary_params, None, &step_circuit1, ro_consts1.clone(), 2);
  let mut cs: ShapeCS<E1> = ShapeCS::new();
  if let Err(e) = circuit1.synthesize(&mut cs) {
    panic!("{}", e)
  }
  let (shape1, ck1) = cs.r1cs_shape_and_key(&*default_ck_hint());
  assert_eq!(cs.num_constraints(), num_constraints_primary);

  // Initialize the shape and ck for the secondary
  let step_circuit2 = TrivialSecondaryCircuit::default();
  let arity2 = step_circuit2.arity();
  let circuit2: SuperNovaAugmentedCircuit<'_, E1, TrivialSecondaryCircuit<<E1 as Engine>::Base>> =
    SuperNovaAugmentedCircuit::new(
      secondary_params,
      None,
      &step_circuit2,
      ro_consts2.clone(),
      2,
    );
  let mut cs: ShapeCS<E2> = ShapeCS::new();
  if let Err(e) = circuit2.synthesize(&mut cs) {
    panic!("{}", e)
  }
  let (shape2, ck2) = cs.r1cs_shape_and_key(&*default_ck_hint());
  assert_eq!(cs.num_constraints(), num_constraints_secondary);

  // Execute the base case for the primary
  let zero1 = <<E2 as Engine>::Base as Field>::ZERO;
  let z0 = vec![zero1; arity1];
  let mut cs1 = SatisfyingAssignment::<E1>::new();
  let inputs1: SuperNovaAugmentedCircuitInputs<'_, E2> = SuperNovaAugmentedCircuitInputs::new(
    scalar_as_base::<E1>(zero1), // pass zero for testing
    zero1,
    &z0,
    None,
    None,
    None,
    None,
    Some(zero1),
    zero1,
  );
  let step_circuit = TrivialTestCircuit::default();
  let circuit1: SuperNovaAugmentedCircuit<'_, E2, TrivialTestCircuit<<E2 as Engine>::Base>> =
    SuperNovaAugmentedCircuit::new(primary_params, Some(inputs1), &step_circuit, ro_consts1, 2);
  if let Err(e) = circuit1.synthesize(&mut cs1) {
    panic!("{}", e)
  }
  let (inst1, witness1) = cs1.r1cs_instance_and_witness(&shape1, &ck1).unwrap();
  // Make sure that this is satisfiable
  assert!(shape1.is_sat(&ck1, &inst1, &witness1).is_ok());

  // Execute the base case for the secondary
  let zero2 = <<E1 as Engine>::Base as Field>::ZERO;
  let z0 = vec![zero2; arity2];
  let mut cs2 = SatisfyingAssignment::<E2>::new();
  let inputs2: SuperNovaAugmentedCircuitInputs<'_, E1> = SuperNovaAugmentedCircuitInputs::new(
    scalar_as_base::<E2>(zero2), // pass zero for testing
    zero2,
    &z0,
    None,
    None,
    Some(&inst1),
    None,
    None,
    zero2,
  );
  let step_circuit = TrivialSecondaryCircuit::default();
  let circuit2: SuperNovaAugmentedCircuit<'_, E1, TrivialSecondaryCircuit<<E1 as Engine>::Base>> =
    SuperNovaAugmentedCircuit::new(
      secondary_params,
      Some(inputs2),
      &step_circuit,
      ro_consts2,
      2,
    );
  if let Err(e) = circuit2.synthesize(&mut cs2) {
    panic!("{}", e)
  }
  let (inst2, witness2) = cs2.r1cs_instance_and_witness(&shape2, &ck2).unwrap();
  // Make sure that it is satisfiable
  assert!(shape2.is_sat(&ck2, &inst2, &witness2).is_ok());
}

#[test]
fn test_recursive_circuit() {
  let params1 = SuperNovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);
  let params2 = SuperNovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, false);
  let ro_consts1: ROConstantsCircuit<VestaEngine> = PoseidonConstantsCircuit::default();
  let ro_consts2: ROConstantsCircuit<PallasEngine> = PoseidonConstantsCircuit::default();

  test_recursive_circuit_with::<PallasEngine, VestaEngine>(
    &params1, &params2, ro_consts1, ro_consts2, 9836, 12017,
  );
}

fn test_pp_digest_with<E1, E2, T1, T2, NC>(non_uniform_circuit: &NC, expected: &str)
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  T1: StepCircuit<E1::Scalar>,
  T2: StepCircuit<E2::Scalar>,
  NC: NonUniformCircuit<E1, E2, T1, T2>,
{
  // TODO: add back in https://github.com/lurk-lab/arecibo/issues/53
  // // this tests public parameters with a size specifically intended for a spark-compressed SNARK
  // let pp_hint1 = Some(SPrime::<G1>::commitment_key_floor());
  // let pp_hint2 = Some(SPrime::<G2>::commitment_key_floor());
  let pp = PublicParams::<E1, E2, T1, T2>::setup(
    non_uniform_circuit,
    &*default_ck_hint(),
    &*default_ck_hint(),
  );

  let digest_str = pp
    .digest()
    .to_repr()
    .as_ref()
    .iter()
    .fold(String::new(), |mut output, b| {
      let _ = write!(output, "{b:02x}");
      output
    });
  assert_eq!(digest_str, expected);
}

#[test]
fn test_supernova_pp_digest() {
  let rom = vec![
    OPCODE_1, OPCODE_1, OPCODE_0, OPCODE_0, OPCODE_1, OPCODE_1, OPCODE_0, OPCODE_0, OPCODE_1,
    OPCODE_1,
  ]; // Rom can be arbitrary length.
  let test_rom = TestROM::<
    PallasEngine,
    VestaEngine,
    TrivialSecondaryCircuit<<VestaEngine as Engine>::Scalar>,
  >::new(rom);

  test_pp_digest_with::<PallasEngine, VestaEngine, _, _, _>(
    &test_rom,
    "e29765288cd7ebc9d717e0bac1bb096d6199adb64a82001b9fdd261536eb9c01",
  );

  let rom = vec![
    OPCODE_1, OPCODE_1, OPCODE_0, OPCODE_0, OPCODE_1, OPCODE_1, OPCODE_0, OPCODE_0, OPCODE_1,
    OPCODE_1,
  ]; // Rom can be arbitrary length.
  let test_rom_grumpkin = TestROM::<
    Bn256EngineIPA,
    GrumpkinEngine,
    TrivialSecondaryCircuit<<GrumpkinEngine as Engine>::Scalar>,
  >::new(rom);

  test_pp_digest_with::<Bn256EngineIPA, GrumpkinEngine, _, _, _>(
    &test_rom_grumpkin,
    "5444078c7a488f2e64205a796d6dfde84a5cf145dc477772dbc1a084efd54f03",
  );

  let rom = vec![
    OPCODE_1, OPCODE_1, OPCODE_0, OPCODE_0, OPCODE_1, OPCODE_1, OPCODE_0, OPCODE_0, OPCODE_1,
    OPCODE_1,
  ]; // Rom can be arbitrary length.
  let test_rom_secp = TestROM::<
    Secp256k1Engine,
    Secq256k1Engine,
    TrivialSecondaryCircuit<<Secq256k1Engine as Engine>::Scalar>,
  >::new(rom);

  test_pp_digest_with::<Secp256k1Engine, Secq256k1Engine, _, _, _>(
    &test_rom_secp,
    "da33f43a9994c7063b10aed412eaefb5e5e6cfcebc72e4c9ab563b2ae6a0d303",
  );
}

// y is a non-deterministic hint representing the cube root of the input at a step.
#[derive(Clone, Debug)]
struct CubeRootCheckingCircuit<F: PrimeField> {
  y: Option<F>,
}

impl<F> StepCircuit<F> for CubeRootCheckingCircuit<F>
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    1
  }

  fn circuit_index(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    _pc: Option<&AllocatedNum<F>>,
    z: &[AllocatedNum<F>],
  ) -> Result<(Option<AllocatedNum<F>>, Vec<AllocatedNum<F>>), SynthesisError> {
    let x = &z[0];

    // we allocate a variable and set it to the provided non-deterministic hint.
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      self.y.ok_or(SynthesisError::AssignmentMissing)
    })?;

    // We now check if y = x^{1/3} by checking if y^3 = x
    let y_sq = y.square(cs.namespace(|| "y_sq"))?;
    let y_cube = y_sq.mul(cs.namespace(|| "y_cube"), &y)?;

    cs.enforce(
      || "y^3 = x",
      |lc| lc + y_cube.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + x.get_variable(),
    );

    let next_pc = alloc_one(&mut cs.namespace(|| "next_pc"));

    Ok((Some(next_pc), vec![y]))
  }
}

// y is a non-deterministic hint representing the fifth root of the input at a step.
#[derive(Clone, Debug)]
struct FifthRootCheckingCircuit<F: PrimeField> {
  y: Option<F>,
}

impl<F> StepCircuit<F> for FifthRootCheckingCircuit<F>
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    1
  }

  fn circuit_index(&self) -> usize {
    1
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    _pc: Option<&AllocatedNum<F>>,
    z: &[AllocatedNum<F>],
  ) -> Result<(Option<AllocatedNum<F>>, Vec<AllocatedNum<F>>), SynthesisError> {
    let x = &z[0];

    // we allocate a variable and set it to the provided non-deterministic hint.
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      self.y.ok_or(SynthesisError::AssignmentMissing)
    })?;

    // We now check if y = x^{1/5} by checking if y^5 = x
    let y_sq = y.square(cs.namespace(|| "y_sq"))?;
    let y_quad = y_sq.square(cs.namespace(|| "y_quad"))?;
    let y_pow_5 = y_quad.mul(cs.namespace(|| "y_fifth"), &y)?;

    cs.enforce(
      || "y^5 = x",
      |lc| lc + y_pow_5.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + x.get_variable(),
    );

    let next_pc = alloc_zero(&mut cs.namespace(|| "next_pc"));

    Ok((Some(next_pc), vec![y]))
  }
}

#[derive(Clone, Debug)]
enum RootCheckingCircuit<F: PrimeField> {
  Cube(CubeRootCheckingCircuit<F>),
  Fifth(FifthRootCheckingCircuit<F>),
}

impl<F: PrimeField> RootCheckingCircuit<F> {
  fn new(num_steps: usize) -> (Vec<F>, Vec<Self>) {
    let mut powers = Vec::new();
    let rng = &mut rand::rngs::OsRng;
    let mut seed = F::random(rng);

    for i in 0..num_steps + 1 {
      let seed_sq = seed.clone().square();
      // Cube-root and fifth-root circuits alternate. We compute the hints backward, so the calculations appear to be
      // associated with the 'wrong' circuit. The final circuit is discarded, and only the final seed is used (as z_0).
      powers.push(if i % 2 == num_steps % 2 {
        seed *= seed_sq;
        Self::Fifth(FifthRootCheckingCircuit { y: Some(seed) })
      } else {
        seed *= seed_sq.clone().square();
        Self::Cube(CubeRootCheckingCircuit { y: Some(seed) })
      })
    }

    // reverse the powers to get roots
    let roots = powers.into_iter().rev().collect::<Vec<Self>>();
    (vec![roots[0].get_y().unwrap()], roots[1..].to_vec())
  }

  fn get_y(&self) -> Option<F> {
    match self {
      Self::Fifth(x) => x.y,
      Self::Cube(x) => x.y,
    }
  }
}

impl<F> StepCircuit<F> for RootCheckingCircuit<F>
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    1
  }

  fn circuit_index(&self) -> usize {
    match self {
      Self::Cube(x) => x.circuit_index(),
      Self::Fifth(x) => x.circuit_index(),
    }
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    pc: Option<&AllocatedNum<F>>,
    z: &[AllocatedNum<F>],
  ) -> Result<(Option<AllocatedNum<F>>, Vec<AllocatedNum<F>>), SynthesisError> {
    match self {
      Self::Cube(c) => c.synthesize(cs, pc, z),
      Self::Fifth(c) => c.synthesize(cs, pc, z),
    }
  }
}

impl<E1, E2>
  NonUniformCircuit<E1, E2, RootCheckingCircuit<E1::Scalar>, TrivialSecondaryCircuit<E1::Base>>
  for RootCheckingCircuit<E1::Scalar>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
{
  fn num_circuits(&self) -> usize {
    2
  }

  fn primary_circuit(&self, circuit_index: usize) -> Self {
    match circuit_index {
      0 => Self::Cube(CubeRootCheckingCircuit { y: None }),
      1 => Self::Fifth(FifthRootCheckingCircuit { y: None }),
      _ => unreachable!(),
    }
  }

  fn secondary_circuit(&self) -> TrivialSecondaryCircuit<E1::Base> {
    TrivialSecondaryCircuit::<E1::Base>::default()
  }
}

fn test_nivc_nondet_with<E1, E2>()
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
{
  let circuit_secondary = TrivialSecondaryCircuit::default();

  let num_steps = 3;

  // produce non-deterministic hint
  let (z0_primary, roots) = RootCheckingCircuit::new(num_steps);
  assert_eq!(num_steps, roots.len());
  let z0_secondary = vec![<E2 as Engine>::Scalar::ZERO];

  // produce public parameters
  let pp = PublicParams::<
    E1,
    E2,
    RootCheckingCircuit<<E1 as Engine>::Scalar>,
    TrivialSecondaryCircuit<<E2 as Engine>::Scalar>,
  >::setup(&roots[0], &*default_ck_hint(), &*default_ck_hint());
  // produce a recursive SNARK

  let circuit_primary = &roots[0];

  let mut recursive_snark = RecursiveSNARK::<E1, E2>::new(
    &pp,
    circuit_primary,
    circuit_primary,
    &circuit_secondary,
    &z0_primary,
    &z0_secondary,
  )
  .map_err(|err| {
    print_constraints_name_on_error_index(&err, &pp, circuit_primary, &circuit_secondary, 2)
  })
  .unwrap();

  for circuit_primary in roots.iter().take(num_steps) {
    let res = recursive_snark.prove_step(&pp, circuit_primary, &circuit_secondary);
    assert!(res
      .map_err(|err| {
        print_constraints_name_on_error_index(&err, &pp, circuit_primary, &circuit_secondary, 2)
      })
      .is_ok());

    // verify the recursive SNARK
    let res = recursive_snark
      .verify(&pp, &z0_primary, &z0_secondary)
      .map_err(|err| {
        print_constraints_name_on_error_index(&err, &pp, circuit_primary, &circuit_secondary, 2)
      });
    assert!(res.is_ok());
  }
}

#[test]
fn test_nivc_nondet() {
  test_nivc_nondet_with::<PallasEngine, VestaEngine>();
  test_nivc_nondet_with::<Bn256EngineKZG, GrumpkinEngine>();
  test_nivc_nondet_with::<Secp256k1Engine, Secq256k1Engine>();
}
