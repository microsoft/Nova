use crate::bellpepper::test_shape_cs::TestShapeCS;
use crate::gadgets::utils::alloc_const;
use crate::gadgets::utils::alloc_num_equals;
use crate::gadgets::utils::conditionally_select;
use crate::traits::circuit_supernova::TrivialTestCircuit;
use crate::{
  compute_digest,
  gadgets::utils::{add_allocated_num, alloc_one, alloc_zero},
};
use bellpepper::gadgets::boolean::Boolean;
use bellpepper_core::num::AllocatedNum;
use bellpepper_core::{ConstraintSystem, LinearCombination, SynthesisError};
use core::marker::PhantomData;
use ff::Field;
use ff::PrimeField;

use super::*;

fn constraint_augmented_circuit_index<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  pc_counter: &AllocatedNum<F>,
  rom: &[AllocatedNum<F>],
  circuit_index: &AllocatedNum<F>,
) -> Result<(), SynthesisError> {
  // select target when index match or empty
  let zero = alloc_zero(cs.namespace(|| "zero"))?;
  let rom_values = rom
    .iter()
    .enumerate()
    .map(|(i, rom_value)| {
      let index_alloc = alloc_const(
        cs.namespace(|| format!("rom_values {} index ", i)),
        F::from(i as u64),
      )?;
      let equal_bit = Boolean::from(alloc_num_equals(
        cs.namespace(|| format!("rom_values {} equal bit", i)),
        &index_alloc,
        pc_counter,
      )?);
      conditionally_select(
        cs.namespace(|| format!("rom_values {} conditionally_select ", i)),
        rom_value,
        &zero,
        &equal_bit,
      )
    })
    .collect::<Result<Vec<AllocatedNum<F>>, SynthesisError>>()?;

  let sum_lc = rom_values
    .iter()
    .fold(LinearCombination::<F>::zero(), |acc_lc, row_value| {
      acc_lc + row_value.get_variable()
    });

  cs.enforce(
    || "sum_lc == circuit_index",
    |lc| lc + circuit_index.get_variable() - &sum_lc,
    |lc| lc + CS::one(),
    |lc| lc,
  );

  Ok(())
}

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

impl<F> StepCircuit<F> for CubicCircuit<F>
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    1 + self.rom_size // value + rom[].len()
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    pc_counter: &AllocatedNum<F>,
    z: &[AllocatedNum<F>],
  ) -> Result<(AllocatedNum<F>, Vec<AllocatedNum<F>>), SynthesisError> {
    // constrain rom[pc] equal to `self.circuit_index`
    let circuit_index = alloc_const(
      cs.namespace(|| "circuit_index"),
      F::from(self.circuit_index as u64),
    )?;
    constraint_augmented_circuit_index(
      cs.namespace(|| "CubicCircuit agumented circuit constraint"),
      pc_counter,
      &z[1..],
      &circuit_index,
    )?;

    let one = alloc_one(cs.namespace(|| "alloc one"))?;
    let pc_next = add_allocated_num(
      // pc = pc + 1
      cs.namespace(|| "pc = pc + 1".to_string()),
      pc_counter,
      &one,
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
    z_next.extend(z[1..].iter().cloned());
    Ok((pc_next, z_next))
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
    1 + self.rom_size // value + rom[].len()
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    pc_counter: &AllocatedNum<F>,
    z: &[AllocatedNum<F>],
  ) -> Result<(AllocatedNum<F>, Vec<AllocatedNum<F>>), SynthesisError> {
    // constrain rom[pc] equal to `self.circuit_index`
    let circuit_index = alloc_const(
      cs.namespace(|| "circuit_index"),
      F::from(self.circuit_index as u64),
    )?;
    constraint_augmented_circuit_index(
      cs.namespace(|| "SquareCircuit agumented circuit constraint"),
      pc_counter,
      &z[1..],
      &circuit_index,
    )?;
    let one = alloc_one(cs.namespace(|| "alloc one"))?;
    let pc_next = add_allocated_num(
      // pc = pc + 1
      cs.namespace(|| "pc = pc + 1"),
      pc_counter,
      &one,
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
    z_next.extend(z[1..].iter().cloned());
    Ok((pc_next, z_next))
  }
}

fn print_constraints_name_on_error_index<G1, G2, Ca, Cb>(
  err: SuperNovaError,
  running_claim: &RunningClaim<G1, G2, Ca, Cb>,
  num_augmented_circuits: usize,
) where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  Ca: StepCircuit<G1::Scalar>,
  Cb: StepCircuit<G2::Scalar>,
{
  match err {
    SuperNovaError::UnSatIndex(msg, index) if msg == "r_primary" => {
      let circuit_primary: SuperNovaAugmentedCircuit<'_, G2, Ca> = SuperNovaAugmentedCircuit::new(
        &running_claim.params.augmented_circuit_params_primary,
        None,
        &running_claim.c_primary,
        running_claim.params.ro_consts_circuit_primary.clone(),
        num_augmented_circuits,
      );
      let mut cs: TestShapeCS<G1> = TestShapeCS::new();
      let _ = circuit_primary.synthesize(&mut cs);
      cs.constraints
        .get(index)
        .map(|constraint| debug!("{msg} failed at constraint {}", constraint.3));
    }
    SuperNovaError::UnSatIndex(msg, index) if msg == "r_secondary" || msg == "l_secondary" => {
      let circuit_secondary: SuperNovaAugmentedCircuit<'_, G1, Cb> = SuperNovaAugmentedCircuit::new(
        &running_claim.params.augmented_circuit_params_secondary,
        None,
        &running_claim.c_secondary,
        running_claim.params.ro_consts_circuit_secondary.clone(),
        num_augmented_circuits,
      );
      let mut cs: TestShapeCS<G2> = TestShapeCS::new();
      let _ = circuit_secondary.synthesize(&mut cs);
      cs.constraints
        .get(index)
        .map(|constraint| debug!("{msg} failed at constraint {}", constraint.3));
    }
    _ => (),
  }
}

const OPCODE_0: usize = 0;
const OPCODE_1: usize = 1;
fn test_trivial_nivc_with<G1, G2>()
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
{
  // Here demo a simple RAM machine
  // - with 2 argumented circuit
  // - each argumented circuit contains primary and secondary circuit
  // - a memory commmitment via a public IO `rom` (like a program) to constraint the sequence execution

  // This test also ready to add more argumented circuit and ROM can be arbitrary length

  // ROM is for constraints the sequence of execution order for opcode
  // program counter initially point to 0

  // TODO: replace with memory commitment along with suggestion from Supernova 4.4 optimisations

  // This is mostly done with the existing Nova code. With additions of U_i[] and program_counter checks
  // in the augmented circuit.

  // To save the test time, after each step of iteration, RecursiveSNARK just verfiy the latest U_i[augmented_circuit_index] needs to be a satisfying instance.
  // TODO At the end of this test, RecursiveSNARK need to verify all U_i[] are satisfying instances

  let rom = [
    OPCODE_1, OPCODE_1, OPCODE_0, OPCODE_0, OPCODE_1, OPCODE_1, OPCODE_0, OPCODE_0, OPCODE_1,
    OPCODE_1,
  ]; // Rom can be arbitrary length.
  let circuit_secondary = TrivialTestCircuit::new(rom.len());
  let num_augmented_circuit = 2;

  // Structuring running claims
  let test_circuit1 = CubicCircuit::new(OPCODE_0, rom.len());
  let mut running_claim1 = RunningClaim::<
    G1,
    G2,
    CubicCircuit<<G1 as Group>::Scalar>,
    TrivialTestCircuit<<G2 as Group>::Scalar>,
  >::new(
    OPCODE_0,
    test_circuit1,
    circuit_secondary.clone(),
    num_augmented_circuit,
  );

  let test_circuit2 = SquareCircuit::new(OPCODE_1, rom.len());
  let mut running_claim2 = RunningClaim::<
    G1,
    G2,
    SquareCircuit<<G1 as Group>::Scalar>,
    TrivialTestCircuit<<G2 as Group>::Scalar>,
  >::new(
    OPCODE_1,
    test_circuit2,
    circuit_secondary,
    num_augmented_circuit,
  );

  // generate the commitkey based on max num of constraints and reused it for all other augmented circuit
  let circuit_public_params = vec![&running_claim1.params, &running_claim2.params];
  let (max_index_circuit, _) = circuit_public_params
    .iter()
    .enumerate()
    .map(|(i, params)| -> (usize, usize) { (i, params.r1cs_shape_primary.num_cons) })
    .max_by(|(_, circuit_size1), (_, circuit_size2)| circuit_size1.cmp(circuit_size2))
    .unwrap();

  let ck_primary =
    gen_commitmentkey_by_r1cs(&circuit_public_params[max_index_circuit].r1cs_shape_primary);
  let ck_secondary =
    gen_commitmentkey_by_r1cs(&circuit_public_params[max_index_circuit].r1cs_shape_secondary);

  // set unified ck_primary, ck_secondary and update digest
  running_claim1.params.ck_primary = Some(ck_primary.clone());
  running_claim1.params.ck_secondary = Some(ck_secondary.clone());

  running_claim2.params.ck_primary = Some(ck_primary);
  running_claim2.params.ck_secondary = Some(ck_secondary);

  let digest = compute_digest::<G1, PublicParams<G1, G2>>(&[
    running_claim1.get_publicparams(),
    running_claim2.get_publicparams(),
  ]);

  let num_steps = rom.len();
  let initial_program_counter = <G1 as Group>::Scalar::from(0);

  // extend z0_primary/secondary with rom content
  let mut z0_primary = vec![<G1 as Group>::Scalar::ONE];
  z0_primary.extend(
    rom
      .iter()
      .map(|opcode| <G1 as Group>::Scalar::from(*opcode as u64)),
  );
  let mut z0_secondary = vec![<G2 as Group>::Scalar::ONE];
  z0_secondary.extend(
    rom
      .iter()
      .map(|opcode| <G2 as Group>::Scalar::from(*opcode as u64)),
  );

  let mut recursive_snark_option: Option<RecursiveSNARK<G1, G2>> = None;

  for _ in 0..num_steps {
    let program_counter = recursive_snark_option
      .as_ref()
      .map(|recursive_snark| recursive_snark.program_counter)
      .unwrap_or_else(|| initial_program_counter);
    let augmented_circuit_index = rom[u32::from_le_bytes(
      // convert program counter from field to usize (only took le 4 bytes)
      program_counter.to_repr().as_ref()[0..4].try_into().unwrap(),
    ) as usize];

    let mut recursive_snark = recursive_snark_option.unwrap_or_else(|| {
      if augmented_circuit_index == OPCODE_0 {
        RecursiveSNARK::iter_base_step(
          &running_claim1,
          digest,
          program_counter,
          augmented_circuit_index,
          num_augmented_circuit,
          &z0_primary,
          &z0_secondary,
        )
        .unwrap()
      } else if augmented_circuit_index == OPCODE_1 {
        RecursiveSNARK::iter_base_step(
          &running_claim2,
          digest,
          program_counter,
          augmented_circuit_index,
          num_augmented_circuit,
          &z0_primary,
          &z0_secondary,
        )
        .unwrap()
      } else {
        unimplemented!()
      }
    });

    if augmented_circuit_index == OPCODE_0 {
      let _ = recursive_snark
        .prove_step(&running_claim1, &z0_primary, &z0_secondary)
        .unwrap();
      let _ = recursive_snark
        .verify(&running_claim1, &z0_primary, &z0_secondary)
        .map_err(|err| {
          print_constraints_name_on_error_index(err, &running_claim1, num_augmented_circuit)
        })
        .unwrap();
    } else if augmented_circuit_index == OPCODE_1 {
      let _ = recursive_snark
        .prove_step(&running_claim2, &z0_primary, &z0_secondary)
        .unwrap();
      let _ = recursive_snark
        .verify(&running_claim2, &z0_primary, &z0_secondary)
        .map_err(|err| {
          print_constraints_name_on_error_index(err, &running_claim2, num_augmented_circuit)
        })
        .unwrap();
    }
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
}

#[test]
fn test_trivial_nivc() {
  type G1 = pasta_curves::pallas::Point;
  type G2 = pasta_curves::vesta::Point;

  //Expirementing with selecting the running claims for nifs
  test_trivial_nivc_with::<G1, G2>();
}
