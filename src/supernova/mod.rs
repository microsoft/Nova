//! This module implements SuperNova NIVC.
// https://eprint.iacr.org/2022/1758.pdf
// By Wyatt Benno 2023.
#![allow(unused_imports)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use crate::ccs;
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NUM_FE_WITHOUT_IO_FOR_CRHF, NUM_HASH_BITS},
  errors::NovaError,
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness},
  scalar_as_base,
  traits::{
    circuit::StepCircuit,
    circuit::TrivialTestCircuit,
    commitment::{CommitmentEngineTrait, CommitmentTrait},
    AbsorbInROTrait, Group, ROConstants, ROConstantsCircuit, ROConstantsTrait, ROTrait,
  },
  Commitment, CommitmentKey, CompressedCommitment,
};

use core::marker::PhantomData;
use ff::Field;
use ff::PrimeField;
use serde::{Deserialize, Serialize};
use std::default;

use crate::bellperson::{
  r1cs::{NovaShape, NovaWitness},
  shape_cs::ShapeCS,
  solver::SatisfyingAssignment,
};
use ::bellperson::{Circuit, ConstraintSystem};
use flate2::{write::ZlibEncoder, Compression};
use sha3::{Digest, Sha3_256};

use crate::nifs::NIFS;

mod circuit; // declare the module first
use circuit::{CircuitInputs, CircuitParams, SuperNovaCircuit};

fn compute_digest<G: Group, T: Serialize>(o: &T) -> G::Scalar {
  // obtain a vector of bytes representing public parameters
  let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
  bincode::serialize_into(&mut encoder, o).unwrap();
  let pp_bytes = encoder.finish().unwrap();

  // convert pp_bytes into a short digest
  let mut hasher = Sha3_256::new();
  hasher.input(&pp_bytes);
  let digest = hasher.result();

  // truncate the digest to NUM_HASH_BITS bits
  let bv = (0..NUM_HASH_BITS).map(|i| {
    let (byte_pos, bit_pos) = (i / 8, i % 8);
    let bit = (digest[byte_pos] >> bit_pos) & 1;
    bit == 1
  });

  // turn the bit vector into a scalar
  let mut digest = G::Scalar::ZERO;
  let mut coeff = G::Scalar::ONE;
  for bit in bv {
    if bit {
      digest += coeff;
    }
    coeff += coeff;
  }
  digest
}

/// A type that holds public parameters of Nova
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct PublicParams<G1, G2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
{
  F_arity_primary: usize,
  F_arity_secondary: usize,
  ro_consts_primary: ROConstants<G1>,
  ro_consts_circuit_primary: ROConstantsCircuit<G2>,
  ck_primary: Option<CommitmentKey<G1>>,
  r1cs_shape_primary: R1CSShape<G1>,
  ro_consts_secondary: ROConstants<G2>,
  ro_consts_circuit_secondary: ROConstantsCircuit<G1>,
  ck_secondary: Option<CommitmentKey<G2>>,
  r1cs_shape_secondary: R1CSShape<G2>,
  augmented_circuit_params_primary: CircuitParams,
  augmented_circuit_params_secondary: CircuitParams,
  digest: G1::Scalar, // digest of everything else with this field set to G1::Scalar::ZERO
  constrain_paths_primary: Vec<String>,
  constrain_paths_secondary: Vec<String>,
}

impl<G1, G2> PublicParams<G1, G2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
{
  /// Create a new `PublicParams`
  pub fn setup_without_commitkey<C1: StepCircuit<G1::Scalar>, C2: StepCircuit<G2::Scalar>>(
    c_primary: C1,
    c_secondary: C2,
    output_U_i_length: usize,
  ) -> Self where {
    let augmented_circuit_params_primary = CircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);
    let augmented_circuit_params_secondary = CircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, false);

    let ro_consts_primary: ROConstants<G1> = ROConstants::<G1>::new();
    let ro_consts_secondary: ROConstants<G2> = ROConstants::<G2>::new();

    let F_arity_primary = c_primary.arity();
    let F_arity_secondary = c_secondary.arity();

    // ro_consts_circuit_primary are parameterized by G2 because the type alias uses G2::Base = G1::Scalar
    let ro_consts_circuit_primary: ROConstantsCircuit<G2> = ROConstantsCircuit::<G2>::new();
    let ro_consts_circuit_secondary: ROConstantsCircuit<G1> = ROConstantsCircuit::<G1>::new();

    // Initialize ck for the primary
    let circuit_primary: SuperNovaCircuit<G2, C1> = SuperNovaCircuit::new(
      augmented_circuit_params_primary.clone(),
      None,
      c_primary,
      ro_consts_circuit_primary.clone(),
      output_U_i_length,
    );
    let mut cs: ShapeCS<G1> = ShapeCS::new();
    let _ = circuit_primary.synthesize(&mut cs);

    // We use the largest commitment_key for all instances
    let (r1cs_shape_primary, constrain_paths_primary) = cs.r1cs_shape_without_commitkey();

    // Initialize ck for the secondary
    let circuit_secondary: SuperNovaCircuit<G1, C2> = SuperNovaCircuit::new(
      augmented_circuit_params_secondary.clone(),
      None,
      c_secondary,
      ro_consts_circuit_secondary.clone(),
      output_U_i_length,
    );
    let mut cs: ShapeCS<G2> = ShapeCS::new();
    let _ = circuit_secondary.synthesize(&mut cs);

    let (r1cs_shape_secondary, constrain_paths_secondary) = cs.r1cs_shape_without_commitkey();

    let mut pp = Self {
      F_arity_primary,
      F_arity_secondary,
      ro_consts_primary,
      ro_consts_circuit_primary,
      ck_primary: None,
      r1cs_shape_primary,
      ro_consts_secondary,
      ro_consts_circuit_secondary,
      ck_secondary: None,
      r1cs_shape_secondary,
      augmented_circuit_params_primary,
      augmented_circuit_params_secondary,
      digest: G1::Scalar::ZERO, // digest will be set later once commitkey ready
      constrain_paths_primary,
      constrain_paths_secondary,
    };

    pp
  }

  /// Returns the number of constraints in the primary and secondary circuits
  pub fn num_constraints(&self) -> (usize, usize) {
    (
      self.r1cs_shape_primary.num_cons,
      self.r1cs_shape_secondary.num_cons,
    )
  }

  /// Returns the number of variables in the primary and secondary circuits
  pub fn num_variables(&self) -> (usize, usize) {
    (
      self.r1cs_shape_primary.num_vars,
      self.r1cs_shape_secondary.num_vars,
    )
  }
}

/*
 SuperNova takes Ui a list of running instances.
 One instance of Ui is a struct called RunningClaim.
*/
#[derive(Clone)]
pub struct RunningClaim<G1, G2, Ca, Cb>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  Ca: StepCircuit<G1::Scalar>,
  Cb: StepCircuit<G2::Scalar>,
{
  _phantom: PhantomData<G1>,
  claim: Ca,
  circuit_secondary: Cb,
  largest: bool,
  params: PublicParams<G1, G2>,
  output_U_i_length: usize,
}

impl<G1, G2, Ca, Cb> RunningClaim<G1, G2, Ca, Cb>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  Ca: StepCircuit<G1::Scalar>,
  Cb: StepCircuit<G2::Scalar>,
{
  pub fn new(
    circuit_primary: Ca,
    circuit_secondary: Cb,
    is_largest: bool,
    output_U_i_length: usize,
  ) -> Self {
    let claim = circuit_primary.clone();
    let largest = is_largest;

    let pp = PublicParams::<G1, G2>::setup_without_commitkey(
      claim.clone(),
      circuit_secondary.clone(),
      output_U_i_length,
    );

    Self {
      _phantom: PhantomData,
      claim,
      circuit_secondary: circuit_secondary.clone(),
      largest,
      params: pp,
      output_U_i_length: output_U_i_length,
    }
  }
}

/// A SNARK that proves the correct execution of an non-uniform incremental computation
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NivcSNARK<G1, G2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
{
  r_W_primary: Vec<Option<RelaxedR1CSWitness<G1>>>,
  r_U_primary: Vec<Option<RelaxedR1CSInstance<G1>>>,
  r_W_secondary: Vec<Option<RelaxedR1CSWitness<G2>>>,
  r_U_secondary: Vec<Option<RelaxedR1CSInstance<G2>>>,
  l_w_secondary: R1CSWitness<G2>,
  l_u_secondary: R1CSInstance<G2>,
  pp_digest: G1::Scalar,
  i: usize,
  zi_primary: Vec<G1::Scalar>,
  zi_secondary: Vec<G2::Scalar>,
  program_counter: usize,
  last_circuit_index_selector: usize,
}

impl<G1, G2> NivcSNARK<G1, G2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
{
  /// Create a new `NivcSNARK` (or updates the provided `NivcSNARK`)
  /// by executing a step of the incremental computation
  pub fn prove_step<C1: StepCircuit<G1::Scalar>, C2: StepCircuit<G2::Scalar>>(
    circuit_index: usize,
    u_i_length: usize,
    program_counter: usize,
    pp: &PublicParams<G1, G2>,
    recursive_snark: Option<Self>,
    c_primary: C1,
    c_secondary: C2,
    z0_primary: Vec<G1::Scalar>,
    z0_secondary: Vec<G2::Scalar>,
    ck_primary: &CommitmentKey<G1>,
    ck_secondary: &CommitmentKey<G2>,
  ) -> Result<Self, NovaError> {
    if z0_primary.len() != pp.F_arity_primary || z0_secondary.len() != pp.F_arity_secondary {
      return Err(NovaError::InvalidInitialInputLength);
    }

    match recursive_snark {
      None => {
        // base case for the primary
        let mut cs_primary: SatisfyingAssignment<G1> = SatisfyingAssignment::new();
        let inputs_primary: CircuitInputs<G2> = CircuitInputs::new(
          scalar_as_base::<G1>(pp.digest),
          G2::Scalar::ZERO,
          G1::Scalar::ZERO,
          z0_primary.clone(),
          None,
          None,
          None,
          None,
          G1::Scalar::from(program_counter as u64), //G1::Scalar::ZERO,
          G1::Scalar::ZERO,                         // set circuit index selector to 0 in base case
        );

        let circuit_primary: SuperNovaCircuit<G2, C1> = SuperNovaCircuit::new(
          pp.augmented_circuit_params_primary.clone(),
          Some(inputs_primary),
          c_primary.clone(),
          pp.ro_consts_circuit_primary.clone(),
          u_i_length,
        );

        let pc_value: Option<G2::Base>;
        if let Some(pc) = circuit_primary.output_program_counter() {
          //println!("Program counter: {:?}", pc);
          pc_value = Some(pc);
        }

        let _ = circuit_primary.synthesize(&mut cs_primary);
        let (u_primary, w_primary) = cs_primary
          .r1cs_instance_and_witness(&pp.r1cs_shape_primary, ck_primary)
          .map_err(|_e| NovaError::UnSat)?;

        // base case for the secondary
        let mut cs_secondary: SatisfyingAssignment<G2> = SatisfyingAssignment::new();
        let inputs_secondary: CircuitInputs<G1> = CircuitInputs::new(
          pp.digest,
          G1::Scalar::ZERO,
          G2::Scalar::ZERO,
          z0_secondary.clone(),
          None,
          None,
          Some(u_primary.clone()),
          None,
          G2::Scalar::from(program_counter as u64), //G2::Scalar::ZERO,
          G2::Scalar::ZERO,                         // set circuit index selector to 0 in base case
        );
        let circuit_secondary: SuperNovaCircuit<G1, C2> = SuperNovaCircuit::new(
          pp.augmented_circuit_params_secondary.clone(),
          Some(inputs_secondary),
          c_secondary.clone(),
          pp.ro_consts_circuit_secondary.clone(),
          u_i_length,
        );
        let _ = circuit_secondary.synthesize(&mut cs_secondary);
        let (u_secondary, w_secondary) = cs_secondary
          .r1cs_instance_and_witness(&pp.r1cs_shape_secondary, ck_secondary)
          .map_err(|_e| NovaError::UnSat)?;

        // IVC proof for the primary circuit
        let l_w_primary = w_primary;
        let l_u_primary = u_primary;
        let r_W_primary =
          RelaxedR1CSWitness::from_r1cs_witness(&pp.r1cs_shape_primary, &l_w_primary);

        let r_U_primary =
          RelaxedR1CSInstance::from_r1cs_instance(ck_primary, &pp.r1cs_shape_primary, &l_u_primary);

        // IVC proof of the secondary circuit
        let l_w_secondary = w_secondary;
        let l_u_secondary = u_secondary;
        let r_W_secondary = RelaxedR1CSWitness::<G2>::default(&pp.r1cs_shape_secondary);
        let r_U_secondary = RelaxedR1CSInstance::default(ck_secondary, &pp.r1cs_shape_secondary);

        // Outputs of the two circuits thus far.
        // For the base case they are just taking the inputs of the function.
        // This is also true of our program_counter and output_U_i.
        let zi_primary = c_primary.output(&z0_primary);
        let zi_secondary = c_secondary.output(&z0_secondary);

        if zi_primary.len() != pp.F_arity_primary || zi_secondary.len() != pp.F_arity_secondary {
          return Err(NovaError::InvalidStepOutputLength);
        }

        // handle the base case by initialize U_next in next round
        let mut r_W_primary_initial_list = (0..u_i_length)
          .map(|_| None)
          .collect::<Vec<Option<RelaxedR1CSWitness<G1>>>>();
        r_W_primary_initial_list[circuit_index] = Some(r_W_primary);

        let mut r_U_primary_initial_list = (0..u_i_length)
          .map(|_| None)
          .collect::<Vec<Option<RelaxedR1CSInstance<G1>>>>();
        r_U_primary_initial_list[circuit_index] = Some(r_U_primary);

        let mut r_W_secondary_initial_list = (0..u_i_length)
          .map(|_| None)
          .collect::<Vec<Option<RelaxedR1CSWitness<G2>>>>();
        r_W_secondary_initial_list[circuit_index] = Some(r_W_secondary);

        let mut r_U_recondary_initial_list = (0..u_i_length)
          .map(|_| None)
          .collect::<Vec<Option<RelaxedR1CSInstance<G2>>>>();
        r_U_recondary_initial_list[circuit_index] = Some(r_U_secondary);

        Ok(Self {
          r_W_primary: r_W_primary_initial_list,
          r_U_primary: r_U_primary_initial_list,
          r_W_secondary: r_W_secondary_initial_list,
          r_U_secondary: r_U_recondary_initial_list,
          l_w_secondary,
          l_u_secondary,
          pp_digest: pp.digest,
          i: 1_usize, // after base case, next iteration start from 1
          zi_primary,
          zi_secondary,
          program_counter: 1_usize,
          last_circuit_index_selector: circuit_index,
        })
      }
      Some(r_snark) => {
        // snark program_counter iteration is equal to pci
        assert!(r_snark.program_counter == program_counter);
        // fold the secondary circuit's instance
        let (nifs_secondary, (r_U_secondary_next, r_W_secondary_next)) = NIFS::prove_supernova(
          ck_secondary,
          &pp.ro_consts_secondary,
          &scalar_as_base::<G1>(r_snark.pp_digest),
          &pp.r1cs_shape_secondary,
          r_snark.last_circuit_index_selector,
          // FIXME: r_U_secondary should be r_U_primary, r_W_secondary should be r_W_primary
          r_snark.r_U_secondary.clone(),
          r_snark.r_W_secondary.clone(),
          &r_snark.l_u_secondary,
          &r_snark.l_w_secondary,
        )?;

        let mut cs_primary: SatisfyingAssignment<G1> = SatisfyingAssignment::new();
        let inputs_primary: CircuitInputs<G2> = CircuitInputs::new(
          scalar_as_base::<G1>(pp.digest),
          scalar_as_base::<G1>(r_snark.pp_digest),
          G1::Scalar::from(r_snark.i as u64),
          z0_primary,
          Some(r_snark.zi_primary.clone()),
          Some(
            r_snark
              .r_U_secondary
              .iter()
              .map(|U| {
                U.clone().unwrap_or_else(|| {
                  RelaxedR1CSInstance::default(ck_secondary, &pp.r1cs_shape_secondary)
                })
              })
              .collect(),
          ),
          Some(r_snark.l_u_secondary.clone()),
          Some(Commitment::<G2>::decompress(&nifs_secondary.comm_T)?),
          G1::Scalar::from(r_snark.program_counter as u64),
          G1::Scalar::from(r_snark.last_circuit_index_selector as u64),
        );

        let circuit_primary: SuperNovaCircuit<G2, C1> = SuperNovaCircuit::new(
          pp.augmented_circuit_params_primary.clone(),
          Some(inputs_primary),
          c_primary.clone(),
          pp.ro_consts_circuit_primary.clone(),
          u_i_length,
        );
        if let Some(pc) = circuit_primary.output_program_counter() {
          //println!("Program counter2: {:?}", pc);
        };
        let _ = circuit_primary.synthesize(&mut cs_primary);

        let (l_u_primary, l_w_primary) = cs_primary
          .r1cs_instance_and_witness(&pp.r1cs_shape_primary, ck_primary)
          .map_err(|_e| NovaError::UnSat)?;

        let (nifs_primary, (r_U_primary_next, r_W_primary_next)) = NIFS::prove_supernova(
          ck_primary,
          &pp.ro_consts_primary,
          &r_snark.pp_digest,
          &pp.r1cs_shape_primary,
          circuit_index,
          r_snark.r_U_primary.to_vec(),
          r_snark.r_W_primary.to_vec(),
          &l_u_primary,
          &l_w_primary,
        )?;

        let mut cs_secondary: SatisfyingAssignment<G2> = SatisfyingAssignment::new();
        let inputs_secondary: CircuitInputs<G1> = CircuitInputs::new(
          pp.digest,
          r_snark.pp_digest,
          G2::Scalar::from(r_snark.i as u64),
          z0_secondary,
          Some(r_snark.zi_secondary.clone()),
          Some(
            r_snark
              .r_U_primary
              .iter()
              .map(|U| {
                U.clone().unwrap_or_else(|| {
                  RelaxedR1CSInstance::default(ck_primary, &pp.r1cs_shape_primary)
                })
              })
              .collect(),
          ),
          Some(l_u_primary),
          Some(Commitment::<G1>::decompress(&nifs_primary.comm_T)?),
          G2::Scalar::from(r_snark.program_counter as u64),
          G2::Scalar::from(circuit_index as u64),
        );

        let circuit_secondary: SuperNovaCircuit<G1, C2> = SuperNovaCircuit::new(
          pp.augmented_circuit_params_secondary.clone(),
          Some(inputs_secondary),
          c_secondary.clone(),
          pp.ro_consts_circuit_secondary.clone(),
          u_i_length,
        );
        let _ = circuit_secondary.synthesize(&mut cs_secondary);

        let (l_u_secondary_next, l_w_secondary_next) = cs_secondary
          .r1cs_instance_and_witness(&pp.r1cs_shape_secondary, ck_secondary)
          .map_err(|_e| NovaError::UnSat)?;

        // update the running instances and witnesses
        let zi_primary = c_primary.output(&r_snark.zi_primary);
        let zi_secondary = c_secondary.output(&r_snark.zi_secondary);

        // FIXME: program counter should be get from primary circuit
        let program_counter_next = program_counter + 1;

        Ok(Self {
          r_W_primary: r_W_primary_next,
          r_U_primary: r_U_primary_next,
          r_W_secondary: r_W_secondary_next,
          r_U_secondary: r_U_secondary_next,
          l_w_secondary: l_w_secondary_next,
          l_u_secondary: l_u_secondary_next,
          pp_digest: pp.digest,
          i: r_snark.i + 1,
          zi_primary,
          zi_secondary,
          program_counter: program_counter_next,
          last_circuit_index_selector: circuit_index,
        })
      }
    }
  }

  pub fn verify(
    &mut self,
    pp: &PublicParams<G1, G2>,
    circuit_index: usize,
    u_i_length: usize,
    z0_primary: Vec<G1::Scalar>,
    z0_secondary: Vec<G2::Scalar>,
    ck_primary: &CommitmentKey<G1>,
    ck_secondary: &CommitmentKey<G2>,
  ) -> Result<(Vec<G1::Scalar>, Vec<G2::Scalar>, usize), NovaError> {
    // number of steps cannot be zero
    if self.i == 0 {
      println!("must verify on valid NivcSNARK where i > 0");
      return Err(NovaError::ProofVerifyError);
    }

    // check the (relaxed) R1CS instances public outputs.
    if self.l_u_secondary.X.len() != 2 {
      return Err(NovaError::ProofVerifyError);
    }

    self.r_U_primary.iter().try_for_each(|U| match U.clone() {
      Some(U) if U.X.len() != 2 => {
        println!("r_U_primary got instance length {:?} != {:?}", U.X.len(), 2);
        Err(NovaError::ProofVerifyError)
      }
      _ => Ok(()),
    })?;

    self
      .r_U_secondary
      .iter()
      .try_for_each(|U| match U.clone() {
        Some(U) if U.X.len() != 2 => {
          println!(
            "r_U_secondary got instance length {:?} != {:?}",
            U.X.len(),
            2
          );
          Err(NovaError::ProofVerifyError)
        }
        _ => Ok(()),
      })?;

    // check if the output hashes in R1CS instances point to the right running instances
    let (hash_primary, hash_secondary) = {
      let mut hasher = <G2 as Group>::RO::new(
        pp.ro_consts_secondary.clone(),
        NUM_FE_WITHOUT_IO_FOR_CRHF + 18 * pp.F_arity_primary,
      );
      hasher.absorb(pp.digest);
      hasher.absorb(G1::Scalar::from(self.i as u64));
      hasher.absorb(G1::Scalar::from(self.program_counter as u64));
      for e in &z0_primary {
        hasher.absorb(*e);
      }
      for e in &self.zi_primary {
        hasher.absorb(*e);
      }
      self.r_U_secondary.iter().for_each(|U| {
        U.clone()
          .unwrap_or_else(|| RelaxedR1CSInstance::default(ck_secondary, &pp.r1cs_shape_secondary))
          .absorb_in_ro(&mut hasher);
      });

      let mut hasher2 = <G1 as Group>::RO::new(
        pp.ro_consts_primary.clone(),
        NUM_FE_WITHOUT_IO_FOR_CRHF + 18 * pp.F_arity_secondary,
      );
      hasher2.absorb(scalar_as_base::<G1>(pp.digest));
      hasher2.absorb(G2::Scalar::from(self.i as u64));
      hasher2.absorb(G2::Scalar::from(self.program_counter as u64));
      for e in &z0_secondary {
        hasher2.absorb(*e);
      }
      for e in &self.zi_secondary {
        hasher2.absorb(*e);
      }
      self.r_U_primary.iter().for_each(|U| {
        U.clone()
          .unwrap_or_else(|| RelaxedR1CSInstance::default(ck_primary, &pp.r1cs_shape_primary))
          .absorb_in_ro(&mut hasher2)
      });

      (
        hasher.squeeze(NUM_HASH_BITS),
        hasher2.squeeze(NUM_HASH_BITS),
      )
    };

    if hash_primary != self.l_u_secondary.X[0] {
      println!(
        "hash_primary {:?} not equal l_u_secondary.X[0] {:?}",
        hash_primary, self.l_u_secondary.X[0]
      );
      return Err(NovaError::ProofVerifyError);
    }
    if hash_secondary != scalar_as_base::<G2>(self.l_u_secondary.X[1]) {
      println!(
        "hash_secondary {:?} not equal l_u_secondary.X[1] {:?}",
        hash_secondary, self.l_u_secondary.X[1]
      );
      return Err(NovaError::ProofVerifyError);
    }

    // check the satisfiability of the provided `circuit_index` instance
    let (res_r_primary, (res_r_secondary, res_l_secondary)) = rayon::join(
      || {
        pp.r1cs_shape_primary.is_sat_relaxed(
          pp.ck_primary.as_ref().unwrap(),
          &self.r_U_primary[circuit_index]
            .clone()
            .unwrap_or_else(|| RelaxedR1CSInstance::default(ck_primary, &pp.r1cs_shape_primary)),
          &self.r_W_primary[circuit_index]
            .clone()
            .unwrap_or_else(|| RelaxedR1CSWitness::default(&pp.r1cs_shape_primary)),
        )
      },
      || {
        rayon::join(
          || {
            pp.r1cs_shape_secondary.is_sat_relaxed(
              pp.ck_secondary.as_ref().unwrap(),
              &self.r_U_secondary[circuit_index]
                .clone()
                .unwrap_or_else(|| {
                  RelaxedR1CSInstance::default(ck_secondary, &pp.r1cs_shape_secondary)
                }),
              &self.r_W_secondary[circuit_index]
                .clone()
                .unwrap_or_else(|| RelaxedR1CSWitness::default(&pp.r1cs_shape_secondary)),
            )
          },
          || {
            pp.r1cs_shape_secondary.is_sat(
              &pp.ck_secondary.as_ref().unwrap(),
              &self.l_u_secondary,
              &self.l_w_secondary,
            )
          },
        )
      },
    );

    res_r_primary?;
    res_r_secondary?;
    res_l_secondary?;

    Ok((
      self.zi_primary.clone(),
      self.zi_secondary.clone(),
      self.program_counter,
    ))
  }

  fn execute_and_verify_circuits<C1, C2>(
    circuit_index: usize,
    u_i_length: usize, // total number of F' circuit
    mut running_claim: RunningClaim<G1, G2, C1, C2>,
    ck_primary: &CommitmentKey<G1>,
    ck_secondary: &CommitmentKey<G2>,
    last_running_instance: Option<NivcSNARK<G1, G2>>,
  ) -> Result<
    (
      NivcSNARK<G1, G2>,
      Result<(Vec<G1::Scalar>, Vec<G2::Scalar>, usize), NovaError>,
    ),
    Box<dyn std::error::Error>,
  >
  where
    G1: Group<Base = <G2 as Group>::Scalar>,
    G2: Group<Base = <G1 as Group>::Scalar>,
    C1: StepCircuit<<G1 as Group>::Scalar> + Clone + Default + std::fmt::Debug,
    C2: StepCircuit<<G2 as Group>::Scalar> + Clone + Default + std::fmt::Debug,
  {
    // Produce a recursive SNARK
    let mut recursive_snark: Option<NivcSNARK<G1, G2>> = last_running_instance.clone();
    let program_counter = last_running_instance
      .as_ref()
      .map(|last_nivcsnark| last_nivcsnark.program_counter)
      .unwrap_or_default();
    let mut final_result: Result<(Vec<G1::Scalar>, Vec<G2::Scalar>, usize), NovaError> =
      Err(NovaError::InvalidInitialInputLength);

    let res = NivcSNARK::prove_step(
      circuit_index,
      u_i_length,
      program_counter,
      &running_claim.params,
      last_running_instance,
      running_claim.claim.clone(),
      running_claim.circuit_secondary.clone(),
      vec![<G1 as Group>::Scalar::ONE],
      vec![<G2 as Group>::Scalar::ZERO],
      ck_primary,
      ck_secondary,
    );

    let mut recursive_snark_unwrapped = res.unwrap();

    // assert NivcSNARK program_counter bump to next rount.
    assert_eq!(
      program_counter + 1,
      recursive_snark_unwrapped.program_counter
    );

    // Verify the recursive Nova snark at each step of recursion
    let res = recursive_snark_unwrapped.verify(
      &running_claim.params,
      circuit_index,
      u_i_length,
      vec![<G1 as Group>::Scalar::ONE],
      vec![<G2 as Group>::Scalar::ZERO],
      ck_primary,
      ck_secondary,
    );
    if let Err(x) = res.clone() {
      println!("res failed {:?}", x);
    }
    assert!(res.is_ok());
    let (zi_primary, zi_secondary, program_counter) = res.unwrap();
    final_result = Ok((zi_primary, zi_secondary, program_counter));
    // Set the running variable for the next iteration
    recursive_snark = Some(recursive_snark_unwrapped);

    recursive_snark
      .ok_or_else(|| {
        Box::new(std::io::Error::new(
          std::io::ErrorKind::Other,
          "an error occured in recursive_snark",
        )) as Box<dyn std::error::Error>
      })
      .map(|snark| (snark, final_result))
  }
}

#[cfg(test)]
mod tests {
  use std::cmp::Ordering;

  use crate::r1cs::R1CS;

  use super::*;

  use ::bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};

  #[derive(Clone, Debug, Default)]
  struct CubicCircuit<F: PrimeField> {
    _p: PhantomData<F>,
  }

  impl<F> StepCircuit<F> for CubicCircuit<F>
  where
    F: PrimeField,
  {
    fn arity(&self) -> usize {
      1
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
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

      Ok(vec![y])
    }

    fn output(&self, z: &[F]) -> Vec<F> {
      vec![z[0] * z[0] * z[0] + z[0] + F::from(5u64)]
    }
  }

  #[derive(Clone, Debug, Default)]
  struct SquareCircuit<F: PrimeField> {
    _p: PhantomData<F>,
  }

  impl<F> StepCircuit<F> for SquareCircuit<F>
  where
    F: PrimeField,
  {
    fn arity(&self) -> usize {
      1
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
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

      Ok(vec![y])
    }

    fn output(&self, z: &[F]) -> Vec<F> {
      vec![z[0] * z[0] + z[0] + F::from(5u64)]
    }
  }

  fn circuit_index_by_program_counter(program_counter: usize) -> usize {
    program_counter % 2
  }

  fn test_trivial_nivc_with<G1, G2>()
  where
    G1: Group<Base = <G2 as Group>::Scalar>,
    G2: Group<Base = <G1 as Group>::Scalar>,
  {
    /*
      'Let the corresponding NIVC proof be Πi , which consists of a vector of ` instances Ui..' pg.15
      The user sets a list 'Ui' of all of the functions that they plan on using. This is used as public input
      for the verifier.
    */
    let rom = [0, 1, 1, 0, 1, 0, 0];
    let circuit_secondary = TrivialTestCircuit::default();

    // Structuring running claims
    let test_circuit1 = CubicCircuit::default();
    let mut running_claim1 = RunningClaim::<
      G1,
      G2,
      CubicCircuit<<G1 as Group>::Scalar>,
      TrivialTestCircuit<<G2 as Group>::Scalar>,
    >::new(test_circuit1, circuit_secondary.clone(), true, U_i.len());

    let test_circuit2 = SquareCircuit::default();
    let mut running_claim2 = RunningClaim::<
      G1,
      G2,
      SquareCircuit<<G1 as Group>::Scalar>,
      TrivialTestCircuit<<G2 as Group>::Scalar>,
    >::new(test_circuit2, circuit_secondary.clone(), false, U_i.len());

    // generate the commitkey from largest F' size circuit and reused it for all circuits
    let circuit_public_params = vec![&running_claim1.params, &running_claim2.params];
    let (max_index_circuit, _) = circuit_public_params
      .iter()
      .enumerate()
      .map(|(i, params)| -> (usize, usize) {
        (
          i,
          params.r1cs_shape_primary.num_io + params.r1cs_shape_primary.num_vars,
        )
      })
      .max_by(|(_, circuit_size1), (_, circuit_size2)| circuit_size1.cmp(&circuit_size2))
      .unwrap();

    let ck_primary =
      R1CS::<G1>::commitment_key(&circuit_public_params[max_index_circuit].r1cs_shape_primary);
    let ck_secondary =
      R1CS::<G2>::commitment_key(&circuit_public_params[max_index_circuit].r1cs_shape_secondary);

    // set unified ck_primary, ck_secondary and update digest
    running_claim1.params.ck_primary = Some(ck_primary.clone());
    running_claim1.params.ck_secondary = Some(ck_secondary.clone());
    running_claim1.params.digest =
      compute_digest::<G1, PublicParams<G1, G2>>(&running_claim1.params);

    running_claim2.params.ck_primary = Some(ck_primary);
    running_claim2.params.ck_secondary = Some(ck_secondary);
    running_claim2.params.digest =
      compute_digest::<G1, PublicParams<G1, G2>>(&running_claim2.params);

    /*
      Needs:
      - We do not know how many times a certain circuit will be run.
      - The user should be able to run any circuits in any order and re-use RunningClaim.
      - Only the commitment key of the largest circuit is needed.

      Representing program_counter and U_i to make sure sequencing is done correctly:

      "1. Checks that Ui and PCi are contained in the public output of the instance ui.
      This enforces that Ui and PCi are indeed produced by the prior step."

      In this implementation we use a vector U_i, which is pre-deteremind by the user.
      i.e. [0, 1, 2, 3] for a 4 running instance proof; 0 is the circuit_index for the first F'.

      We check that both U_i and program_counter are contained in the public output of instance ui.
      For the SuperNova proof we check each F'[i for F'i]
      is a satisfying Nova instance.

      If all U_i are satisfying and U_i, pci, and a pre-image of a hash(U_i-1, pci-1, zi-1) are
      in the public output we have a valid SuperNova proof.

      program_counter is enforced in the augmented circuit as program_counter + 1 increment.
      ϕ does not exist in this implementation as F' are chosen by the user.
      program_counter and U_i are checked in the augmented verifier.
      https://youtu.be/ilrvqajkrYY?t=2559

      "So, the verifier can check the NIVC statement (i, z0, zi) by checking the following:
      (ui,wi) is a satisfying instance witness pair with respect to function F0_pci,
      the public IO of ui contains Ui and pci, and for each j ∈ {1, . . . , `} check that (Ui [j], Wi
      [j]) is a satisfying instance-witness pair with respect to function F0j." pg15.

      ---------

      "2. Runs the non-interactive folding scheme’s verifier to fold an instance that
        claims the correct execution of the previous step, ui, into Ui [pci] to produce
        an updated list of running instances Ui+1. This ensures that checking Ui+1
        implies checking Ui and ui while maintaining that Ui+1 does not grow in size
        with respect to Ui."

      This is mostly done with the existing Nova code. With additions of U_i and pci checks
      in the augmented circuit. The latest U_i[pci] at that index needs to be a satisfying instance.

      Would mixing up the order of U_i as input break this? i.e. [1, 0, 2, 3]
      Correct sequencing is enfored by checking the preimage hash(pci - 1, U_i - 1, zi -1).

      --------

      "3. Invokes the function ϕ on input (zi, ωi) to compute pci+1, which represents
      the index of the function Fj currently being run. pci+1 is then sent to the
      next invocation of an augmented circuit (which contains a verifier circuit)."

      This is done in the augmented circuit and is just pci+1.
      The circuit_index, and U_i are decided by the user.

    */

    let num_steps = 10;
    let mut program_counter: usize = 0;
    let mut recursive_snark: Option<NivcSNARK<G1, G2>> = None;
    // we use first F'circuit commitkey as major commitkey
    let (ck_primary, ck_secondary) = (
      running_claim1.clone().params.ck_primary.unwrap(),
      running_claim1.clone().params.ck_secondary.unwrap(),
    );
    let mut final_result: Result<(Vec<G1::Scalar>, Vec<G2::Scalar>, usize), NovaError> =
      Err(NovaError::InvalidInitialInputLength);

    for _ in 0..num_steps {
      if let Some(nivc_snark) = recursive_snark.clone() {
        program_counter = nivc_snark.program_counter;
      }
      let selector = circuit_index_by_program_counter(program_counter);
      let (nivc_snark, result) = match selector {
        0 => NivcSNARK::execute_and_verify_circuits(
          selector,
          2, // FIXME: remove hardcode number of circuit
          running_claim1.clone(),
          &ck_primary,
          &ck_secondary,
          recursive_snark, // last running instance.
        )
        .unwrap(),
        1 => NivcSNARK::execute_and_verify_circuits(
          selector,
          2, // FIXME: remove hardcode number of circuit
          running_claim2.clone(),
          &ck_primary,
          &ck_secondary,
          recursive_snark, // last running instance.
        )
        .unwrap(),
        _ => unimplemented!(),
      };
      recursive_snark = Some(nivc_snark);
      final_result = result;
    }

    println!("final Nivc SNARK {:?}", recursive_snark);
    assert!(recursive_snark.is_some());

    // Now you can handle the Result using if let
    if let Ok((zi_primary, zi_secondary, program_counter)) = final_result {
      println!("zi_primary: {:?}", zi_primary);
      println!("zi_secondary: {:?}", zi_secondary);
      println!("final program_counter: {:?}", program_counter);
    }
  }

  #[test]
  fn test_trivial_nivc() {
    type G1 = pasta_curves::pallas::Point;
    type G2 = pasta_curves::vesta::Point;

    //Expirementing with selecting the running claims for nifs
    test_trivial_nivc_with::<G1, G2>();
  }
}
