//! There are two Verification Circuits. Each of them is over a Pasta curve but
//! only one of them executes the next step of the computation by applying the inner function F.
//! There are also two running relaxed r1cs instances.
//!
//! When we build a circuit we denote u1 the running relaxed r1cs instance of
//! the circuit and u2 the running relaxed r1cs instance of the other circuit.
//! The circuit takes as input two hashes h1 and h2.
//! If the circuit applies the inner function F, then
//! h1 = H(u2, i, z0, zi) and h2 = H(u1, i)
//! otherwise
//! h1 = H(u2, i) and h2 = H(u1, i, z0, zi)

use super::commitments::Commitment;
use super::gadgets::{
  ecc_circuit::AllocatedPoint,
  utils::{alloc_num_equals, alloc_one, alloc_zero, conditionally_select, le_bits_to_num},
};
use super::poseidon::NovaPoseidonConstants;
use super::poseidon::PoseidonROGadget;
use super::r1cs::RelaxedR1CSInstance;
use super::traits::{Group, InnerCircuit, PrimeField};
use bellperson::{
  gadgets::{boolean::Boolean, num::AllocatedNum, Assignment},
  Circuit, ConstraintSystem, SynthesisError,
};
use ff::PrimeFieldBits;

pub struct VerificationCircuitInputs<G>
where
  G: Group,
{
  h1: G::Base,
  h2: G::Scalar,
  u2: RelaxedR1CSInstance<G>,
  i: G::Base,
  z0: G::Base,
  zi: G::Base,
  params: G::Base, //Hash(Shape of u2, Gens for u2). Needed for computing the challenge. TODO: Hard code them or make them part of the input hash
  T: Commitment<G>,
  w: Commitment<G>, //The commitment to the witness of the fresh r1cs instance
}

impl<G> VerificationCircuitInputs<G>
where
  G: Group,
{
  ///Create new inputs/witness for the verification circuit
  #[allow(dead_code)]
  pub fn new(
    h1: G::Base,
    u2: RelaxedR1CSInstance<G>,
    i: G::Base,
    z0: G::Base,
    zi: G::Base,
    h2: G::Scalar,
    params: G::Base,
    T: Commitment<G>,
    w: Commitment<G>,
  ) -> Self {
    Self {
      h1,
      u2,
      i,
      z0,
      zi,
      h2,
      params,
      T,
      w,
    }
  }
}

///Circuit that encodes only the folding verifier
pub struct VerificationCircuit<G, IC>
where
  G: Group,
  <G as Group>::Base: ff::PrimeField,
  IC: InnerCircuit<G::Base>,
{
  inputs: Option<VerificationCircuitInputs<G>>,
  inner_circuit: Option<IC>, //The function that is applied for each step. may be None.
  poseidon_constants: NovaPoseidonConstants<G::Base>,
}

impl<G, IC> VerificationCircuit<G, IC>
where
  G: Group,
  <G as Group>::Base: ff::PrimeField,
  IC: InnerCircuit<G::Base>,
{
  ///Create a new verification circuit for the input relaxed r1cs instances
  #[allow(dead_code)]
  pub fn new(
    inputs: Option<VerificationCircuitInputs<G>>,
    inner_circuit: Option<IC>,
    poseidon_constants: NovaPoseidonConstants<G::Base>,
  ) -> Self
  where
    <G as Group>::Base: ff::PrimeField,
  {
    Self {
      inputs,
      inner_circuit,
      poseidon_constants,
    }
  }
}

impl<G, IC> Circuit<<G as Group>::Base> for VerificationCircuit<G, IC>
where
  G: Group,
  <G as Group>::Base: ff::PrimeField + PrimeField + PrimeFieldBits,
  <G as Group>::Scalar: PrimeFieldBits,
  IC: InnerCircuit<G::Base>,
{
  fn synthesize<CS: ConstraintSystem<<G as Group>::Base>>(
    self,
    cs: &mut CS,
  ) -> Result<(), SynthesisError> {
    /***********************************************************************/
    //This circuit does not modify h2 but it outputs it.
    //Allocate it and output it.
    /***********************************************************************/

    //TODO: Make this work when |G::Scalar| > |G::Base|

    let h2 = AllocatedNum::alloc(cs.namespace(|| "h2"), || {
      let h2_bits = self.inputs.get()?.h2.to_le_bits();
      let mut mult = G::Base::one();
      let mut h2 = G::Base::zero();
      for bit in h2_bits {
        if bit {
          h2 = h2 + mult;
        }
        mult = mult + mult;
      }
      Ok(h2)
    })?;

    let _ = h2.inputize(cs.namespace(|| "Output 1"))?;

    /***********************************************************************/
    //Allocate h1
    /***********************************************************************/

    let h1 = AllocatedNum::alloc(cs.namespace(|| "h1"), || Ok(self.inputs.get()?.h1))?;

    /***********************************************************************/
    //Allocate u2 by allocating W_r, E_r, u_r, io_r
    /***********************************************************************/

    //W_r = (x, y, infinity)
    let W_r_x = AllocatedNum::alloc(cs.namespace(|| "W_r.x"), || {
      Ok(self.inputs.get()?.u2.comm_W.comm.to_coordinates().0)
    })?;
    let W_r_y = AllocatedNum::alloc(cs.namespace(|| "W_r.y"), || {
      Ok(self.inputs.get()?.u2.comm_W.comm.to_coordinates().1)
    })?;
    let W_r_inf = AllocatedNum::alloc(cs.namespace(|| "W_r.inf"), || {
      let infty = self.inputs.get()?.u2.comm_W.comm.to_coordinates().2;
      if infty {
        Ok(G::Base::one())
      } else {
        Ok(G::Base::zero())
      }
    })?;

    let W_r = AllocatedPoint::new(W_r_x.clone(), W_r_y.clone(), W_r_inf.clone());
    let _ = W_r.check_is_infinity(cs.namespace(|| "W_r check is_infinity"))?;

    //E_r = (x, y, infinity)
    let E_r_x = AllocatedNum::alloc(cs.namespace(|| "E_r.x"), || {
      Ok(self.inputs.get()?.u2.comm_E.comm.to_coordinates().0)
    })?;
    let E_r_y = AllocatedNum::alloc(cs.namespace(|| "E_r.y"), || {
      Ok(self.inputs.get()?.u2.comm_E.comm.to_coordinates().1)
    })?;
    let E_r_inf = AllocatedNum::alloc(cs.namespace(|| "E_r.inf"), || {
      let infty = self.inputs.get()?.u2.comm_E.comm.to_coordinates().2;
      if infty {
        Ok(G::Base::one())
      } else {
        Ok(G::Base::zero())
      }
    })?;

    let E_r = AllocatedPoint::new(E_r_x.clone(), E_r_y.clone(), E_r_inf.clone());
    let _ = E_r.check_is_infinity(cs.namespace(|| "E_r check is_infinity"));

    //u_r << |G::Base| despite the fact that u_r is a scalar.
    //So we parse all of its bytes as a G::Base element
    let u_r = AllocatedNum::alloc(cs.namespace(|| "u_r"), || {
      let u_bits = self.inputs.get()?.u2.u.to_le_bits();
      let mut mult = G::Base::one();
      let mut u = G::Base::zero();
      for bit in u_bits {
        if bit {
          u = u + mult;
        }
        mult = mult + mult;
      }
      Ok(u)
    })?;

    /***********************************************************************/
    //Allocate i
    /***********************************************************************/

    let i = AllocatedNum::alloc(cs.namespace(|| "i"), || Ok(self.inputs.get()?.i))?;

    /***********************************************************************/
    //Allocate T
    /***********************************************************************/

    //T = (x, y, infinity)
    let T_x = AllocatedNum::alloc(cs.namespace(|| "T.x"), || {
      Ok(self.inputs.get()?.T.comm.to_coordinates().0)
    })?;
    let T_y = AllocatedNum::alloc(cs.namespace(|| "T.y"), || {
      Ok(self.inputs.get()?.T.comm.to_coordinates().1)
    })?;
    let T_inf = AllocatedNum::alloc(cs.namespace(|| "T.inf"), || {
      let infty = self.inputs.get()?.T.comm.to_coordinates().2;
      if infty {
        Ok(G::Base::one())
      } else {
        Ok(G::Base::zero())
      }
    })?;

    let T = AllocatedPoint::new(T_x.clone(), T_y.clone(), T_inf.clone());
    let _ = T.check_is_infinity(cs.namespace(|| "T check is_infinity"));

    /***********************************************************************/
    //Allocate params
    /***********************************************************************/

    let params = AllocatedNum::alloc(cs.namespace(|| "params"), || Ok(self.inputs.get()?.params))?;

    /***********************************************************************/
    //Allocate W
    /***********************************************************************/

    //T = (x, y, infinity)
    let W_x = AllocatedNum::alloc(cs.namespace(|| "W.x"), || {
      Ok(self.inputs.get()?.w.comm.to_coordinates().0)
    })?;
    let W_y = AllocatedNum::alloc(cs.namespace(|| "W.y"), || {
      Ok(self.inputs.get()?.w.comm.to_coordinates().1)
    })?;
    let W_inf = AllocatedNum::alloc(cs.namespace(|| "w.inf"), || {
      let infty = self.inputs.get()?.w.comm.to_coordinates().2;
      if infty {
        Ok(G::Base::one())
      } else {
        Ok(G::Base::zero())
      }
    })?;

    let W = AllocatedPoint::new(W_x.clone(), W_y.clone(), W_inf.clone());
    let _ = W.check_is_infinity(cs.namespace(|| "W check is_infinity"));

    /***********************************************************************/
    //U2' = default if i == 0, otherwise NIFS.V(pp, u_new, U, T)
    /***********************************************************************/

    //Allocate 0 and 1
    let zero = alloc_zero(cs.namespace(|| "zero"))?;
    let one = alloc_one(cs.namespace(|| "one"))?;

    //Compute default values of U2':
    let zero_commitment = AllocatedPoint::new(zero.clone(), zero.clone(), one.clone());

    //W_default and E_default are a commitment to zero
    let W_default = zero_commitment.clone();
    let E_default = zero_commitment.clone();

    //u_default = 0

    let u_default = zero.clone();

    //TODO: Add X_default

    //Compute r:

    let mut ro: PoseidonROGadget<G::Base> = PoseidonROGadget::new(self.poseidon_constants.clone());

    ro.absorb(params);
    ro.absorb(h1.clone());
    ro.absorb(h2);
    ro.absorb(W_x);
    ro.absorb(W_y);
    ro.absorb(W_inf);
    ro.absorb(T_x);
    ro.absorb(T_y);
    ro.absorb(T_inf);

    let r_bits = ro.get_challenge(cs.namespace(|| "r bits"))?;
    let r = le_bits_to_num(cs.namespace(|| "r"), r_bits.clone())?;

    //W_fold = W_r + r * W
    let rW = W.scalar_mul(cs.namespace(|| "r * W"), r_bits.clone())?;
    let W_fold = W_r.add(cs.namespace(|| "W_r + r * W"), &rW)?;

    //E_fold = E_r + r * T
    let rT = T.scalar_mul(cs.namespace(|| "r * T"), r_bits)?;
    let E_fold = E_r.add(cs.namespace(|| "E_r + r * T"), &rT)?;

    //u_fold = u_r + r
    let u_fold = AllocatedNum::alloc(cs.namespace(|| "u_fold"), || {
      Ok(*u_r.get_value().get()? + r.get_value().get()?)
    })?;
    cs.enforce(
      || "Check u_fold",
      |lc| lc,
      |lc| lc,
      |lc| lc + u_fold.get_variable() - u_r.get_variable() - r.get_variable(),
    );

    //TODO: Add folding of io

    //Now select the default values if i == 0 otherwise the fold values
    let base_case = Boolean::from(alloc_num_equals(
      cs.namespace(|| "Check if base case"),
      i.clone(),
      zero.clone(),
    )?);

    let W_new = AllocatedPoint::conditionally_select(
      cs.namespace(|| "W_new"),
      &W_default,
      &W_fold,
      &base_case,
    )?;

    let E_new = AllocatedPoint::conditionally_select(
      cs.namespace(|| "E_new"),
      &E_default,
      &E_fold,
      &base_case,
    )?;

    let u_new = conditionally_select(cs.namespace(|| "u_new"), &u_default, &u_fold, &base_case)?;

    /***********************************************************************/
    //Compute i + 1
    /***********************************************************************/

    let next_i = AllocatedNum::alloc(cs.namespace(|| "i + 1"), || {
      Ok(*i.get_value().get()? + G::Base::one())
    })?;

    cs.enforce(
      || "check i + 1",
      |lc| lc,
      |lc| lc,
      |lc| lc + next_i.get_variable() - CS::one() - i.get_variable(),
    );

    if self.inner_circuit.is_some() {
      /***********************************************************************/
      //Allocate z0
      /***********************************************************************/

      let z_0 = AllocatedNum::alloc(cs.namespace(|| "z0"), || Ok(self.inputs.get()?.z0))?;

      /***********************************************************************/
      //Allocate zi
      /***********************************************************************/

      let z_i = AllocatedNum::alloc(cs.namespace(|| "zi"), || Ok(self.inputs.get()?.zi))?;

      /***********************************************************************/
      //Check that h1 = Hash(u2,i,z0,zi)
      /***********************************************************************/

      let mut h1_hash: PoseidonROGadget<G::Base> =
        PoseidonROGadget::new(self.poseidon_constants.clone());

      h1_hash.absorb(W_r_x.clone());
      h1_hash.absorb(W_r_y.clone());
      h1_hash.absorb(W_r_inf.clone());
      h1_hash.absorb(E_r_x.clone());
      h1_hash.absorb(E_r_y.clone());
      h1_hash.absorb(E_r_inf.clone());
      h1_hash.absorb(u_r.clone());
      //TODO: Add X_r

      h1_hash.absorb(i.clone());
      h1_hash.absorb(z_0.clone());
      h1_hash.absorb(z_i.clone());

      let hash_bits = h1_hash.get_challenge(cs.namespace(|| "Input hash"))?;
      let hash = le_bits_to_num(cs.namespace(|| "bits to hash"), hash_bits)?;

      cs.enforce(
        || "check h1",
        |lc| lc,
        |lc| lc,
        |lc| lc + h1.get_variable() - hash.get_variable(),
      );

      /***********************************************************************/
      //Compute z_{i+1}
      /***********************************************************************/

      let z_next = self
        .inner_circuit
        .unwrap()
        .synthesize(&mut cs.namespace(|| "F"), z_i)?;

      /***********************************************************************/
      //Compute the new hash H(u2, i+1, z0, z_{i+1})
      /***********************************************************************/

      h1_hash.flush_state();
      h1_hash.absorb(W_new.x.clone());
      h1_hash.absorb(W_new.y.clone());
      h1_hash.absorb(W_new.is_infinity.clone());
      h1_hash.absorb(E_new.x.clone());
      h1_hash.absorb(E_new.y.clone());
      h1_hash.absorb(E_new.is_infinity.clone());
      h1_hash.absorb(u_new.clone());
      //TODO: Add X_r
      h1_hash.absorb(next_i.clone());
      h1_hash.absorb(z_0.clone());
      h1_hash.absorb(z_next.clone());
      let h1_new_bits = h1_hash.get_challenge(cs.namespace(|| "h1_new bits"))?;
      let h1_new = le_bits_to_num(cs.namespace(|| "h1_new"), h1_new_bits.clone())?;
      let _ = h1_new.inputize(cs.namespace(|| "output h1_new"))?;
    } else {
      /***********************************************************************/
      //Check that h1 = Hash(u2, i)
      /***********************************************************************/

      let mut h1_hash: PoseidonROGadget<G::Base> =
        PoseidonROGadget::new(self.poseidon_constants.clone());

      h1_hash.absorb(W_r_x.clone());
      h1_hash.absorb(W_r_y.clone());
      h1_hash.absorb(W_r_inf.clone());
      h1_hash.absorb(E_r_x.clone());
      h1_hash.absorb(E_r_y.clone());
      h1_hash.absorb(E_r_inf.clone());
      h1_hash.absorb(u_r.clone());
      h1_hash.absorb(i.clone());
      //TODO: Add X_r

      let hash_bits = h1_hash.get_challenge(cs.namespace(|| "Input hash"))?;
      let hash = le_bits_to_num(cs.namespace(|| "bits to hash"), hash_bits)?;

      cs.enforce(
        || "check h1",
        |lc| lc,
        |lc| lc,
        |lc| lc + h1.get_variable() - hash.get_variable(),
      );

      /***********************************************************************/
      //Compute the new hash H(u2')
      /***********************************************************************/

      h1_hash.flush_state();
      h1_hash.absorb(W_new.x.clone());
      h1_hash.absorb(W_new.y.clone());
      h1_hash.absorb(W_new.is_infinity.clone());
      h1_hash.absorb(E_new.x.clone());
      h1_hash.absorb(E_new.y.clone());
      h1_hash.absorb(E_new.is_infinity.clone());
      h1_hash.absorb(u_new.clone());
      h1_hash.absorb(next_i.clone());
      //TODO: Add X_r
      let h1_new_bits = h1_hash.get_challenge(cs.namespace(|| "h1_new bits"))?;
      let h1_new = le_bits_to_num(cs.namespace(|| "h1_new"), h1_new_bits.clone())?;
      let _ = h1_new.inputize(cs.namespace(|| "output h1_new"))?;
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::bellperson::shape_cs::ShapeCS;
  use crate::bellperson::solver::SatisfyingAssignment;
  type G1 = pasta_curves::pallas::Point;
  type G2 = pasta_curves::vesta::Point;
  use crate::bellperson::r1cs::{NovaShape, NovaWitness};
  use crate::commitments::CommitTrait;
  use std::marker::PhantomData;

  struct TestCircuit<F>
  where
    F: PrimeField + ff::PrimeField,
  {
    _p: PhantomData<F>,
  }

  impl<F> InnerCircuit<F> for TestCircuit<F>
  where
    F: PrimeField + ff::PrimeField,
  {
    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: AllocatedNum<F>,
    ) -> Result<AllocatedNum<F>, SynthesisError> {
      Ok(z.clone())
    }
  }

  #[test]
  fn test_verification_circuit() {
    //The first circuit that verifies G2
    let poseidon_constants1: NovaPoseidonConstants<<G2 as Group>::Base> =
      NovaPoseidonConstants::new();
    let circuit1: VerificationCircuit<G2, TestCircuit<<G2 as Group>::Base>> =
      VerificationCircuit::new(
        None,
        Some(TestCircuit {
          _p: Default::default(),
        }),
        poseidon_constants1.clone(),
      );
    //First create the shape
    let mut cs: ShapeCS<G1> = ShapeCS::new();
    let _ = circuit1.synthesize(&mut cs);
    let shape1 = cs.r1cs_shape();
    let gens1 = cs.r1cs_gens();
    println!(
      "Circuit1 -> Number of constraints: {}",
      cs.num_constraints()
    );

    //The second circuit that verifies G1
    let poseidon_constants2: NovaPoseidonConstants<<G1 as Group>::Base> =
      NovaPoseidonConstants::new();
    let circuit2: VerificationCircuit<G1, TestCircuit<<G1 as Group>::Base>> =
      VerificationCircuit::new(None, None, poseidon_constants2.clone());
    //First create the shape
    let mut cs: ShapeCS<G2> = ShapeCS::new();
    let _ = circuit2.synthesize(&mut cs);
    let shape2 = cs.r1cs_shape();
    let gens2 = cs.r1cs_gens();
    println!(
      "Circuit2 -> Number of constraints: {}",
      cs.num_constraints()
    );

    //TODO: We need to hardwire default hash or give it as input
    let default_hash = <<G2 as Group>::Base as ff::PrimeField>::from_str_vartime(
      "178164938390736661201066150778981665957",
    )
    .unwrap();
    let T = vec![<G2 as Group>::Scalar::zero()].commit(&gens2.gens);
    let w = vec![<G2 as Group>::Scalar::zero()].commit(&gens2.gens);
    //Now get an assignment
    let mut cs: SatisfyingAssignment<G1> = SatisfyingAssignment::new();
    let inputs: VerificationCircuitInputs<G2> = VerificationCircuitInputs::new(
      default_hash,
      RelaxedR1CSInstance::default(&gens2, &shape2),
      <<G2 as Group>::Base as PrimeField>::zero(), //TODO: Fix This
      <<G2 as Group>::Base as PrimeField>::zero(), //TODO: Fix This
      <<G2 as Group>::Base as PrimeField>::zero(), //TODO: Fix This
      <<G2 as Group>::Scalar as PrimeField>::zero(), //TODO: Fix This
      <<G2 as Group>::Base as PrimeField>::zero(), //TODO: Fix This
      T,                                           //TODO: Fix This
      w,
    );
    let circuit: VerificationCircuit<G2, TestCircuit<<G2 as Group>::Base>> =
      VerificationCircuit::new(
        Some(inputs),
        Some(TestCircuit {
          _p: Default::default(),
        }),
        poseidon_constants1.clone(),
      );
    let _ = circuit.synthesize(&mut cs);
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape1, &gens1).unwrap();

    //Make sure that this is satisfiable
    assert!(shape1.is_sat(&gens1, &inst, &witness).is_ok());
  }
}
