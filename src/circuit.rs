//! There are two Verification Circuits. Each of them is over a Pasta curve but
//! only one of them executes the next step of the computation by applying the inner function F.
//! There are also two running relaxed r1cs instances.
//!
//! When we build a circuit we denote u1 the running relaxed r1cs instance of
//! the circuit and u2 the running relaxed r1cs instance of the other circuit.
//! The circuit takes as input two hashes h1 and h2.
//! If the circuit applies the inner function F, then
//! h1 = H(params = H(shape, gens), u2, i, z0, zi) and h2 = H(u1, i)
//! otherwise
//! h1 = H(u2, i) and h2 = H(params = H(shape, gens), u1, i, z0, zi)

use super::commitments::Commitment;
use super::gadgets::{
  ecc::AllocatedPoint,
  utils::{
    alloc_bignat_constant, alloc_num_equals, alloc_one, alloc_zero, conditionally_select,
    conditionally_select_bignat, le_bits_to_num,
  },
};
use super::poseidon::NovaPoseidonConstants;
use super::poseidon::PoseidonROGadget;
use super::r1cs::RelaxedR1CSInstance;
use super::traits::{Group, InnerCircuit, PrimeField};
use bellperson::{
  gadgets::{boolean::Boolean, num::AllocatedNum, Assignment},
  Circuit, ConstraintSystem, SynthesisError,
};
use bellperson_nonnative::{
  mp::bignat::BigNat,
  util::{convert::f_to_nat, num::Num},
};
use ff::PrimeFieldBits;

#[derive(Debug, Clone)]
pub struct NIFSVerifierCircuitParams {
  limb_width: usize,
  n_limbs: usize,
}

impl NIFSVerifierCircuitParams {
  #[allow(dead_code)]
  pub fn new(limb_width: usize, n_limbs: usize) -> Self {
    Self {
      limb_width,
      n_limbs,
    }
  }
}

pub struct NIFSVerifierCircuitInputs<G>
where
  G: Group,
{
  h1: G::Base,
  h2: G::Scalar,
  u2: RelaxedR1CSInstance<G>,
  i: G::Base,
  z0: G::Base,
  zi: G::Base,
  params: G::Base, // Hash(Shape of u2, Gens for u2). Needed for computing the challenge.
  T: Commitment<G>,
  w: Commitment<G>, // The commitment to the witness of the fresh r1cs instance
}

impl<G> NIFSVerifierCircuitInputs<G>
where
  G: Group,
{
  /// Create new inputs/witness for the verification circuit
  #[allow(dead_code, clippy::too_many_arguments)]
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

/// Circuit that encodes only the folding verifier
pub struct NIFSVerifierCircuit<G, IC>
where
  G: Group,
  <G as Group>::Base: ff::PrimeField,
  IC: InnerCircuit<G::Base>,
{
  params: NIFSVerifierCircuitParams,
  inputs: Option<NIFSVerifierCircuitInputs<G>>,
  inner_circuit: Option<IC>, // The function that is applied for each step. may be None.
  poseidon_constants: NovaPoseidonConstants<G::Base>,
}

impl<G, IC> NIFSVerifierCircuit<G, IC>
where
  G: Group,
  <G as Group>::Base: ff::PrimeField,
  IC: InnerCircuit<G::Base>,
{
  /// Create a new verification circuit for the input relaxed r1cs instances
  #[allow(dead_code)]
  pub fn new(
    params: NIFSVerifierCircuitParams,
    inputs: Option<NIFSVerifierCircuitInputs<G>>,
    inner_circuit: Option<IC>,
    poseidon_constants: NovaPoseidonConstants<G::Base>,
  ) -> Self
  where
    <G as Group>::Base: ff::PrimeField,
  {
    Self {
      params,
      inputs,
      inner_circuit,
      poseidon_constants,
    }
  }
}

impl<G, IC> Circuit<<G as Group>::Base> for NIFSVerifierCircuit<G, IC>
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
    // This circuit does not modify h2 but it outputs it.
    // Allocate it and output it.
    /***********************************************************************/

    // Allocate h2 as a big number with 8 limbs
    let h2_bn = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate h2"),
      || Ok(f_to_nat(&self.inputs.get()?.h2)),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    let _ = h2_bn.inputize(cs.namespace(|| "Output 1"))?;

    /***********************************************************************/
    // Allocate h1
    /***********************************************************************/

    let h1 = AllocatedNum::alloc(cs.namespace(|| "allocate h1"), || Ok(self.inputs.get()?.h1))?;
    let h1_bn = BigNat::from_num(
      cs.namespace(|| "allocate h1_bn"),
      Num::from(h1.clone()),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    /***********************************************************************/
    // Allocate u2 by allocating W_r, E_r, u_r, X_r
    /***********************************************************************/

    // W_r = (x, y, infinity)
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

    // E_r = (x, y, infinity)
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

    // u_r << |G::Base| despite the fact that u_r is a scalar.
    // So we parse all of its bytes as a G::Base element
    let u_r = AllocatedNum::alloc(cs.namespace(|| "u_r"), || {
      let u_bits = self.inputs.get()?.u2.u.to_le_bits();
      let mut mult = G::Base::one();
      let mut u = G::Base::zero();
      for bit in u_bits {
        if bit {
          u += mult;
        }
        mult = mult + mult;
      }
      Ok(u)
    })?;

    // The running X is two items! the running h1 and the running h2
    let Xr0 = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate X_r[0]"),
      || Ok(f_to_nat(&self.inputs.get()?.u2.X[0])),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    // Analyze Xr0 as limbs to use later
    let Xr0_bn = Xr0
      .as_limbs::<CS>()
      .iter()
      .enumerate()
      .map(|(i, limb)| {
        limb
          .as_sapling_allocated_num(cs.namespace(|| format!("convert limb {} of X_r[0] to num", i)))
      })
      .collect::<Result<Vec<AllocatedNum<G::Base>>, _>>()?;

    let Xr1 = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate X_r[1]"),
      || Ok(f_to_nat(&self.inputs.get()?.u2.X[1])),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    // Analyze Xr1 as limbs to use later
    let Xr1_bn = Xr1
      .as_limbs::<CS>()
      .iter()
      .enumerate()
      .map(|(i, limb)| {
        limb
          .as_sapling_allocated_num(cs.namespace(|| format!("convert limb {} of X_r[1] to num", i)))
      })
      .collect::<Result<Vec<AllocatedNum<G::Base>>, _>>()?;

    /***********************************************************************/
    // Allocate i
    /***********************************************************************/

    let i = AllocatedNum::alloc(cs.namespace(|| "i"), || Ok(self.inputs.get()?.i))?;

    /***********************************************************************/
    // Allocate T
    /***********************************************************************/

    // T = (x, y, infinity)
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
    // Allocate params
    /***********************************************************************/

    let params = AllocatedNum::alloc(cs.namespace(|| "params"), || Ok(self.inputs.get()?.params))?;

    /***********************************************************************/
    // Allocate W
    /***********************************************************************/

    // T = (x, y, infinity)
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
    // U2' = default if i == 0, otherwise NIFS.V(pp, u_new, U, T)
    /***********************************************************************/

    //Allocate 0 and 1
    let zero = alloc_zero(cs.namespace(|| "zero"))?;
    // Hack: We just do this because the number of inputs must be even!!
    zero.inputize(cs.namespace(|| "allocate zero as input"))?;
    let one = alloc_one(cs.namespace(|| "one"))?;

    // Compute default values of U2':
    let zero_commitment = AllocatedPoint::new(zero.clone(), zero.clone(), one);

    //W_default and E_default are a commitment to zero
    let W_default = zero_commitment.clone();
    let E_default = zero_commitment;

    // u_default = 0
    let u_default = zero.clone();

    // X_default = 0
    let X0_default = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate x_default[0]"),
      || Ok(f_to_nat(&G::Scalar::zero())),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    let X1_default = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate x_default[1]"),
      || Ok(f_to_nat(&G::Scalar::zero())),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    // Compute r:

    let mut ro: PoseidonROGadget<G::Base> = PoseidonROGadget::new(self.poseidon_constants.clone());

    ro.absorb(h1.clone());
    // absorb each of the limbs of h2
    // TODO: Check if it is more efficient to treat h2 as allocNum
    for (i, limb) in h2_bn.as_limbs::<CS>().iter().enumerate() {
      let limb_num = limb
        .as_sapling_allocated_num(cs.namespace(|| format!("convert limb {} of h2 to num", i)))?;
      ro.absorb(limb_num);
    }
    ro.absorb(W_x);
    ro.absorb(W_y);
    ro.absorb(W_inf);
    ro.absorb(T_x);
    ro.absorb(T_y);
    ro.absorb(T_inf);
    // absorb each of the limbs of X_r[0]
    for limb in Xr0_bn.clone().into_iter() {
      ro.absorb(limb);
    }

    // absorb each of the limbs of X_r[1]
    for limb in Xr1_bn.clone().into_iter() {
      ro.absorb(limb);
    }

    let r_bits = ro.get_challenge(cs.namespace(|| "r bits"))?;
    let r = le_bits_to_num(cs.namespace(|| "r"), r_bits.clone())?;

    // W_fold = W_r + r * W
    let rW = W.scalar_mul(cs.namespace(|| "r * W"), r_bits.clone())?;
    let W_fold = W_r.add(cs.namespace(|| "W_r + r * W"), &rW)?;

    // E_fold = E_r + r * T
    let rT = T.scalar_mul(cs.namespace(|| "r * T"), r_bits)?;
    let E_fold = E_r.add(cs.namespace(|| "E_r + r * T"), &rT)?;

    // u_fold = u_r + r
    let u_fold = AllocatedNum::alloc(cs.namespace(|| "u_fold"), || {
      Ok(*u_r.get_value().get()? + r.get_value().get()?)
    })?;
    cs.enforce(
      || "Check u_fold",
      |lc| lc,
      |lc| lc,
      |lc| lc + u_fold.get_variable() - u_r.get_variable() - r.get_variable(),
    );

    // Fold the IO:
    // Analyze r into limbs
    let r_bn = BigNat::from_num(
      cs.namespace(|| "allocate r_bn"),
      Num::from(r.clone()),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    // Allocate the order of the non-native field as a constant
    let m_bn = alloc_bignat_constant(
      cs.namespace(|| "alloc m"),
      &G::Scalar::get_order(),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    // First the fold h1 with X_r[0];
    let (_, r_0) = h1_bn.mult_mod(cs.namespace(|| "r*h1"), &r_bn, &m_bn)?;
    // add X_r[0]
    let r_new_0 = Xr0.add::<CS>(&r_0)?;
    // Now reduce
    let Xr0_fold = r_new_0.red_mod(cs.namespace(|| "reduce folded X_r[0]"), &m_bn)?;

    // First the fold h2 with X_r[1];
    let (_, r_1) = h2_bn.mult_mod(cs.namespace(|| "r*h2"), &r_bn, &m_bn)?;
    // add X_r[1]
    let r_new_1 = Xr1.add::<CS>(&r_1)?;
    // Now reduce
    let Xr1_fold = r_new_1.red_mod(cs.namespace(|| "reduce folded X_r[1]"), &m_bn)?;

    // Now select the default values if i == 0 otherwise the fold values
    let base_case = Boolean::from(alloc_num_equals(
      cs.namespace(|| "Check if base case"),
      i.clone(),
      zero,
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

    let Xr0_new = conditionally_select_bignat(
      cs.namespace(|| "X_r_new[0]"),
      &X0_default,
      &Xr0_fold,
      &base_case,
    )?;

    // Analyze Xr0_new as limbs to use later
    let Xr0_new_bn = Xr0_new
      .as_limbs::<CS>()
      .iter()
      .enumerate()
      .map(|(i, limb)| {
        limb.as_sapling_allocated_num(
          cs.namespace(|| format!("convert limb {} of X_r_new[0] to num", i)),
        )
      })
      .collect::<Result<Vec<AllocatedNum<G::Base>>, _>>()?;

    let Xr1_new = conditionally_select_bignat(
      cs.namespace(|| "X_r_new[1]"),
      &X1_default,
      &Xr1_fold,
      &base_case,
    )?;

    // Analyze Xr1_new as limbs to use later
    let Xr1_new_bn = Xr1_new
      .as_limbs::<CS>()
      .iter()
      .enumerate()
      .map(|(i, limb)| {
        limb.as_sapling_allocated_num(
          cs.namespace(|| format!("convert limb {} of X_r_new[1] to num", i)),
        )
      })
      .collect::<Result<Vec<AllocatedNum<G::Base>>, _>>()?;

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
      //Check that if i == 0, z0 = zi, that is (i == 0) AND (z0 != zi) = false
      /***********************************************************************/

      let z0_is_zi = Boolean::from(alloc_num_equals(
        cs.namespace(|| "z0 = zi"),
        z_0.clone(),
        z_i.clone(),
      )?);

      cs.enforce(
        || "i == 0 and z0 != zi = false",
        |_| base_case.lc(CS::one(), G::Base::one()),
        |_| z0_is_zi.not().lc(CS::one(), G::Base::one()),
        |lc| lc,
      );

      /***********************************************************************/
      // Check that h1 = Hash(params, u2,i,z0,zi)
      /***********************************************************************/

      let mut h1_hash: PoseidonROGadget<G::Base> =
        PoseidonROGadget::new(self.poseidon_constants.clone());

      h1_hash.absorb(params.clone());
      h1_hash.absorb(W_r_x);
      h1_hash.absorb(W_r_y);
      h1_hash.absorb(W_r_inf);
      h1_hash.absorb(E_r_x);
      h1_hash.absorb(E_r_y);
      h1_hash.absorb(E_r_inf);
      h1_hash.absorb(u_r.clone());

      // absorb each of the limbs of X_r[0]
      for limb in Xr0_bn.into_iter() {
        h1_hash.absorb(limb);
      }

      // absorb each of the limbs of X_r[1]
      for limb in Xr1_bn.into_iter() {
        h1_hash.absorb(limb);
      }

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
      // Compute z_{i+1}
      /***********************************************************************/

      let z_next = self
        .inner_circuit
        .unwrap()
        .synthesize(&mut cs.namespace(|| "F"), z_i)?;

      /***********************************************************************/
      // Compute the new hash H(params, u2_new, i+1, z0, z_{i+1})
      /***********************************************************************/

      h1_hash.flush_state();
      h1_hash.absorb(params);
      h1_hash.absorb(W_new.x.clone());
      h1_hash.absorb(W_new.y.clone());
      h1_hash.absorb(W_new.is_infinity);
      h1_hash.absorb(E_new.x.clone());
      h1_hash.absorb(E_new.y.clone());
      h1_hash.absorb(E_new.is_infinity);
      h1_hash.absorb(u_new);

      // absorb each of the limbs of X_r_new[0]
      for limb in Xr0_new_bn.into_iter() {
        h1_hash.absorb(limb);
      }

      // absorb each of the limbs of X_r_new[1]
      for limb in Xr1_new_bn.into_iter() {
        h1_hash.absorb(limb);
      }

      h1_hash.absorb(next_i.clone());
      h1_hash.absorb(z_0);
      h1_hash.absorb(z_next);
      let h1_new_bits = h1_hash.get_challenge(cs.namespace(|| "h1_new bits"))?;
      let h1_new = le_bits_to_num(cs.namespace(|| "h1_new"), h1_new_bits)?;
      let _ = h1_new.inputize(cs.namespace(|| "output h1_new"))?;
    } else {
      /***********************************************************************/
      // Check that h1 = Hash(params, u2, i)
      /***********************************************************************/

      let mut h1_hash: PoseidonROGadget<G::Base> = PoseidonROGadget::new(self.poseidon_constants);

      h1_hash.absorb(params.clone());
      h1_hash.absorb(W_r_x);
      h1_hash.absorb(W_r_y);
      h1_hash.absorb(W_r_inf);
      h1_hash.absorb(E_r_x);
      h1_hash.absorb(E_r_y);
      h1_hash.absorb(E_r_inf);
      h1_hash.absorb(u_r);
      h1_hash.absorb(i.clone());

      //absorb each of the limbs of X_r[0]
      for limb in Xr0_bn.into_iter() {
        h1_hash.absorb(limb);
      }

      //absorb each of the limbs of X_r[1]
      for limb in Xr1_bn.into_iter() {
        h1_hash.absorb(limb);
      }

      let hash_bits = h1_hash.get_challenge(cs.namespace(|| "Input hash"))?;
      let hash = le_bits_to_num(cs.namespace(|| "bits to hash"), hash_bits)?;

      cs.enforce(
        || "check h1",
        |lc| lc,
        |lc| lc,
        |lc| lc + h1.get_variable() - hash.get_variable(),
      );

      /***********************************************************************/
      // Compute the new hash H(params, u2')
      /***********************************************************************/

      h1_hash.flush_state();
      h1_hash.absorb(params);
      h1_hash.absorb(W_new.x.clone());
      h1_hash.absorb(W_new.y.clone());
      h1_hash.absorb(W_new.is_infinity);
      h1_hash.absorb(E_new.x.clone());
      h1_hash.absorb(E_new.y.clone());
      h1_hash.absorb(E_new.is_infinity);
      h1_hash.absorb(u_new);
      h1_hash.absorb(next_i.clone());

      // absorb each of the limbs of X_r_new[0]
      for limb in Xr0_new_bn.into_iter() {
        h1_hash.absorb(limb);
      }

      // absorb each of the limbs of X_r_new[1]
      for limb in Xr1_new_bn.into_iter() {
        h1_hash.absorb(limb);
      }

      let h1_new_bits = h1_hash.get_challenge(cs.namespace(|| "h1_new bits"))?;
      let h1_new = le_bits_to_num(cs.namespace(|| "h1_new"), h1_new_bits)?;
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
      _cs: &mut CS,
      z: AllocatedNum<F>,
    ) -> Result<AllocatedNum<F>, SynthesisError> {
      Ok(z)
    }
  }

  #[test]
  fn test_verification_circuit() {
    // We experiment with 8 limbs of 32 bits each
    let params = NIFSVerifierCircuitParams::new(32, 8);
    // The first circuit that verifies G2
    let poseidon_constants1: NovaPoseidonConstants<<G2 as Group>::Base> =
      NovaPoseidonConstants::new();
    let circuit1: NIFSVerifierCircuit<G2, TestCircuit<<G2 as Group>::Base>> =
      NIFSVerifierCircuit::new(
        params.clone(),
        None,
        Some(TestCircuit {
          _p: Default::default(),
        }),
        poseidon_constants1.clone(),
      );
    // First create the shape
    let mut cs: ShapeCS<G1> = ShapeCS::new();
    let _ = circuit1.synthesize(&mut cs);
    let shape1 = cs.r1cs_shape();
    let gens1 = cs.r1cs_gens();
    println!(
      "Circuit1 -> Number of constraints: {}",
      cs.num_constraints()
    );

    // The second circuit that verifies G1
    let poseidon_constants2: NovaPoseidonConstants<<G1 as Group>::Base> =
      NovaPoseidonConstants::new();
    let circuit2: NIFSVerifierCircuit<G1, TestCircuit<<G1 as Group>::Base>> =
      NIFSVerifierCircuit::new(params.clone(), None, None, poseidon_constants2);
    // First create the shape
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
      "332553638888022689042501686561503049809",
    )
    .unwrap();
    let T = vec![<G2 as Group>::Scalar::zero()].commit(&gens2.gens_E);
    let w = vec![<G2 as Group>::Scalar::zero()].commit(&gens2.gens_E);
    // Now get an assignment
    let mut cs: SatisfyingAssignment<G1> = SatisfyingAssignment::new();
    let inputs: NIFSVerifierCircuitInputs<G2> = NIFSVerifierCircuitInputs::new(
      default_hash,
      RelaxedR1CSInstance::default(&gens2, &shape2),
      <<G2 as Group>::Base as PrimeField>::zero(), // TODO: provide real inputs
      <<G2 as Group>::Base as PrimeField>::zero(), // TODO: provide real inputs
      <<G2 as Group>::Base as PrimeField>::zero(), // TODO: provide real inputs
      <<G2 as Group>::Scalar as PrimeField>::zero(), // TODO: provide real inputs
      <<G2 as Group>::Base as PrimeField>::zero(), // TODO: provide real inputs
      T,                                           // TODO: provide real inputs
      w,
    );
    let circuit: NIFSVerifierCircuit<G2, TestCircuit<<G2 as Group>::Base>> =
      NIFSVerifierCircuit::new(
        params,
        Some(inputs),
        Some(TestCircuit {
          _p: Default::default(),
        }),
        poseidon_constants1,
      );
    let _ = circuit.synthesize(&mut cs);
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape1, &gens1).unwrap();

    // Make sure that this is satisfiable
    assert!(shape1.is_sat(&gens1, &inst, &witness).is_ok());
  }
}
