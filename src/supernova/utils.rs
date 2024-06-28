use bellpepper_core::{
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
  ConstraintSystem, LinearCombination, SynthesisError,
};
use ff::PrimeField;
use itertools::Itertools as _;

use crate::{
  gadgets::r1cs::{conditionally_select_alloc_relaxed_r1cs, AllocatedRelaxedR1CSInstance},
  traits::Engine,
};

/// Return the element of `a` given by the indicator bit in `selector_vec`.
///
/// This function assumes `selector_vec` has been properly constrained", i.e. that exactly one entry is equal to 1.  
//
// NOTE: When `a` is greater than 5 (estimated), it will be cheaper to use a multicase gadget.
//
// We should plan to rely on a well-designed gadget offering a common interface but that adapts its implementation based
// on the size of inputs (known at synthesis time). The threshold size depends on the size of the elements of `a`. The
// larger the elements, the fewer are needed before multicase becomes cost-effective.
pub fn get_from_vec_alloc_relaxed_r1cs<E: Engine, CS: ConstraintSystem<<E as Engine>::Base>>(
  mut cs: CS,
  a: &[AllocatedRelaxedR1CSInstance<E>],
  selector_vec: &[Boolean],
) -> Result<AllocatedRelaxedR1CSInstance<E>, SynthesisError> {
  assert_eq!(a.len(), selector_vec.len());

  // Compare all instances in `a` to the first one
  let first: AllocatedRelaxedR1CSInstance<E> = a
    .first()
    .cloned()
    .ok_or_else(|| SynthesisError::IncompatibleLengthVector("empty vec length".to_string()))?;

  // Since `selector_vec` is correct, only one entry is 1.
  // If selector_vec[0] is 1, then all `conditionally_select` will return `first`.
  // Otherwise, the correct instance will be selected.
  let selected = a
    .iter()
    .zip_eq(selector_vec.iter())
    .enumerate()
    .skip(1)
    .try_fold(first, |matched, (i, (candidate, equal_bit))| {
      conditionally_select_alloc_relaxed_r1cs(
        cs.namespace(|| format!("next_matched_allocated-{:?}", i)),
        candidate,
        &matched,
        equal_bit,
      )
    })?;

  Ok(selected)
}

/// Compute a selector vector `s` of size `num_indices`, such that
/// `s[i] == 1` if i == `target_index` and 0 otherwise.
pub fn get_selector_vec_from_index<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  target_index: &AllocatedNum<F>,
  num_indices: usize,
) -> Result<Vec<Boolean>, SynthesisError> {
  assert_ne!(num_indices, 0);

  // Compute the selector vector non-deterministically
  let selector = (0..num_indices)
    .map(|idx| {
      // b <- idx == target_index
      Ok(Boolean::Is(AllocatedBit::alloc(
        cs.namespace(|| format!("allocate s_{:?}", idx)),
        target_index.get_value().map(|v| v == F::from(idx as u64)),
      )?))
    })
    .collect::<Result<Vec<Boolean>, SynthesisError>>()?;

  // Enforce ∑ selector[i] = 1
  {
    let selected_sum = selector.iter().fold(LinearCombination::zero(), |lc, bit| {
      lc + &bit.lc(CS::one(), F::ONE)
    });
    cs.enforce(
      || "exactly-one-selection",
      |_| selected_sum,
      |lc| lc + CS::one(),
      |lc| lc + CS::one(),
    );
  }

  // Enforce `target_index - ∑ i * selector[i] = 0``
  {
    let selected_value = selector
      .iter()
      .enumerate()
      .fold(LinearCombination::zero(), |lc, (i, bit)| {
        lc + &bit.lc(CS::one(), F::from(i as u64))
      });
    cs.enforce(
      || "target_index - ∑ i * selector[i] = 0",
      |lc| lc,
      |lc| lc,
      |lc| lc + target_index.get_variable() - &selected_value,
    );
  }

  Ok(selector)
}

#[cfg(test)]
mod test {
  use crate::provider::PallasEngine;

  use super::*;
  use bellpepper_core::test_cs::TestConstraintSystem;
  use pasta_curves::pallas::Base;

  #[test]
  fn test_get_from_vec_alloc_relaxed_r1cs_bounds() {
    let n = 3;
    for selected in 0..(2 * n) {
      let mut cs = TestConstraintSystem::<Base>::new();

      let allocated_target = AllocatedNum::alloc_infallible(&mut cs.namespace(|| "target"), || {
        Base::from(selected as u64)
      });

      let selector_vec = get_selector_vec_from_index(&mut cs, &allocated_target, n).unwrap();

      let vec = (0..n)
        .map(|i| {
          AllocatedRelaxedR1CSInstance::<PallasEngine>::default(
            &mut cs.namespace(|| format!("elt-{i}")),
            4,
            64,
          )
          .unwrap()
        })
        .collect::<Vec<_>>();

      get_from_vec_alloc_relaxed_r1cs(&mut cs.namespace(|| "test-fn"), &vec, &selector_vec)
        .unwrap();

      if selected < n {
        assert!(cs.is_satisfied())
      } else {
        // If selected is out of range, the circuit must be unsatisfied.
        assert!(!cs.is_satisfied())
      }
    }
  }

  #[test]
  fn test_get_selector() {
    for n in 1..4 {
      for selected in 0..(2 * n) {
        let mut cs = TestConstraintSystem::<Base>::new();

        let allocated_target =
          AllocatedNum::alloc_infallible(&mut cs.namespace(|| "target"), || {
            Base::from(selected as u64)
          });

        let selector_vec = get_selector_vec_from_index(&mut cs, &allocated_target, n).unwrap();

        if selected < n {
          // Check that the selector bits are correct
          assert_eq!(selector_vec.len(), n);
          for (i, bit) in selector_vec.iter().enumerate() {
            assert_eq!(bit.get_value().unwrap(), i == selected);
          }

          assert!(cs.is_satisfied());
        } else {
          // If selected is out of range, the circuit must be unsatisfied.
          assert!(!cs.is_satisfied());
        }
      }
    }
  }
}
