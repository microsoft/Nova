use ff::PrimeField;
#[cfg(not(target_arch = "wasm32"))]
use proptest::prelude::*;

/// Wrapper struct around a field element that implements additional traits
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FWrap<F: PrimeField>(pub F);

impl<F: PrimeField> Copy for FWrap<F> {}

#[cfg(not(target_arch = "wasm32"))]
/// Trait implementation for generating `FWrap<F>` instances with proptest
impl<F: PrimeField> Arbitrary for FWrap<F> {
  type Parameters = ();
  type Strategy = BoxedStrategy<Self>;

  fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
    use rand::rngs::StdRng;
    use rand_core::SeedableRng;

    let strategy = any::<[u8; 32]>()
      .prop_map(|seed| FWrap(F::random(StdRng::from_seed(seed))))
      .no_shrink();
    strategy.boxed()
  }
}
