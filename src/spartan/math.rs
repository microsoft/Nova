//! This module provides mathematical utilities for Spartan.

/// A trait providing mathematical operations on `usize`.
pub trait Math {
  /// Computes the base-2 logarithm of the value.
  ///
  /// # Panics
  /// Panics if the value is zero.
  fn log_2(self) -> usize;
}

impl Math for usize {
  fn log_2(self) -> usize {
    assert_ne!(self, 0);

    if self.is_power_of_two() {
      (1usize.leading_zeros() - self.leading_zeros()) as usize
    } else {
      (0usize.leading_zeros() - self.leading_zeros()) as usize
    }
  }
}
