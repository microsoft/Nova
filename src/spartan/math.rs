pub trait Math {
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
