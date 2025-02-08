//! This module implements a folding scheme for R1CS using techniques from the NeutronNova paper

pub mod nifs;
pub mod running_instance;
pub(crate) mod sumfold;
#[cfg(test)]
mod tests;
