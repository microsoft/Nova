//! Logup-GKR fractional-sum memory-check argument.
//!
//! Replaces the inverse-logup memory-check in `ppsnark.rs` (the
//! `MemorySumcheckInstance` 6-route sumcheck + 4 inverse-polynomial
//! commitments) with four equal-height fractional-sum GKR trees: the table and
//! access sides of the row and column relations. Projective fractions keep the
//! circuit inversion-free and eliminate the four inverse-polynomial
//! commitments.
//!
//! ## Module map
//! - [`fraction`]: projective fraction + 2-to-1 gate (pure).
//! - [`layer`]: the `Layer` type (one tree level: num/den MLEs).
//! - [`proof`]: proof/claim interface.
//! - [`prover`]: fold-up + batched sumcheck prover.
//! - [`verifier`]: fold-down + root check, emits the shared opening claim.
//!
//! ## Boundary with ppSNARK (host reconcile contract)
//! The argument owns no commitment scheme. Its verifier returns a
//! [`proof::LogupGkrOpeningClaim`]: a single shared `eval_point` plus the
//! four input-layer fractions `openings`, ordered `[row_table, row_access,
//! col_table, col_access]`. The **host** then:
//! 1. rerandomizes the seven claimed columns `[L_row, L_col, addr_row, addr_col,
//!    ts_row, ts_col, mem_col]` from `eval_point` into the inner sumcheck and
//!    binds them at `r_inner_batched` through the batched PCS opening;
//! 2. recomputes the four fractions from those claims (`num = ts` on the table
//!    sides, `num = -1` on the access sides) and checks them against `openings`;
//! 3. runs the `0/den` zero-sum balance check.
//!
//! Steps 2-3 are the host's job, never the GKR verifier's.

pub mod fraction;
pub mod layer;
pub mod proof;
pub mod prover;
pub mod verifier;

pub use proof::{LogupGkrOpeningClaim, LogupGkrProof};
