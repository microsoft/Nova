//! This module defines relations used in the Neutron folding scheme
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS},
  digest::{DigestComputer, SimpleDigestible},
  errors::NovaError,
  gadgets::{
    nonnative::{bignat::nat_to_limbs, util::f_to_nat},
    utils::scalar_as_base,
  },
  traits::{
    commitment::CommitmentEngineTrait, AbsorbInROTrait, Engine, ROTrait, TranscriptReprTrait,
  },
  Commitment, CommitmentKey, DerandKey, CE,
};
use core::{cmp::max, marker::PhantomData};
use ff::Field;
use once_cell::sync::OnceCell;
use rand_core::OsRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};