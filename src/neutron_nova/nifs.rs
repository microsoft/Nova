//! This module implements a non-interactive folding scheme specified in the NeutronNova paper

use super::{
  running_instance::{
    NSCInstance, NSCPCInstance, NSCPCWitness, NSCWitness, RunningZFInstance, RunningZFWitness,
    ZCPCInstance, ZCPCWitness,
  },
  sumfold::SumFoldProof,
};

use crate::{
  errors::NovaError,
  neutron_nova::sumfold::{
    nsc_pc_to_sumfold_inputs, nsc_to_sumfold_inputs, sumfold, PCSumFoldInputs, SumFoldInputTrait,
  },
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness},
  spartan::polys::eq::EqPolynomial,
  spartan::{math::Math, polys::power2::PowPoly},
  traits::{Engine, TranscriptEngineTrait},
  Commitment, CommitmentKey,
};
use ff::Field;
use rand_core::OsRng;
use serde::{Deserialize, Serialize};

/// A SNARK that holds the proof of a step of an incremental computation
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NIFS<E>
where
  E: Engine,
{
  comm_e: Commitment<E>,
  sf_proof: SumFoldProof<E>,
  T: E::Scalar,
  T_pc: E::Scalar,
}

impl<E> NIFS<E>
where
  E: Engine,
{
  /// Implement prover for the R1CS NeutronNova folding scheme
  pub fn prove(
    S: &R1CSShape<E>,
    ck: &CommitmentKey<E>,
    U1: &RunningZFInstance<E>,
    W1: &RunningZFWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
  ) -> Result<(Self, (RunningZFInstance<E>, RunningZFWitness<E>)), NovaError> {
    let mut transcript = E::TE::new(b"NeutronNova");
    transcript.absorb(b"U2", U2);
    let (nsc_U2, nsc_W2, nsc_pc_U2, nsc_pc_W2, new_zc_pc_U, new_zc_pc_W) =
      ZeroCheckReduction::prove(
        ck,
        &mut transcript,
        U1.zc_pc(),
        W1.zc_pc(),
        U2.clone(),
        W2.clone(),
        S.num_cons.log_2(),
      )?;
    let g = nsc_to_sumfold_inputs(S, U1.nsc().U(), W1.nsc().W(), W1.nsc().e())?;
    let h = nsc_to_sumfold_inputs(S, nsc_U2.U(), nsc_W2.W(), nsc_W2.e())?;
    let F =
      |az: E::Scalar, bz: E::Scalar, cz: E::Scalar, h1: E::Scalar, h2: E::Scalar| -> E::Scalar {
        (az * bz - cz) * h1 * h2
      };
    let g_pc = nsc_pc_to_sumfold_inputs(W1.nsc_pc().e(), W1.nsc_pc().new_e())?;
    let h_pc = nsc_pc_to_sumfold_inputs(nsc_pc_W2.e(), nsc_pc_W2.new_e())?;
    let F_pc =
      |g1: E::Scalar, g2: E::Scalar, g3: E::Scalar, h1: E::Scalar, h2: E::Scalar| -> E::Scalar {
        (g1 - g2 * g3) * h1 * h2
      };
    let gamma = transcript.squeeze(b"gamma")?;
    let (_, sf_proof, _, r_b) = sumfold(
      &mut transcript,
      &g,
      &h,
      U1.nsc().T(),
      F,
      &g_pc,
      &h_pc,
      U1.nsc_pc().T(),
      F_pc,
      gamma,
    )?;

    // NSCInstance::new(T, U, folded_comm_e)
    let (U1_r1cs, U2_r1cs) = (U1.nsc().U(), nsc_U2.U());
    let U_r1cs = U1_r1cs.fold(U2_r1cs, r_b);
    let folded_comm_e_nsc = U1.nsc().comm_e() * (E::Scalar::ONE - r_b) + nsc_U2.comm_e() * r_b;

    // NSCPCInstance::new(T, folded_comm_e, tau, folded_new_comm_e)
    let folded_comm_e_pc = U1.nsc_pc().comm_e() * (E::Scalar::ONE - r_b) + nsc_pc_U2.comm_e() * r_b;
    let folded_new_comm_e_pc =
      U1.nsc_pc().new_comm_e() * (E::Scalar::ONE - r_b) + nsc_pc_U2.new_comm_e() * r_b;
    let tau_pc = U1.nsc_pc().tau() * (E::Scalar::ONE - r_b) + nsc_pc_U2.tau() * r_b;

    let comm_e = new_zc_pc_U.comm_e();

    // NSCWitness::new(W, folded_e, r_e)
    let (W1_r1cs, W2_r1cs) = (W1.nsc().W(), nsc_W2.W());
    let W_r1cs = W1_r1cs.fold(W2_r1cs, r_b);
    let folded_e_r1cs = W1.nsc().e().fold(nsc_W2.e(), r_b);
    let r_e_r1cs = W1.nsc().r_e() * (E::Scalar::ONE - r_b) + nsc_W2.r_e() * r_b;

    // NSCPCWitness::new(folded_e, new_folded_e, r_e, new_r_e)
    let folded_e_pc = W1.nsc_pc().e().fold(nsc_pc_W2.e(), r_b);
    let new_folded_e_pc = W1.nsc_pc().new_e().fold(nsc_pc_W2.new_e(), r_b);
    let r_e_pc = W1.nsc_pc().r_e() * (E::Scalar::ONE - r_b) + nsc_pc_W2.r_e() * r_b;
    let new_r_e_pc = W1.nsc_pc().new_r_e() * (E::Scalar::ONE - r_b) + nsc_pc_W2.new_r_e() * r_b;

    // Compute T
    let new_g = nsc_to_sumfold_inputs(S, &U_r1cs, &W_r1cs, &folded_e_r1cs)?;
    let (az, bz, cz, h1, h2) = new_g.inner();
    let T: E::Scalar = (0..2usize.pow(new_g.num_vars() as u32))
      .map(|i| F(az[i], bz[i], cz[i], h1[i], h2[i]))
      .sum();
    transcript.absorb(b"T", &T);

    // Compute T_pc
    let new_g_pc: PCSumFoldInputs<E> = nsc_pc_to_sumfold_inputs(&folded_e_pc, &new_folded_e_pc)?;
    let (g1_pc, g2_pc, g3_pc, h1_pc, h2_pc) = new_g_pc.inner();
    let T_pc: E::Scalar = (0..2usize.pow(new_g_pc.num_vars() as u32))
      .map(|i| F_pc(g1_pc[i], g2_pc[i], g3_pc[i], h1_pc[i], h2_pc[i]))
      .sum();
    transcript.absorb(b"T_pc", &T_pc);

    let nifs = Self {
      comm_e,
      sf_proof,
      T,
      T_pc,
    };
    let U = RunningZFInstance {
      nsc: NSCInstance::new(T, U_r1cs, folded_comm_e_nsc),
      nsc_pc: NSCPCInstance::new(T_pc, folded_comm_e_pc, tau_pc, folded_new_comm_e_pc),
      zc_pc: new_zc_pc_U,
    };
    let W = RunningZFWitness {
      nsc: NSCWitness::new(W_r1cs, folded_e_r1cs, r_e_r1cs),
      nsc_pc: NSCPCWitness::new(folded_e_pc, new_folded_e_pc, r_e_pc, new_r_e_pc),
      zc_pc: new_zc_pc_W,
    };

    Ok((nifs, (U, W)))
  }

  /// Implement verifier for the R1CS NeutronNova folding scheme
  pub fn verify(
    &self,
    U1: &RunningZFInstance<E>,
    U2: &R1CSInstance<E>,
  ) -> Result<RunningZFInstance<E>, NovaError> {
    let mut transcript = E::TE::new(b"NeutronNova");
    transcript.absorb(b"U2", U2);
    let (nsc_U2, nsc_pc_U2, new_zc_pc_U) =
      ZeroCheckReduction::verify(&mut transcript, U1.zc_pc(), U2.clone(), self.comm_e)?;
    let gamma = transcript.squeeze(b"gamma")?;
    let (c, beta, r_b) = self.sf_proof.verify(
      &mut transcript,
      U1.nsc().T() + gamma * U1.nsc_pc.T(),
      E::Scalar::ZERO,
    )?;
    // NSCInstance::new(T, U, folded_comm_e)
    let (U1_r1cs, U2_r1cs) = (U1.nsc().U(), nsc_U2.U());
    let U_r1cs = U1_r1cs.fold(U2_r1cs, r_b);
    let folded_comm_e_nsc = U1.nsc().comm_e() * (E::Scalar::ONE - r_b) + nsc_U2.comm_e() * r_b;

    // NSCPCInstance::new(T, folded_comm_e, tau, folded_new_comm_e)
    let folded_comm_e_pc = U1.nsc_pc().comm_e() * (E::Scalar::ONE - r_b) + nsc_pc_U2.comm_e() * r_b;
    let folded_new_comm_e_pc =
      U1.nsc_pc().new_comm_e() * (E::Scalar::ONE - r_b) + nsc_pc_U2.new_comm_e() * r_b;
    let tau_pc = U1.nsc_pc().tau() * (E::Scalar::ONE - r_b) + nsc_pc_U2.tau() * r_b;
    transcript.absorb(b"T", &self.T);
    transcript.absorb(b"T_pc", &self.T_pc);

    //  Check T_γ = T + γ · T_pc,
    let T_gamma = c
      * (EqPolynomial::new(vec![beta])
        .evaluate(&[r_b])
        .invert()
        .unwrap());
    if T_gamma != self.T + gamma * self.T_pc {
      return Err(NovaError::ProofVerifyError);
    }

    // Ouput instance
    let U = RunningZFInstance {
      nsc: NSCInstance::new(self.T, U_r1cs, folded_comm_e_nsc),
      nsc_pc: NSCPCInstance::new(self.T_pc, folded_comm_e_pc, tau_pc, folded_new_comm_e_pc),
      zc_pc: new_zc_pc_U,
    };
    Ok(U)
  }
}

struct ZeroCheckReduction;

impl ZeroCheckReduction {
  pub fn prove<E>(
    ck: &CommitmentKey<E>,
    transcript: &mut E::TE,
    ZC_PC_U: &ZCPCInstance<E>,
    ZC_PC_W: &ZCPCWitness<E>,
    U: R1CSInstance<E>,
    W: R1CSWitness<E>,
    ell: usize,
  ) -> Result<
    (
      NSCInstance<E>,
      NSCWitness<E>,
      NSCPCInstance<E>,
      NSCPCWitness<E>,
      ZCPCInstance<E>,
      ZCPCWitness<E>,
    ),
    NovaError,
  >
  where
    E: Engine,
  {
    // let tau = ro.squeeze(NUM_CHALLENGE_BITS);
    let tau = transcript.squeeze(b"tau")?;
    let e = PowPoly::new(tau, ell);
    let r_e = E::Scalar::random(&mut OsRng);
    let comm_e = e.commit::<E>(ck, r_e);
    transcript.absorb(b"comm_e", &comm_e);
    let nsc_U = NSCInstance::new(E::Scalar::ZERO, U, comm_e);
    let nsc_W = NSCWitness::new(W, e.clone(), r_e);
    let nsc_pc_U = NSCPCInstance::new(E::Scalar::ZERO, ZC_PC_U.comm_e(), tau, comm_e);
    let nsc_pc_W = NSCPCWitness::new(ZC_PC_W.e().clone(), e.clone(), ZC_PC_W.r_e(), r_e);
    let new_zc_pc_U = ZCPCInstance::new(comm_e, tau);
    let new_zc_pc_W = ZCPCWitness::new(e, r_e);
    Ok((nsc_U, nsc_W, nsc_pc_U, nsc_pc_W, new_zc_pc_U, new_zc_pc_W))
  }

  pub fn verify<E>(
    transcript: &mut E::TE,
    ZC_PC_U: &ZCPCInstance<E>,
    U: R1CSInstance<E>,
    comm_e: Commitment<E>,
  ) -> Result<(NSCInstance<E>, NSCPCInstance<E>, ZCPCInstance<E>), NovaError>
  where
    E: Engine,
  {
    let tau = transcript.squeeze(b"tau")?;
    transcript.absorb(b"comm_e", &comm_e);
    let nsc_U = NSCInstance::new(E::Scalar::ZERO, U, comm_e);
    let nsc_pc_U = NSCPCInstance::new(E::Scalar::ZERO, ZC_PC_U.comm_e(), tau, comm_e);
    let new_zc_pc_U = ZCPCInstance::new(comm_e, tau);
    Ok((nsc_U, nsc_pc_U, new_zc_pc_U))
  }
}
