//! Demonstrates how to use Nova to produce a recursive proof of the correct execution of
//! iterations of the key generation function.
//! Generate public key using G1 curve, and prove using G2 curve.

use bellperson::{
  gadgets::{boolean::AllocatedBit, num::AllocatedNum},
  ConstraintSystem, SynthesisError,
};
use ff::{Field, PrimeField, PrimeFieldBits};
use generic_array::typenum::U5;
use neptune::{
  circuit::poseidon_hash,
  poseidon::{Poseidon, PoseidonConstants},
  Strength,
};
use nova_snark::{
  gadgets::ecc::AllocatedPoint,
  traits::{
    circuit::{StepCircuit, TrivialTestCircuit},
    Group as Nova_Group,
  },
  CompressedSNARK, PublicParams, RecursiveSNARK,
};
use pasta_curves::{
  arithmetic::CurveAffine,
  group::{Curve, Group},
};
use rand::{rngs::OsRng, RngCore};
use subtle::Choice;

type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;
type S1 = nova_snark::spartan_with_ipa_pc::RelaxedR1CSSNARK<G2>;
type S2 = nova_snark::spartan_with_ipa_pc::RelaxedR1CSSNARK<G1>;

#[derive(Debug, Clone, Copy)]
pub struct SecretKey(pub <G1 as Group>::Scalar);

impl SecretKey {
  pub fn random(mut rng: impl RngCore) -> Self {
    let secret = <G1 as Group>::Scalar::random(&mut rng);
    Self(secret)
  }
}

#[derive(Debug, Clone, Copy)]
pub struct PublicKey(pub G1);

impl PublicKey {
  pub fn from_secret_key(s: &SecretKey) -> Self {
    let point = G1::generator() * s.0;
    Self(point)
  }
}

// An affine point coordinate that is on the curve.
#[derive(Clone, Copy, Debug)]
pub struct Coordinate<F>
where
  F: PrimeField<Repr = [u8; 32]>,
{
  pub x: F,
  pub y: F,
  pub is_infinity: bool,
}

impl<F> Coordinate<F>
where
  F: PrimeField<Repr = [u8; 32]>,
{
  // New affine point coordiante on the curve so is_infinity = false.
  pub fn new(x: F, y: F) -> Self {
    Self {
      x,
      y,
      is_infinity: false,
    }
  }
}

// A generated key
#[derive(Clone, Debug)]
pub struct GeneratedKey<Fb, Fs>
where
  Fb: PrimeField<Repr = [u8; 32]>,
  Fs: PrimeField<Repr = [u8; 32]> + PrimeFieldBits,
{
  g: Coordinate<Fb>, // generator of the group; could be omitted if Nova's traits allow accessing the generator
  pk: Coordinate<Fb>, // public key
  sk: Fs,            // secret key
}

impl<Fb, Fs> GeneratedKey<Fb, Fs>
where
  Fb: PrimeField<Repr = [u8; 32]>,
  Fs: PrimeField<Repr = [u8; 32]> + PrimeFieldBits,
{
  pub fn new(g: Coordinate<Fb>, pk: Coordinate<Fb>, sk: Fs) -> Self {
    Self { g, pk, sk }
  }
}

// An key generation proof that we will use on the primary curve
#[derive(Clone, Debug)]
pub struct KeyGenerationCircuit<F>
where
  F: PrimeField<Repr = [u8; 32]>,
{
  pub z_g: Coordinate<F>,
  pub z_pk: Coordinate<F>,
  pub z_sk: F,
  pub g: Coordinate<F>,
  pub pk: Coordinate<F>,
  pub sk: F,
  pub sk_bits: Vec<Choice>,
  pub pc: PoseidonConstants<F, U5>,
}

impl<F> KeyGenerationCircuit<F>
where
  F: PrimeField<Repr = [u8; 32]>,
{
  // Creates a new [`GeneratedKey<Fb, Fs>`]. The base and scalar field elements from the curve
  // field used by the signature are converted to scalar field elements from the cyclic curve
  // field used by the circuit.
  pub fn new<Fb, Fs>(
    num_steps: usize,
    generated_keys: &[GeneratedKey<Fb, Fs>],
    pc: &PoseidonConstants<F, U5>,
  ) -> (F, Vec<Self>)
  where
    Fb: PrimeField<Repr = [u8; 32]>,
    Fs: PrimeField<Repr = [u8; 32]> + PrimeFieldBits,
  {
    let mut z0 = F::zero();
    let mut circuits = Vec::new();
    for i in 0..num_steps {
      let mut j = i;
      if i > 0 {
        j = i - 1
      };
      let z_generated_key = &generated_keys[j];
      let z_g = Coordinate::new(
        F::from_repr(z_generated_key.g.x.to_repr()).unwrap(),
        F::from_repr(z_generated_key.g.y.to_repr()).unwrap(),
      );

      let z_pk = Coordinate::new(
        F::from_repr(z_generated_key.pk.x.to_repr()).unwrap(),
        F::from_repr(z_generated_key.pk.y.to_repr()).unwrap(),
      );

      let z_sk = F::from_repr(z_generated_key.sk.to_repr()).unwrap();

      let generated_key = &generated_keys[i];
      let g = Coordinate::new(
        F::from_repr(generated_key.g.x.to_repr()).unwrap(),
        F::from_repr(generated_key.g.y.to_repr()).unwrap(),
      );

      let pk = Coordinate::new(
        F::from_repr(generated_key.pk.x.to_repr()).unwrap(),
        F::from_repr(generated_key.pk.y.to_repr()).unwrap(),
      );

      let sk_bits = Self::to_le_bits(&generated_key.sk);
      let sk = F::from_repr(generated_key.sk.to_repr()).unwrap();

      let circuit = KeyGenerationCircuit {
        z_g,
        z_pk,
        z_sk,
        g,
        pk,
        sk,
        sk_bits,
        pc: pc.clone(),
      };
      circuits.push(circuit);

      if i == 0 {
        z0 = Poseidon::<F, U5>::new_with_preimage(&[g.x, g.y, pk.x, pk.y, sk], pc).hash();
      }
    }

    (z0, circuits)
  }

  // Converts the scalar field element from the curve used by the signature to a bit represenation
  // for later use in scalar multiplication using the cyclic curve used by the circuit.
  fn to_le_bits<Fs>(fs: &Fs) -> Vec<Choice>
  where
    Fs: PrimeField<Repr = [u8; 32]> + PrimeFieldBits,
  {
    let bits = fs
      .to_repr()
      .iter()
      .flat_map(|byte| (0..8).map(move |i| Choice::from((byte >> i) & 1u8)))
      .collect::<Vec<Choice>>();
    bits
  }

  // Synthesize a bit representation into circuit gadgets.
  fn synthesize_bits<CS: ConstraintSystem<F>>(
    cs: &mut CS,
    bits: &[Choice],
  ) -> Result<Vec<AllocatedBit>, SynthesisError> {
    let alloc_bits: Vec<AllocatedBit> = bits
      .iter()
      .enumerate()
      .map(|(i, bit)| {
        AllocatedBit::alloc(
          cs.namespace(|| format!("bit {}", i)),
          Some(bit.unwrap_u8() == 1u8),
        )
      })
      .collect::<Result<Vec<AllocatedBit>, SynthesisError>>()
      .unwrap();
    Ok(alloc_bits)
  }
}

impl<F> StepCircuit<F> for KeyGenerationCircuit<F>
where
  F: PrimeField<Repr = [u8; 32]> + PrimeFieldBits,
{
  // Prove knowledge of the sk used to generate the public key PK.
  // pk == [sk]G
  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: AllocatedNum<F>,
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    let z_gx = AllocatedNum::alloc(cs.namespace(|| "z_gx"), || Ok(self.z_g.x))?;
    let z_gy = AllocatedNum::alloc(cs.namespace(|| "z_gy"), || Ok(self.z_g.y))?;
    let z_pkx = AllocatedNum::alloc(cs.namespace(|| "z_pkx"), || Ok(self.z_pk.x))?;
    let z_pky = AllocatedNum::alloc(cs.namespace(|| "z_pky"), || Ok(self.z_pk.y))?;
    let z_sk = AllocatedNum::alloc(cs.namespace(|| "z_sk"), || Ok(self.z_sk))?;

    let z_hash = poseidon_hash(
      cs.namespace(|| "input hash"),
      vec![z_gx, z_gy, z_pkx, z_pky, z_sk],
      &self.pc,
    )?;

    cs.enforce(
      || "z == z1",
      |lc| lc + z.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + z_hash.get_variable(),
    );

    let g = AllocatedPoint::alloc(
      cs.namespace(|| "G"),
      Some((self.g.x, self.g.y, self.g.is_infinity)),
    )?;
    let sk_bits = Self::synthesize_bits(&mut cs.namespace(|| "sk_bits"), &self.sk_bits)?;
    let skg = g.scalar_mul(cs.namespace(|| "[sk]G"), sk_bits)?;
    let pk = AllocatedPoint::alloc(
      cs.namespace(|| "PK"),
      Some((self.pk.x, self.pk.y, self.pk.is_infinity)),
    )?;

    let (skg_x, skg_y, _) = skg.get_coordinates();
    let (pk_x, pk_y, _) = pk.get_coordinates();

    cs.enforce(
      || "skg_x == pk_x",
      |lc| lc + skg_x.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + pk_x.get_variable(),
    );

    cs.enforce(
      || "skg_y == pk_y",
      |lc| lc + skg_y.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + pk_y.get_variable(),
    );

    let gx = AllocatedNum::alloc(cs.namespace(|| "gx"), || Ok(self.g.x))?;
    let gy = AllocatedNum::alloc(cs.namespace(|| "gy"), || Ok(self.g.y))?;
    let pkx = AllocatedNum::alloc(cs.namespace(|| "pkx"), || Ok(self.pk.x))?;
    let pky = AllocatedNum::alloc(cs.namespace(|| "pky"), || Ok(self.pk.y))?;
    let sk = AllocatedNum::alloc(cs.namespace(|| "sk"), || Ok(self.sk))?;

    poseidon_hash(
      cs.namespace(|| "output hash"),
      vec![gx, gy, pkx, pky, sk],
      &self.pc,
    )
  }

  fn compute(&self, z: &F) -> F {
    let z_hash = Poseidon::<F, U5>::new_with_preimage(
      &[self.z_g.x, self.z_g.y, self.z_pk.x, self.z_pk.y, self.z_sk],
      &self.pc,
    )
    .hash();
    debug_assert_eq!(z, &z_hash);

    Poseidon::<F, U5>::new_with_preimage(
      &[self.g.x, self.g.y, self.pk.x, self.pk.y, self.sk],
      &self.pc,
    )
    .hash()
  }
}

fn main() {
  // produce public parameters
  println!("Generating public parameters...");

  let pc = PoseidonConstants::<<G2 as Group>::Scalar, U5>::new_with_strength(Strength::Standard);
  let circuit_primary = KeyGenerationCircuit::<<G2 as Nova_Group>::Scalar> {
    z_g: Coordinate::new(
      <G2 as Nova_Group>::Scalar::zero(),
      <G2 as Nova_Group>::Scalar::zero(),
    ),
    z_pk: Coordinate::new(
      <G2 as Nova_Group>::Scalar::zero(),
      <G2 as Nova_Group>::Scalar::zero(),
    ),
    z_sk: <G2 as Nova_Group>::Scalar::zero(),
    g: Coordinate::new(
      <G2 as Nova_Group>::Scalar::zero(),
      <G2 as Nova_Group>::Scalar::zero(),
    ),
    pk: Coordinate::new(
      <G2 as Nova_Group>::Scalar::zero(),
      <G2 as Nova_Group>::Scalar::zero(),
    ),
    sk: <G2 as Nova_Group>::Scalar::zero(),
    sk_bits: vec![Choice::from(0u8); 256],
    pc: pc.clone(),
  };

  let circuit_secondary = TrivialTestCircuit::default();

  let pp = PublicParams::<
    G2,
    G1,
    KeyGenerationCircuit<<G2 as Group>::Scalar>,
    TrivialTestCircuit<<G1 as Group>::Scalar>,
  >::setup(circuit_primary, circuit_secondary.clone());

  // produce non-deterministic advice
  println!("Generating non-deterministic advice...");

  let num_steps = 3;

  let generated_keys = || {
    let mut generated_keys = Vec::new();
    for _ in 0..num_steps {
      let sk = SecretKey::random(&mut OsRng);
      let pk = PublicKey::from_secret_key(&sk);

      // Affine coordinates guaranteed to be on the curve
      let gxy = G1::generator().to_affine().coordinates().unwrap();
      let pkxy = pk.0.to_affine().coordinates().unwrap();

      generated_keys.push(GeneratedKey::<
        <G1 as Nova_Group>::Base,
        <G1 as Nova_Group>::Scalar,
      >::new(
        Coordinate::<<G1 as Nova_Group>::Base>::new(*gxy.x(), *gxy.y()),
        Coordinate::<<G1 as Nova_Group>::Base>::new(*pkxy.x(), *pkxy.y()),
        sk.0,
      ));
    }
    generated_keys
  };

  let (z0_primary, circuits_primary) = KeyGenerationCircuit::<<G2 as Nova_Group>::Scalar>::new::<
    <G1 as Nova_Group>::Base,
    <G1 as Nova_Group>::Scalar,
  >(num_steps, &generated_keys(), &pc);

  // Secondary circuit
  let z0_secondary = <G1 as Group>::Scalar::zero();

  // produce a recursive SNARK
  println!("Generating a RecursiveSNARK...");

  type C1 = KeyGenerationCircuit<<G2 as Nova_Group>::Scalar>;
  type C2 = TrivialTestCircuit<<G1 as Nova_Group>::Scalar>;

  let mut recursive_snark: Option<RecursiveSNARK<G2, G1, C1, C2>> = None;

  for (i, circuit_primary) in circuits_primary.iter().take(num_steps).enumerate() {
    let result = RecursiveSNARK::prove_step(
      &pp,
      recursive_snark,
      circuit_primary.clone(),
      circuit_secondary.clone(),
      z0_primary,
      z0_secondary,
    );
    assert!(result.is_ok());
    println!("RecursiveSNARK::prove_step {}: {:?}", i, result.is_ok());
    recursive_snark = Some(result.unwrap());
  }

  assert!(recursive_snark.is_some());
  let recursive_snark = recursive_snark.unwrap();

  // verify the recursive SNARK
  println!("Verifying the RecursiveSNARK...");
  let res = recursive_snark.verify(&pp, num_steps, z0_primary, z0_secondary);
  println!("RecursiveSNARK::verify: {:?}", res.is_ok());
  assert!(res.is_ok());

  // produce a compressed SNARK
  println!("Generating a CompressedSNARK...");
  let res = CompressedSNARK::<_, _, _, _, S1, S2>::prove(&pp, &recursive_snark);
  println!("CompressedSNARK::prove: {:?}", res.is_ok());
  assert!(res.is_ok());
  let compressed_snark = res.unwrap();

  // verify the compressed SNARK
  println!("Verifying a CompressedSNARK...");
  let res = compressed_snark.verify(&pp, num_steps, z0_primary, z0_secondary);
  println!("CompressedSNARK::verify: {:?}", res.is_ok());
  assert!(res.is_ok());
}
