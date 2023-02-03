use bellperson::{
  gadgets::{boolean::AllocatedBit, test::TestConstraintSystem},
  ConstraintSystem, SynthesisError,
};
use core::ops::{AddAssign, MulAssign};
use ff::{
  derive::byteorder::{ByteOrder, LittleEndian},
  Field, PrimeField, PrimeFieldBits,
};
use nova_snark::{gadgets::ecc::AllocatedPoint, traits::Group as NovaGroup};
use num_bigint::BigUint;
use pasta_curves::{
  arithmetic::CurveAffine,
  group::{Curve, Group},
};
use rand::{rngs::OsRng, RngCore};
use sha3::{Digest, Sha3_512};

#[derive(Debug, Clone, Copy)]
pub struct SecretKey<G: Group>(G::Scalar);

impl<G> SecretKey<G>
where
  G: Group,
{
  pub fn random(mut rng: impl RngCore) -> Self {
    let secret = G::Scalar::random(&mut rng);
    Self(secret)
  }
}

#[derive(Debug, Clone, Copy)]
pub struct PublicKey<G: Group>(G);

impl<G> PublicKey<G>
where
  G: Group,
{
  pub fn from_secret_key(s: &SecretKey<G>) -> Self {
    let point = G::generator() * s.0;
    Self(point)
  }
}

#[derive(Clone)]
pub struct Signature<G: Group> {
  pub r: G,
  pub s: G::Scalar,
}

impl<G> SecretKey<G>
where
  G: Group,
{
  pub fn sign(self, c: G::Scalar, mut rng: impl RngCore) -> Signature<G> {
    // T
    let mut t = [0u8; 80];
    rng.fill_bytes(&mut t[..]);

    // h = H*(T || M)
    let h = Self::hash_to_scalar(b"Nova_Ecdsa_Hash", &t[..], c.to_repr().as_mut());

    // R = [h]G
    let r = G::generator().mul(h);

    // s = h + c * sk
    let mut s = c;

    s.mul_assign(&self.0);
    s.add_assign(&h);

    Signature { r, s }
  }

  fn mul_bits<B: AsRef<[u64]>>(s: &G::Scalar, bits: BitIterator<B>) -> G::Scalar {
    let mut x = G::Scalar::zero();
    for bit in bits {
      x = x.double();

      if bit {
        x.add_assign(s)
      }
    }
    x
  }

  fn to_uniform(digest: &[u8]) -> G::Scalar {
    assert_eq!(digest.len(), 64);
    let mut bits: [u64; 8] = [0; 8];
    LittleEndian::read_u64_into(digest, &mut bits);
    Self::mul_bits(&G::Scalar::one(), BitIterator::new(bits))
  }

  pub fn to_uniform_32(digest: &[u8]) -> G::Scalar {
    assert_eq!(digest.len(), 32);
    let mut bits: [u64; 4] = [0; 4];
    LittleEndian::read_u64_into(digest, &mut bits);
    Self::mul_bits(&G::Scalar::one(), BitIterator::new(bits))
  }

  pub fn hash_to_scalar(persona: &[u8], a: &[u8], b: &[u8]) -> G::Scalar {
    let mut hasher = Sha3_512::new();
    hasher.input(persona);
    hasher.input(a);
    hasher.input(b);
    let digest = hasher.result();
    Self::to_uniform(digest.as_ref())
  }
}

impl<G> PublicKey<G>
where
  G: Group,
  G::Scalar: PrimeFieldBits,
{
  pub fn verify(&self, c: G::Scalar, signature: &Signature<G>) -> bool {
    let modulus = Self::modulus_as_scalar();
    let order_check_pk = self.0.mul(modulus);
    if !order_check_pk.eq(&G::identity()) {
      return false;
    }

    let order_check_r = signature.r.mul(modulus);
    if !order_check_r.eq(&G::identity()) {
      return false;
    }

    // 0 = [-s]G + R + [c]PK
    self
      .0
      .mul(c)
      .add(&signature.r)
      .add(G::generator().mul(signature.s).neg())
      .eq(&G::identity())
  }

  fn modulus_as_scalar() -> G::Scalar {
    let mut bits = G::Scalar::char_le_bits().to_bitvec();
    let mut acc = BigUint::new(Vec::<u32>::new());
    while let Some(b) = bits.pop() {
      acc <<= 1_i32;
      acc += b as u8;
    }
    let modulus = acc.to_str_radix(10);
    G::Scalar::from_str_vartime(&modulus).unwrap()
  }
}

#[derive(Debug)]
pub struct BitIterator<E> {
  t: E,
  n: usize,
}

impl<E: AsRef<[u64]>> BitIterator<E> {
  pub fn new(t: E) -> Self {
    let n = t.as_ref().len() * 64;

    BitIterator { t, n }
  }
}

impl<E: AsRef<[u64]>> Iterator for BitIterator<E> {
  type Item = bool;

  fn next(&mut self) -> Option<bool> {
    if self.n == 0 {
      None
    } else {
      self.n -= 1;
      let part = self.n / 64;
      let bit = self.n - (64 * part);

      Some(self.t.as_ref()[part] & (1 << bit) > 0)
    }
  }
}

// Synthesize a bit representation into circuit gadgets.
pub fn synthesize_bits<F: PrimeField, CS: ConstraintSystem<F>>(
  cs: &mut CS,
  bits: Option<Vec<bool>>,
) -> Result<Vec<AllocatedBit>, SynthesisError> {
  (0..F::NUM_BITS)
    .map(|i| {
      AllocatedBit::alloc(
        cs.namespace(|| format!("bit {i}")),
        Some(bits.as_ref().unwrap()[i as usize]),
      )
    })
    .collect::<Result<Vec<AllocatedBit>, SynthesisError>>()
}

pub fn verify_signature<G: NovaGroup, CS: ConstraintSystem<G::Base>>(
  cs: &mut CS,
  pk: AllocatedPoint<G>,
  r: AllocatedPoint<G>,
  s_bits: Vec<AllocatedBit>,
  c_bits: Vec<AllocatedBit>,
) -> Result<(), SynthesisError> {
  let g = AllocatedPoint::<G>::alloc(
    cs.namespace(|| "g"),
    Some((
      G::Base::from_str_vartime(
        "28948022309329048855892746252171976963363056481941647379679742748393362948096",
      )
      .unwrap(),
      G::Base::from_str_vartime("2").unwrap(),
      false,
    )),
  )
  .unwrap();

  cs.enforce(
    || "gx is vesta curve",
    |lc| lc + g.get_coordinates().0.get_variable(),
    |lc| lc + CS::one(),
    |lc| {
      lc + (
        G::Base::from_str_vartime(
          "28948022309329048855892746252171976963363056481941647379679742748393362948096",
        )
        .unwrap(),
        CS::one(),
      )
    },
  );

  cs.enforce(
    || "gy is vesta curve",
    |lc| lc + g.get_coordinates().1.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc + (G::Base::from_str_vartime("2").unwrap(), CS::one()),
  );

  let sg = g.scalar_mul(cs.namespace(|| "[s]G"), s_bits)?;
  let cpk = pk.scalar_mul(&mut cs.namespace(|| "[c]PK"), c_bits)?;
  let rcpk = cpk.add(&mut cs.namespace(|| "R + [c]PK"), &r)?;

  let (rcpk_x, rcpk_y, _) = rcpk.get_coordinates();
  let (sg_x, sg_y, _) = sg.get_coordinates();

  cs.enforce(
    || "sg_x == rcpk_x",
    |lc| lc + sg_x.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc + rcpk_x.get_variable(),
  );

  cs.enforce(
    || "sg_y == rcpk_y",
    |lc| lc + sg_y.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc + rcpk_y.get_variable(),
  );

  Ok(())
}

type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;

fn main() {
  let mut cs = TestConstraintSystem::<<G1 as Group>::Scalar>::new();
  assert!(cs.is_satisfied());
  assert_eq!(cs.num_constraints(), 0);

  let sk = SecretKey::<G2>::random(&mut OsRng);
  let pk = PublicKey::from_secret_key(&sk);

  // generate a random message to sign
  let c = <G2 as Group>::Scalar::random(&mut OsRng);

  // sign and verify
  let signature = sk.sign(c, &mut OsRng);
  let result = pk.verify(c, &signature);
  assert!(result);

  // prepare inputs to the circuit gadget
  let pk = {
    let pkxy = pk.0.to_affine().coordinates().unwrap();

    AllocatedPoint::<G2>::alloc(
      cs.namespace(|| "pub key"),
      Some((*pkxy.x(), *pkxy.y(), false)),
    )
    .unwrap()
  };
  let r = {
    let rxy = signature.r.to_affine().coordinates().unwrap();
    AllocatedPoint::alloc(cs.namespace(|| "r"), Some((*rxy.x(), *rxy.y(), false))).unwrap()
  };
  let s = {
    let s_bits = signature
      .s
      .to_le_bits()
      .iter()
      .map(|b| *b)
      .collect::<Vec<bool>>();

    synthesize_bits(&mut cs.namespace(|| "s bits"), Some(s_bits)).unwrap()
  };
  let c = {
    let c_bits = c.to_le_bits().iter().map(|b| *b).collect::<Vec<bool>>();

    synthesize_bits(&mut cs.namespace(|| "c bits"), Some(c_bits)).unwrap()
  };

  // Check the signature was signed by the correct sk using the pk
  verify_signature(&mut cs, pk, r, s, c).unwrap();

  assert!(cs.is_satisfied());
}
