use bellperson::{
  gadgets::{boolean::AllocatedBit, num::AllocatedNum},
  ConstraintSystem, SynthesisError,
};
use ff::{PrimeField, PrimeFieldBits};
use nova_snark::{gadgets::ecc::AllocatedPoint, traits::circuit::StepCircuit};
use subtle::Choice;

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

// An ECDSA signature
#[derive(Clone, Debug)]
pub struct EcdsaSignature<Fb, Fs>
where
  Fb: PrimeField<Repr = [u8; 32]>,
  Fs: PrimeField<Repr = [u8; 32]> + PrimeFieldBits,
{
  pk: Coordinate<Fb>, // public key
  r: Coordinate<Fb>,  // (r, s) is the ECDSA signature
  s: Fs,
  c: Fs,             // hash of the message
  g: Coordinate<Fb>, // generator of the group; could be omitted if Nova's traits allow accessing the generator
}

impl<Fb, Fs> EcdsaSignature<Fb, Fs>
where
  Fb: PrimeField<Repr = [u8; 32]>,
  Fs: PrimeField<Repr = [u8; 32]> + PrimeFieldBits,
{
  pub fn new(pk: Coordinate<Fb>, r: Coordinate<Fb>, s: Fs, c: Fs, g: Coordinate<Fb>) -> Self {
    Self { pk, r, s, c, g }
  }
}

// An ECDSA signature proof that we will use on the primary curve
#[derive(Clone, Debug)]
pub struct EcdsaCircuit<F>
where
  F: PrimeField<Repr = [u8; 32]>,
{
  pub r: Coordinate<F>,
  pub g: Coordinate<F>,
  pub pk: Coordinate<F>,
  pub c: F,
  pub s: F,
  pub c_bits: Vec<Choice>,
  pub s_bits: Vec<Choice>,
}

impl<F> EcdsaCircuit<F>
where
  F: PrimeField<Repr = [u8; 32]>,
{
  // Creates a new [`EcdsaCircuit<Fb, Fs>`]. The base and scalar field elements from the curve
  // field used by the signature are converted to scalar field elements from the cyclic curve
  // field used by the circuit.
  pub fn new<Fb, Fs>(num_steps: usize, signatures: &[EcdsaSignature<Fb, Fs>]) -> (Vec<F>, Vec<Self>)
  where
    Fb: PrimeField<Repr = [u8; 32]>,
    Fs: PrimeField<Repr = [u8; 32]> + PrimeFieldBits,
  {
    let mut z0 = Vec::new();
    let mut circuits = Vec::new();
    for i in 0..num_steps {
      let signature = &signatures[i];
      let r = Coordinate::new(
        F::from_repr(signature.r.x.to_repr()).unwrap(),
        F::from_repr(signature.r.y.to_repr()).unwrap(),
      );

      let g = Coordinate::new(
        F::from_repr(signature.g.x.to_repr()).unwrap(),
        F::from_repr(signature.g.y.to_repr()).unwrap(),
      );

      let pk = Coordinate::new(
        F::from_repr(signature.pk.x.to_repr()).unwrap(),
        F::from_repr(signature.pk.y.to_repr()).unwrap(),
      );

      let c_bits = Self::to_le_bits(&signature.c);
      let s_bits = Self::to_le_bits(&signature.s);
      let c = F::from_repr(signature.c.to_repr()).unwrap();
      let s = F::from_repr(signature.s.to_repr()).unwrap();

      let circuit = EcdsaCircuit {
        r,
        g,
        pk,
        c,
        s,
        c_bits,
        s_bits,
      };
      circuits.push(circuit);

      if i == 0 {
        z0 = vec![r.x, r.y, g.x, g.y, pk.x, pk.y, c, s];
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

impl<F> StepCircuit<F> for EcdsaCircuit<F>
where
  F: PrimeField<Repr = [u8; 32]> + PrimeFieldBits,
{
  fn arity(&self) -> usize {
    8
  }

  // Prove knowledge of the sk used to generate the Ecdsa signature (R,s)
  // with public key PK and message commitment c.
  // [s]G == R + [c]PK
  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    _z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    let g = AllocatedPoint::alloc(
      cs.namespace(|| "G"),
      Some((self.g.x, self.g.y, self.g.is_infinity)),
    )?;
    let s_bits = Self::synthesize_bits(&mut cs.namespace(|| "s_bits"), &self.s_bits)?;
    let sg = g.scalar_mul(cs.namespace(|| "[s]G"), s_bits)?;
    let r = AllocatedPoint::alloc(
      cs.namespace(|| "R"),
      Some((self.r.x, self.r.y, self.r.is_infinity)),
    )?;
    let c_bits = Self::synthesize_bits(&mut cs.namespace(|| "c_bits"), &self.c_bits)?;
    let pk = AllocatedPoint::alloc(
      cs.namespace(|| "PK"),
      Some((self.pk.x, self.pk.y, self.pk.is_infinity)),
    )?;
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

    let rx = AllocatedNum::alloc(cs.namespace(|| "rx"), || Ok(self.r.x))?;
    let ry = AllocatedNum::alloc(cs.namespace(|| "ry"), || Ok(self.r.y))?;
    let gx = AllocatedNum::alloc(cs.namespace(|| "gx"), || Ok(self.g.x))?;
    let gy = AllocatedNum::alloc(cs.namespace(|| "gy"), || Ok(self.g.y))?;
    let pkx = AllocatedNum::alloc(cs.namespace(|| "pkx"), || Ok(self.pk.x))?;
    let pky = AllocatedNum::alloc(cs.namespace(|| "pky"), || Ok(self.pk.y))?;
    let c = AllocatedNum::alloc(cs.namespace(|| "c"), || Ok(self.c))?;
    let s = AllocatedNum::alloc(cs.namespace(|| "s"), || Ok(self.s))?;

    Ok(vec![rx, ry, gx, gy, pkx, pky, c, s])
  }

  fn output(&self, _z: &[F]) -> Vec<F> {
    vec![
      self.r.x, self.r.y, self.g.x, self.g.y, self.pk.x, self.pk.y, self.c, self.s,
    ]
  }
}
