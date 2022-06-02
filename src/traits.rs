//! This module defines various traits required by the users of the library to implement.
use bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
use core::{
  fmt::Debug,
  ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};
use ff::{PrimeField, PrimeFieldBits};
use merlin::Transcript;
use neptune::{circuit::poseidon_hash, poseidon::PoseidonConstants, Arity, Poseidon};
use num_bigint::BigInt;

/// Represents an element of a group
pub trait Group:
  Clone
  + Copy
  + Debug
  + Eq
  + Sized
  + GroupOps
  + GroupOpsOwned
  + ScalarMul<<Self as Group>::Scalar>
  + ScalarMulOwned<<Self as Group>::Scalar>
{
  /// A type representing an element of the base field of the group
  type Base: PrimeField + PrimeFieldBits;

  /// A type representing an element of the scalar field of the group
  type Scalar: PrimeField + PrimeFieldBits + ChallengeTrait;

  /// A type representing the compressed version of the group element
  type CompressedGroupElement: CompressedGroup<GroupElement = Self>;

  /// A type representing preprocessed group element
  type PreprocessedGroupElement;

  /// A type that represents a hash function that consumes elements
  /// from the base field and squeezes out elements of the scalar field
  type HashFunc: HashFuncTrait<Self::Base, Self::Scalar>;

  /// A method to compute a multiexponentation
  fn vartime_multiscalar_mul(
    scalars: &[Self::Scalar],
    bases: &[Self::PreprocessedGroupElement],
  ) -> Self;

  /// Compresses the group element
  fn compress(&self) -> Self::CompressedGroupElement;

  /// Produce a vector of group elements using a static label
  fn from_label(label: &'static [u8], n: usize) -> Vec<Self::PreprocessedGroupElement>;

  /// Returns the affine coordinates (x, y, infinty) for the point
  fn to_coordinates(&self) -> (Self::Base, Self::Base, bool);

  /// Returns the order of the group as a big integer
  fn get_order() -> BigInt;
}

/// Represents a compressed version of a group element
pub trait CompressedGroup: Clone + Copy + Debug + Eq + Sized + Send + Sync + 'static {
  /// A type that holds the decompressed version of the compressed group element
  type GroupElement: Group;

  /// Decompresses the compressed group element
  fn decompress(&self) -> Option<Self::GroupElement>;

  /// Returns a byte array representing the compressed group element
  fn as_bytes(&self) -> &[u8];
}

/// A helper trait to append different types to the transcript
pub trait AppendToTranscriptTrait {
  /// appends the value to the transcript under the provided label
  fn append_to_transcript(&self, label: &'static [u8], transcript: &mut Transcript);
}

/// A helper trait to absorb different objects in RO
pub trait AbsorbInROTrait<G: Group> {
  /// Absorbs the value in the provided RO
  fn absorb_in_ro(&self, ro: &mut G::HashFunc);
}

/// A helper trait to generate challenges using a transcript object
pub trait ChallengeTrait {
  /// Returns a Scalar representing the challenge using the transcript
  fn challenge(label: &'static [u8], transcript: &mut Transcript) -> Self;
}

/// A helper trait that defines the behavior of a hash function that we use as an RO
pub trait HashFuncTrait<Base, Scalar> {
  /// A type representing constants/parameters associated with the hash function
  type Constants: HashFuncConstantsTrait<Base> + Clone;

  /// Initializes the hash function
  fn new(constants: Self::Constants) -> Self;

  /// Adds a scalar to the internal state
  fn absorb(&mut self, e: Base);

  /// Returns a random challenge by hashing the internal state
  fn get_challenge(&self) -> Scalar;

  /// Returns a hash of the internal state
  fn get_hash(&self) -> Scalar;
}

/// A helper trait that defines the constants associated with a hash function
pub trait HashFuncConstantsTrait<Base> {
  /// produces constants/parameters associated with the hash function
  fn new() -> Self;
}

/// A helper trait for types with a group operation.
pub trait GroupOps<Rhs = Self, Output = Self>:
  Add<Rhs, Output = Output> + Sub<Rhs, Output = Output> + AddAssign<Rhs> + SubAssign<Rhs>
{
}

impl<T, Rhs, Output> GroupOps<Rhs, Output> for T where
  T: Add<Rhs, Output = Output> + Sub<Rhs, Output = Output> + AddAssign<Rhs> + SubAssign<Rhs>
{
}

/// A helper trait for references with a group operation.
pub trait GroupOpsOwned<Rhs = Self, Output = Self>: for<'r> GroupOps<&'r Rhs, Output> {}
impl<T, Rhs, Output> GroupOpsOwned<Rhs, Output> for T where T: for<'r> GroupOps<&'r Rhs, Output> {}

/// A helper trait for types implementing group scalar multiplication.
pub trait ScalarMul<Rhs, Output = Self>: Mul<Rhs, Output = Output> + MulAssign<Rhs> {}

impl<T, Rhs, Output> ScalarMul<Rhs, Output> for T where T: Mul<Rhs, Output = Output> + MulAssign<Rhs>
{}

/// A helper trait for references implementing group scalar multiplication.
pub trait ScalarMulOwned<Rhs, Output = Self>: for<'r> ScalarMul<&'r Rhs, Output> {}
impl<T, Rhs, Output> ScalarMulOwned<Rhs, Output> for T where T: for<'r> ScalarMul<&'r Rhs, Output> {}

/// A helper trait for synthesizing a step of the incremental computation (i.e., circuit for F)
pub trait StepCircuit<F: PrimeField, A: Arity<F>>: Sized {
  /// Sythesize the circuit for a unary computation step and return variable
  /// that corresponds to the output of the step z_{i+1}
  /// An implementation must be provided if F is unary, and should not be otherwise.
  fn synthesize_step<CS: ConstraintSystem<F>>(
    &self,
    _cs: &mut CS,
    _z: AllocatedNum<F>,
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    unimplemented!();
  }

  /// Sythesize the circuit for a computation step and return variable
  /// that corresponds to the output of the step z_{i+1}
  /// The default implementation wraps a call to `synthesize_step_inner`, hashing input and output.
  fn synthesize_step_outer<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: AllocatedNum<F>,
    p: Option<&PoseidonConstants<F, A>>,
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    if p.is_none() {
      return self.synthesize_step(cs, z);
    }
    match self.io() {
      IO::Val(_) => unreachable!(),
      IO::Vals(vals, _) => {
        let mut z_vec = Vec::with_capacity(vals.len());

        for (i, v) in vals.iter().enumerate() {
          let allocated = AllocatedNum::alloc(cs.namespace(|| format!("z[{}]", i)), || Ok(*v))?;
          z_vec.push(allocated);
        }

        let hash = poseidon_hash(
          &mut cs.namespace(|| "hash"),
          z_vec.clone(),
          p.expect("PoseidonConstants missing"),
        )?;

        cs.enforce(
          || "hash = z",
          |lc| lc + z.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc + hash.get_variable(),
        );

        let inner_output = self.synthesize_step_inner(&mut cs.namespace(|| "inner"), z_vec)?;

        let output_hash = poseidon_hash(
          &mut cs.namespace(|| "output"),
          inner_output,
          p.expect("PoseidonConstants missing"),
        )?;

        Ok(output_hash)
      }
      IO::Blank => {
        let arity = A::to_usize();
        let mut z_vec = Vec::with_capacity(arity);

        for i in 0..arity {
          let allocated = AllocatedNum::alloc(cs.namespace(|| format!("z[{}]", i)), || {
            Err(SynthesisError::AssignmentMissing)
          })?;
          z_vec.push(allocated);
        }

        let hash = poseidon_hash(
          &mut cs.namespace(|| "hash"),
          z_vec.clone(),
          p.expect("PoseidonConstants missing"),
        )?;

        cs.enforce(
          || "hash = z",
          |lc| lc + z.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc + hash.get_variable(),
        );

        let inner_output = self.synthesize_step_inner(&mut cs.namespace(|| "inner"), z_vec)?;

        let output_hash = poseidon_hash(
          &mut cs.namespace(|| "output"),
          inner_output,
          p.expect("PoseidonConstants missing"),
        )?;

        Ok(output_hash)
      }
    }
  }

  /// Synthesize the circuit for a computation step with multiple inputs and outputs.
  /// This method must be implemented if F is a non-unary function.
  fn synthesize_step_inner<CS: ConstraintSystem<F>>(
    &self,
    _cs: &mut CS,
    _z_vec: Vec<AllocatedNum<F>>,
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    unimplemented!()
  }

  /// Return the IO value
  fn io<'a>(&'a self) -> &'a IO<F, A> {
    unimplemented!()
  }
}

/// Enum for holding the computation F's input/output value(s).
#[derive(Clone, Debug)]
pub enum IO<'a, F: PrimeField, A: Arity<F>> {
  /// Unary input/output value
  Val(F),
  /// Non-unary input/output values
  Vals(Vec<F>, &'a PoseidonConstants<F, A>),
  /// Placeholder for constructing R1CS shape
  Blank,
}

impl<'a, F: PrimeField, A: Arity<F>> IO<'a, F, A> {
  /// Return a single F corresponding to this `IO`.
  pub fn output(&self) -> F {
    match self {
      Self::Val(val) => *val,
      Self::Vals(vals, p) => {
        let mut hasher = Poseidon::<F, A>::new_with_preimage(vals, p);
        hasher.hash()
      }
      Self::Blank => unreachable!(),
    }
  }

  /// Return a new `IO::Vals`, sharing this `IO`'s poseidon constants
  pub fn new_vals(&self, vals: Vec<F>) -> Self {
    match self {
      IO::Val(_) => unreachable!(),
      IO::Vals(_, p) => IO::Vals(vals, p),
      IO::Blank => unreachable!(),
    }
  }
}

/// A helper trait for computing a step of the incremental computation (i.e., F itself)
pub trait StepCompute<'a, F: PrimeField, A: Arity<F>>: Sized {
  /// Compute F for a unary computation, returning a new circuit and output
  fn compute(&self, _z: &F) -> Option<(Self, F)> {
    unimplemented!();
  }

  /// Compute F, delegating to `compute` or `compute_inner` depending on whether F is unary
  fn compute_io(&self, z: &IO<'a, F, A>) -> Option<(Self, IO<'a, F, A>)> {
    match z {
      IO::Val(val) => self
        .compute(val)
        .map(|(new, new_val)| (new, IO::Val(new_val))),
      IO::Vals(vals, p) => {
        assert!(
          A::to_usize() != 1,
          "Unary step functions must use IO::Val to avoid superfluous hashing."
        );
        self
          .compute_inner(vals, p)
          .map(|(new, new_vals)| (new, z.new_vals(new_vals)))
      }
      IO::Blank => unreachable!(),
    }
  }

  /// Compute F for a non-unary computation, returning a new circuit and output
  /// This method must be implemented for non-unary F
  fn compute_inner(&self, _z: &[F], _p: &'a PoseidonConstants<F, A>) -> Option<(Self, Vec<F>)> {
    unimplemented!();
  }
}

impl<F: PrimeField> AppendToTranscriptTrait for F {
  fn append_to_transcript(&self, label: &'static [u8], transcript: &mut Transcript) {
    transcript.append_message(label, self.to_repr().as_ref());
  }
}

impl<F: PrimeField> AppendToTranscriptTrait for [F] {
  fn append_to_transcript(&self, label: &'static [u8], transcript: &mut Transcript) {
    for s in self {
      s.append_to_transcript(label, transcript);
    }
  }
}
