use bellperson::{Circuit, ConstraintSystem, SynthesisError, gadgets::num::AllocatedNum};
use ff::{PrimeFieldBits};
use generic_array::typenum;
use super::r1cs::RelaxedR1CSInstance;
use super::traits::{PrimeField, Group, CompressedGroup};
use super::gadgets::{utils::le_bits_to_num, ro::PoseidonRO};

///Circuit that encodes only the folding verifier
pub struct VerificationCircuit<G1, G2>
where 
    G1: Group,
    G2: Group,
{
    h1: G1::Scalar, //H(u2, i, z0, zi) where u2 is the running instance for the other circuit
    u2: RelaxedR1CSInstance<G2>,
    i: G1::Scalar,
    z0: G1::Scalar,
    zi: G1::Scalar,
    h2: G2::Scalar, //Hash of the running relaxed r1cs instance of this circuit
}

impl<G1, G2> VerificationCircuit<G1, G2>
where
    G1: Group,
    G2: Group,
{
    ///Create a new verification circuit for the input relaxed r1cs instances
    pub fn new(
        h1: G1::Scalar, 
        u2: RelaxedR1CSInstance<G2>,
        i: G1::Scalar,
        z0: G1::Scalar,
        zi: G1::Scalar,
        h2: G2::Scalar
    ) -> Self {
        Self {
            h1,
            u2,
            i,
            z0,
            zi,
            h2,
        }
    }
}

impl<G1, G2> Circuit<<G1 as Group>::Scalar> for VerificationCircuit<G1, G2>
where 
    G1: Group,
    G2: Group,
    <G1 as Group>::Scalar: PrimeField + PrimeFieldBits,
    <G2 as Group>::Scalar: PrimeField + PrimeFieldBits,
{
    fn synthesize<CS: ConstraintSystem<<G1 as Group>::Scalar>>(
        self,
        cs: &mut CS,
    ) -> Result<(), SynthesisError> {
        
        //TODO: For now assume that |G2| < |G1| so we don't need to do anything         
        
        /***********************************************************************/
        //This circuit does not modify h2 but it outputs it. 
        //Allocate it and output it. 
        /***********************************************************************/
        
        let h2_bytes = self.h2.as_bytes();
        let h2_encoded = G1::Scalar::from_bytes_mod_order_wide(&h2_bytes).unwrap();
        let h2 = AllocatedNum::alloc(cs.namespace(|| "h2"), || Ok(h2_encoded))?;
        let _ = h2.inputize(cs.namespace(|| "Output 1"))?;
        
        /***********************************************************************/
        //Check that H = Hash(U)
        /***********************************************************************/
       
        let h1_bytes = self.h1.as_bytes();
        let h1_encoded = G1::Scalar::from_bytes_mod_order_wide(&h1_bytes).unwrap();
        let h1 = AllocatedNum::alloc(cs.namespace(|| "h1"), || Ok(h1_encoded))?;

        //TODO: Change this to U7
        let mut hasher: PoseidonRO<G1::Scalar, typenum::U6>= PoseidonRO::new();

        //W_r and E_r are both G2::points so represent them natively as G1::points
        let W_r = AllocatedNum::alloc(
            cs.namespace(|| "W_r"), 
            || Ok(G1::Scalar::from_bytes_mod_order_wide(&self.u2.comm_W.compress().comm.as_bytes()).unwrap())
        )?; 
        hasher.absorb(W_r.clone());

        let E_r = AllocatedNum::alloc(
            cs.namespace(|| "E_r"), 
            || Ok(G1::Scalar::from_bytes_mod_order_wide(&self.u2.comm_E.compress().comm.as_bytes()).unwrap())
        )?; 
        hasher.absorb(E_r.clone());
        
        //TODO: Add X_r
        //let X_r = self.u2.X;
        
        let u_r = AllocatedNum::alloc(
            cs.namespace(|| "u_r"), 
            || Ok(G1::Scalar::from_bytes_mod_order_wide(&self.u2.u.as_bytes()).unwrap())
        )?;
        hasher.absorb(u_r.clone());

        let i = AllocatedNum::alloc(
            cs.namespace(|| "i"), 
            || Ok(G1::Scalar::from_bytes_mod_order_wide(&self.i.as_bytes()).unwrap())
        )?;
        hasher.absorb(i.clone());

        let z_0 = AllocatedNum::alloc(
            cs.namespace(|| "z0"), 
            || Ok(G1::Scalar::from_bytes_mod_order_wide(&self.z0.as_bytes()).unwrap())
        )?;
        hasher.absorb(z_0.clone());

        let z_i = AllocatedNum::alloc(
            cs.namespace(|| "zi"), 
            || Ok(G1::Scalar::from_bytes_mod_order_wide(&self.zi.as_bytes()).unwrap())
        )?;
        hasher.absorb(z_i.clone());

        let hash_bits = hasher.get_challenge(cs.namespace(|| "Input hash"))?;
        let hash = le_bits_to_num(cs.namespace(|| "bits to hash"), hash_bits)?;
        
        //make sure that hash == h
        cs.enforce(
            || "check h1", 
            |lc| lc,
            |lc| lc,
            |lc| lc + h1.get_variable() - hash.get_variable(),
        );
        Ok(())
    }
}
