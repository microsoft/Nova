use ff::PrimeField;
use neptune::{poseidon::PoseidonConstants, Strength} ;
use generic_array::typenum;

///All Poseidon Constants that need in Nova:
#[derive(Clone)]
pub struct NovaPoseidonConstants <F>
where 
    F: PrimeField
{
        pub(crate) constants9: PoseidonConstants<F, typenum::U9>, //TODO: Change these to actual arities
        pub(crate) constants10: PoseidonConstants<F, typenum::U10>,
}

impl<F> NovaPoseidonConstants <F> 
where
    F: PrimeField
{
    ///Generate Poseidon constants for the arities that Nova uses
    pub fn new() -> Self {
        let constants9 = PoseidonConstants::<F, typenum::U9>::new_with_strength(Strength::Strengthened);
        let constants10 = PoseidonConstants::<F, typenum::U10>::new_with_strength(Strength::Strengthened);
        Self {
            constants9,
            constants10,
        }
    }    
}
