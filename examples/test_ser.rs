// Quick test to check serialization size of R1CSInstance
use nova_snark::provider::Bn256EngineKZG;
use nova_snark::traits::Engine;
use nova_snark::r1cs::R1CSInstance;
use halo2curves::group::Group;
use halo2curves::bn256::Fr;
use ff::Field;

type E = Bn256EngineKZG;

fn main() {
    // Create an R1CSInstance with 3 elements in X
    let comm_w = <E as Engine>::GE::generator();
    let x = vec![Fr::ONE, Fr::from(2), Fr::from(3)];
    
    let instance = R1CSInstance::<E>::new_unchecked(comm_w, x);
    
    // Serialize with bincode
    let config = bincode::config::standard()
        .with_big_endian()
        .with_fixed_int_encoding();
    let bytes = bincode::serde::encode_to_vec(&instance, config).unwrap();
    
    println!("Serialized R1CSInstance length: {} bytes", bytes.len());
    println!("Expected: 64 (G1) + 8 (len) + 3*32 (Fr) = 168 bytes");
    
    // Print structure
    println!("\nByte breakdown:");
    println!("Bytes 0-31 (should be G1.x): {:02x?}", &bytes[0..32]);
    println!("Bytes 32-63 (should be G1.y): {:02x?}", &bytes[32..64]);
    println!("Bytes 64-71 (should be len=3): {:02x?}", &bytes[64..72]);
    println!("Bytes 72-103 (should be X[0]): {:02x?}", &bytes[72..104]);
}
