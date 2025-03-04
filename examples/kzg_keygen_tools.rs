use std::{
  fs::{File, OpenOptions},
  io::{BufReader, BufWriter},
};

use halo2curves::bn256;
use nova_snark::{
  provider::{hyperkzg::CommitmentKey, id_of, read_ptau, write_ptau, Bn256EngineKZG, CommitmentKeyIO},
  traits::Engine,
};
use rand_core::OsRng;

type E = Bn256EngineKZG;

const KZG_KEY_DIR: &str = "/tmp/";

pub fn get_key_file_path(num_gens: usize) -> String {
  let group_id = id_of::<<Bn256EngineKZG as Engine>::GE>().unwrap();
  let base_dir = KZG_KEY_DIR.trim_end_matches("/");
  format!("{}/kzg_{}_{}.keys", base_dir, group_id, num_gens)
}

const LABEL: &[u8; 4] = b"test";

const MAX_NUM_GENS: usize = 1 << 21;

macro_rules! timeit {
  ($e:expr) => {{
    let start = std::time::Instant::now();
    let res = $e();
    let dur = start.elapsed();
    (res, dur)
  }};
}

fn keygen_save_large() {
  const BUFFER_SIZE: usize = 64 * 1024;

  type EG = Bn256EngineKZG;

  let path = get_key_file_path(MAX_NUM_GENS);

  if !CommitmentKey::<EG>::check_sanity_of_file(&path, MAX_NUM_GENS) {
    println!("Generating {} KZG keys ", MAX_NUM_GENS);

    let (ck, dur) = timeit!(|| { CommitmentKey::<E>::setup_from_rng(LABEL, MAX_NUM_GENS, OsRng) });

    println!("Generated {} keys in {:?}", MAX_NUM_GENS, dur);

    let file = OpenOptions::new()
      .write(true)
      .create(true)
      .truncate(true)
      .open(&path)
      .unwrap();
    let mut writer = BufWriter::with_capacity(BUFFER_SIZE, &file);

    let (_, dur) = timeit!(|| {
      ck.save_to(&mut writer).unwrap();
    });

    println!(
      "Saved {} keys to {} in {:?}, file size={}MB",
      MAX_NUM_GENS,
      &path,
      dur,
      file.metadata().unwrap().len() / 1024 / 1024
    );
  } else {
    println!("Key file already exists at {}", &path);
  }

  let (_, dur) = timeit!(|| {
    let file = OpenOptions::new().read(true).open(&path).unwrap();
    let mut reader = BufReader::new(file);
    CommitmentKey::<E>::load_from(&mut reader, MAX_NUM_GENS)
  });

  println!("Loaded {} keys from {} in {:?}", MAX_NUM_GENS, &path, dur);
}

fn main() {
  // keygen_save_large();
  let mut reader = BufReader::new(File::open("/tmp/ppot_0080_21.ptau").unwrap());
  let ((r1, r2), dur) = timeit!(|| {
    read_ptau::<bn256::G1Affine, bn256::G2Affine>(&mut reader, 200_0000, 20).unwrap()
  });

  dbg!(dur);

  let mut writer = BufWriter::new(File::create("8.ptau").unwrap());
  write_ptau(&mut writer, r1, r2, 22).unwrap();
  
  let mut reader = BufReader::new(File::open("8.ptau").unwrap());
    let ((r1, r2), dur) = timeit!(|| {
      read_ptau::<bn256::G1Affine, bn256::G2Affine>(&mut reader, 200_0000, 20).unwrap()
    });
  


}
