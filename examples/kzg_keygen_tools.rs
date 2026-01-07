use std::{
  fs::OpenOptions,
  io::{BufReader, BufWriter},
};

use halo2curves::bn256;
use nova_snark::{
  provider::{
    check_sanity_of_ptau_file,
    hyperkzg::{CommitmentEngine, CommitmentKey},
    Bn256EngineKZG,
  },
  traits::commitment::CommitmentEngineTrait,
};
use rand_core::OsRng;

type E = Bn256EngineKZG;

const KZG_KEY_DIR: &str = "/tmp/";

pub fn get_key_file_path(num_gens: usize) -> String {
  let id = std::any::type_name::<E>().chars()
        .filter(|c| c.is_alphanumeric())  // Keep only alphanumeric characters
        .collect::<String>();

  let base_dir = KZG_KEY_DIR.trim_end_matches("/");
  format!("{base_dir}/kzg_{id}_{num_gens}.keys")
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

  let path = get_key_file_path(MAX_NUM_GENS);

  if check_sanity_of_ptau_file::<bn256::G1Affine>(&path, MAX_NUM_GENS + 1, 1).is_err() {
    println!("Generating {MAX_NUM_GENS} KZG keys ");

    let (ck, dur) = timeit!(|| { CommitmentKey::<E>::setup_from_rng(LABEL, MAX_NUM_GENS, OsRng) });

    println!("Generated {MAX_NUM_GENS} keys in {dur:?}");

    let file = OpenOptions::new()
      .write(true)
      .create(true)
      .truncate(true)
      .open(&path)
      .unwrap();
    let mut writer = BufWriter::with_capacity(BUFFER_SIZE, &file);

    let (_, dur) = timeit!(|| {
      CommitmentEngine::<E>::save_setup(&ck, &mut writer).unwrap();
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

  let (res, dur) = timeit!(|| {
    let file = OpenOptions::new().read(true).open(&path).unwrap();
    let mut reader = BufReader::new(file);
    CommitmentEngine::<E>::load_setup(&mut reader, LABEL, MAX_NUM_GENS)
  });

  assert!(res.is_ok());

  println!("Loaded {} keys from {} in {:?}", MAX_NUM_GENS, &path, dur);
}

fn main() {
  keygen_save_large();
}
