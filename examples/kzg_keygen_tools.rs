use std::{
  fs::OpenOptions,
  io::{BufReader, BufWriter},
};

use ff::Field;
use nova_snark::{
  provider::{
    hyperkzg::{name_of_engine, CommitmentKey},
    Bn256EngineKZG,
  },
  traits::Engine,
};
use rand_core::OsRng;

type E = Bn256EngineKZG;
type Fr = <E as Engine>::Scalar;

const KZG_KEY_DIR: &str = "/tmp/";

pub fn get_key_file_path<E: Engine>(num_gens: usize) -> String {
  let engine_name = name_of_engine::<E>();
  let base_dir = KZG_KEY_DIR.trim_end_matches("/");
  format!("{}/kzg_{}_{}.keys", base_dir, engine_name, num_gens)
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

fn compute_power_of_tau_benchmark() {
  let tau = Fr::random(OsRng);

  println!(
    "| {:<12} | {:<12} | {:<12} | {:<12} | {:<12} |",
    "num", "base_time", "base_pps", "par_time", "par_pps"
  );
  println!("|--------------|--------------|--------------|--------------|--------------|");

  for p in [2, 5, 12, 15, 18, 20, 21] {
    let n = 1 << p;

    let (res_base, dur_base) = timeit!(|| { CommitmentKey::<E>::compute_powers_serial(tau, n) });

    let (res_par, dur_par) = timeit!(|| { CommitmentKey::<E>::compute_powers_par(tau, n) });

    println!(
      "| {:>10} | {:>12?} | {:>12.2} | {:>12?} | {:>12.2} |",
      n,
      dur_base,
      n as f64 / dur_base.as_nanos() as f64 * 1e9,
      dur_par,
      n as f64 / dur_par.as_nanos() as f64 * 1e9
    );

    assert_eq!(
      res_base, res_par,
      "The results of serial and parallel computation should be the same"
    );
  }
}

fn key_gen_from_tau_benchmark() {
  let ps = [2, 5, 12, 15, 18, 20, 21];

  let n = 1 << ps.last().unwrap();
  let tau = Fr::random(OsRng);
  let powers_of_tau = CommitmentKey::<E>::compute_powers_par(tau, n);

  println!(
    "| {:<12} | {:<12} | {:<12} | {:<12} | {:<12} |",
    "num", "base_time", "base_pps", "par_time", "par_pps"
  );
  println!("|--------------|--------------|--------------|--------------|--------------|");

  for p in ps {
    let n = 1 << p;

    let (res_base, dur_base) =
      timeit!(|| { CommitmentKey::<E>::setup_from_tau_direct(LABEL, &powers_of_tau[..n]) });

    let (res_par, dur_par) =
      timeit!(|| { CommitmentKey::<E>::setup_from_tau_fixed_base_exp(LABEL, &powers_of_tau[..n]) });

    println!(
      "| {:>12} | {:>12?} | {:>12.2} | {:>12?} | {:>12.2} |",
      n,
      dur_base,
      n as f64 / dur_base.as_nanos() as f64 * 1e9,
      dur_par,
      n as f64 / dur_par.as_nanos() as f64 * 1e9
    );

    assert_eq!(res_base.h(), res_par.h(),);
    assert_eq!(res_base.tau_H(), res_par.tau_H(),);
    assert_eq!(res_base.ck(), res_par.ck(),);
  }
}

fn keygen_save_large() {
  const BUFFER_SIZE: usize = 64 * 1024;

  type EG = Bn256EngineKZG;

  let path = get_key_file_path::<EG>(MAX_NUM_GENS);
  if !CommitmentKey::<EG>::check_sanity_of_file(&path, MAX_NUM_GENS) {
    println!("Generating keys for {} KZG", MAX_NUM_GENS);

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
}

fn load_keys_benchmark() {
  let path = get_key_file_path::<E>(MAX_NUM_GENS);

  if !CommitmentKey::<E>::check_sanity_of_file(&path, MAX_NUM_GENS) {
    keygen_save_large();
  }

  println!("| {:<12} | {:<12} |", "size", "load_time");
  println!("|--------------|--------------|");

  for p in 0..21 {
    if 1 << p > MAX_NUM_GENS {
      break;
    }

    let n = 1 << p;

    let file = OpenOptions::new().read(true).open(&path).unwrap();
    let mut reader = BufReader::new(file);

    let (ck, dur) = timeit!(|| { CommitmentKey::<E>::load_from(&mut reader, n).unwrap() });

    println!("| {:<12} | {:<12?} |", n, dur);

    assert_eq!(ck.ck().len(), n);
  }
}

fn main() {
  let args: Vec<String> = std::env::args().collect();

  match String::as_str(&args[1]){
    "benchmark_powers_of_tau" => {
      compute_power_of_tau_benchmark();
    }
    "benchmark_key_gen_from_tau" => {
      key_gen_from_tau_benchmark();
    }
    "benchmark_load_keys" => {
      load_keys_benchmark();
    }
    "keygen_save" => {
      keygen_save_large();
    }
    _ => {
        println!("Usage: {} <benchmark_powers_of_tau|benchmark_key_gen_from_tau|benchmark_load_keys|keygen_save>", args[0]);
    }
  }
}
