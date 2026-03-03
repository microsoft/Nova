use std::env;

fn main() {
  #[cfg(feature = "sppark")]
  build_sppark_msm();
}

#[cfg(feature = "sppark")]
fn build_sppark_msm() {
  use std::path::PathBuf;

  // Find nvcc: NVCC env > PATH > /usr/local/cuda/bin/nvcc
  let nvcc = env::var("NVCC")
    .ok()
    .and_then(|v| which::which(v).ok())
    .or_else(|| which::which("nvcc").ok())
    .or_else(|| {
      let p = PathBuf::from("/usr/local/cuda/bin/nvcc");
      p.exists().then_some(p)
    })
    .expect(
      "sppark feature requires nvcc. Install the CUDA toolkit and ensure nvcc is in PATH, \
             or set NVCC=/path/to/nvcc.",
    );

  if let Some(cuda_dir) = nvcc.parent().and_then(|bin| bin.parent()) {
    let lib64 = cuda_dir.join("lib64");
    if lib64.exists() {
      println!("cargo:rustc-link-search=native={}", lib64.display());
    }
  }

  env::set_var("NVCC", &nvcc);

  // sppark's build.rs emits DEP_SPPARK_ROOT pointing to its headers directory
  let sppark_root =
    PathBuf::from(env::var("DEP_SPPARK_ROOT").expect(
      "DEP_SPPARK_ROOT not set. Ensure sppark dependency does not use the 'build' feature.",
    ));

  let mut build = cc::Build::new();
  build.cuda(true);
  build.flag("-t0");

  // GPU architecture: CUDA_ARCH env var (e.g. "sm_80") or multi-arch default
  if let Ok(arch) = env::var("CUDA_ARCH") {
    build.flag(&format!("-arch={arch}"));
  } else {
    for (compute, sm) in [
      ("compute_70", "sm_70"), // V100
      ("compute_80", "sm_80"), // A100, 3090
      ("compute_89", "sm_89"), // 4090
      ("compute_90", "sm_90"), // H100
    ] {
      build
        .flag("-gencode")
        .flag(&format!("arch={compute},code={sm}"));
    }
    // PTX fallback for future GPUs
    build
      .flag("-gencode")
      .flag("arch=compute_90,code=compute_90");
  }

  #[cfg(not(target_env = "msvc"))]
  build.flag("-Xcompiler").flag("-Wno-unused-function");
  build.define("TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE", None);
  build.define("FEATURE_BN254", None);
  #[cfg(target_arch = "x86_64")]
  build.define("__ADX__", None);

  // Include paths from dependency build scripts
  build.include(&sppark_root);
  if let Some(blst_src) = env::var_os("DEP_BLST_C_SRC") {
    build.include(blst_src);
  }
  if let Some(semolina_inc) = env::var_os("DEP_SEMOLINA_C_INCLUDE") {
    build.include(semolina_inc);
  }
  build.include("gpu");

  build.file("gpu/sppark_msm.cu");

  // DEP_SPPARK_TARGET is set only when sppark's build.rs successfully compiled
  // all_gpus.cpp. If it didn't (e.g. nvcc not in PATH), we compile it ourselves.
  if env::var("DEP_SPPARK_TARGET").is_ok() {
    // sppark compiled all_gpus.cpp; ensure its symbols are available to our code
    println!("cargo:rustc-link-lib=static:+whole-archive=sppark_cuda");
  } else {
    build.file(sppark_root.join("util/all_gpus.cpp"));
  }

  build.compile("nova_sppark_msm");

  // Our CUDA code calls blst host-side field arithmetic. Force whole-archive
  // so blst symbols are available regardless of link order.
  println!("cargo:rustc-link-lib=static:+whole-archive=blst");
  println!("cargo:rustc-link-lib=cudart");
  println!("cargo:rerun-if-changed=gpu/sppark_msm.cu");
  println!("cargo:rerun-if-changed=gpu/msm_parallel.cuh");
  println!("cargo:rerun-if-env-changed=NVCC");
  println!("cargo:rerun-if-env-changed=CUDA_ARCH");
}
