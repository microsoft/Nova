use std::env;

fn main() {
    #[cfg(feature = "sppark")]
    build_sppark_msm();
}

#[cfg(feature = "sppark")]
fn build_sppark_msm() {
    use std::path::PathBuf;

    // Find nvcc: check NVCC env, then PATH, then common install locations
    let nvcc_path = env::var("NVCC")
        .ok()
        .and_then(|v| which::which(v).ok())
        .or_else(|| which::which("nvcc").ok())
        .or_else(|| {
            let p = PathBuf::from("/usr/local/cuda/bin/nvcc");
            p.exists().then_some(p)
        });

    let nvcc_path = nvcc_path.expect(
        "sppark feature requires CUDA. Install the CUDA toolkit, add nvcc to PATH, or set NVCC.",
    );

    // Add CUDA lib dir to link search path
    if let Some(cuda_dir) = nvcc_path.parent().and_then(|bin| bin.parent()) {
        let lib64 = cuda_dir.join("lib64");
        if lib64.exists() {
            println!("cargo:rustc-link-search=native={}", lib64.display());
        }
    }

    // Ensure cc crate can find nvcc
    env::set_var("NVCC", &nvcc_path);

    // Find sppark headers: DEP_SPPARK_ROOT (if set) or locate via cargo registry
    let sppark_root = env::var_os("DEP_SPPARK_ROOT")
        .map(PathBuf::from)
        .or_else(|| {
            // With sppark's "build" feature, DEP_SPPARK_ROOT isn't set.
            // Find the sppark crate's header directory via cargo metadata.
            let out_dir = PathBuf::from(env::var_os("OUT_DIR")?);
            // Walk up from OUT_DIR to find the registry source
            for ancestor in out_dir.ancestors() {
                let candidate = ancestor.join("sppark");
                if candidate.join("msm/pippenger.cuh").exists() {
                    return Some(candidate);
                }
            }
            // Search common registry paths
            if let Ok(home) = env::var("CARGO_HOME").or_else(|_| env::var("HOME").map(|h| format!("{}/.cargo", h))) {
                let registry = PathBuf::from(home).join("registry/src");
                if let Ok(entries) = std::fs::read_dir(&registry) {
                    for entry in entries.flatten() {
                        let glob = entry.path();
                        if let Ok(sub) = std::fs::read_dir(&glob) {
                            for pkg in sub.flatten() {
                                let name = pkg.file_name().to_string_lossy().to_string();
                                if name.starts_with("sppark-") {
                                    let candidate = pkg.path().join("sppark");
                                    if candidate.join("msm/pippenger.cuh").exists() {
                                        return Some(candidate);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            None
        })
        .expect("Cannot find sppark headers. Ensure the sppark crate is installed.");

    // Compile all_gpus.cpp (GPU context management, required by msm_t)
    let all_gpus = sppark_root.join("util/all_gpus.cpp");

    let mut nvcc = cc::Build::new();
    nvcc.cuda(true);
    nvcc.flag("-arch=sm_80");
    nvcc.flag("-gencode")
        .flag("arch=compute_70,code=sm_70");
    nvcc.flag("-t0");
    #[cfg(not(target_env = "msvc"))]
    nvcc.flag("-Xcompiler").flag("-Wno-unused-function");
    nvcc.define("TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE", None);
    nvcc.define("FEATURE_BN254", None);
    // blst host-side field arithmetic uses ADX variants on x86-64
    #[cfg(target_arch = "x86_64")]
    nvcc.define("__ADX__", None);

    nvcc.include(&sppark_root);
    if let Some(include) = env::var_os("DEP_BLST_C_SRC") {
        nvcc.include(include);
    }
    if let Some(include) = env::var_os("DEP_SEMOLINA_C_INCLUDE") {
        nvcc.include(include);
    }
    nvcc.include("gpu");

    nvcc.file("gpu/sppark_msm.cu")
        .file(&all_gpus)
        .compile("nova_sppark_msm");

    // Our CUDA code references blst field arithmetic on the host side.
    // Force blst to be fully included to resolve cross-library dependencies.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    if let Some(build_dir) = out_dir.ancestors().nth(2) {
        for entry in std::fs::read_dir(build_dir).into_iter().flatten().flatten() {
            if entry.file_name().to_string_lossy().starts_with("blst-") {
                let blst_lib = entry.path().join("out/libblst.a");
                if blst_lib.exists() {
                    println!(
                        "cargo:rustc-link-arg=-Wl,--whole-archive,{},--no-whole-archive",
                        blst_lib.display()
                    );
                    break;
                }
            }
        }
    }

    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rerun-if-changed=gpu/sppark_msm.cu");
    println!("cargo:rerun-if-changed=gpu/msm_parallel.cuh");
}
