# Nova: High-Speed Recursive Arguments from Folding Schemes
## Project Overview
Nova is a cutting-edge cryptographic library designed to realize [incrementally verifiable computation (IVC)](https://iacr.org/archive/tcc2008/49480001/49480001.pdf) without relying on succinct non-interactive arguments of knowledge (SNARKs). Instead, Nova introduces and employs folding schemes, a simpler and more efficient primitive that reduces the task of checking two instances in some relation to the task of checking a single instance. This approach results in improved efficiency, making Nova the fastest prover in the literature for IVC.

---
# Table of Contents

1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Applications](#applications)
4. [Library Details](#library-details)
5. [Commitment Schemes in Nova](#commitment-schemes-in-nova)
6. [SNARK Implementation for Compressing IVC Proofs](#snark-implementation-for-compressing-ivc-proofs)
7. [Front-End Integration](#front-end-integration)
8. [Testing](#testing)
9. [Acknowledgments](#acknowledgments)
10. [References](#references)
11. [Contributing](#contributing)
12. [Additional Guidelines](#additional-guidelines)
13. [Trademarks](#trademarks)


---
## Introduction
Nova is a library that leverages folding schemes to achieve efficient incrementally verifiable computation (IVC). Traditional approaches to IVC often rely on SNARKs, which can be impractical due to their complexity and overhead. Nova's innovative use of folding schemes offers a more efficient and scalable solution.

Nova's folding schemes are designed to handle NP relations, reducing the task of verifying two instances to a single instance. This results in a constant-sized verifier circuit and the fastest prover in the literature. Nova is implemented in Rust and is designed to be generic over a cycle of elliptic curves and a hash function.

By default, Nova enables the `asm` feature of the underlying library, boosting performance by up to 50%. If the library fails to build or run, you can pass `--no-default-features` to the cargo commands.

Explore the repository to find examples, documentation, and tools to get started with Nova. Whether you're working on verifiable delay functions, succinct blockchains, or verifiable state machines, Nova provides the efficiency and scalability you need.

---

## Key Features
1. Incrementally Verifiable Computation (IVC): Nova enables IVC, a cryptographic primitive that facilitates proofs for "long-running" sequential computations incrementally. For instance, Nova enables the following:
   - The prover takes as input a proof $\pi_i$ proving the first $i$ steps of its computation and then updates it to produce a proof $\pi_{i+1}$ proving the correct execution of the first $i+1$ steps.
   - Crucially, the prover's work to update the proof does not depend on the number of steps executed thus far.
   - The verifier's work to verify a proof does not grow with the number of steps in the computation.
3. Efficiency: Nova achieves the smallest recursion overhead in the literature, with the prover's work at each step dominated by two multiexponentiations of size $O(|F|)$.
4. No Trusted Setup: Nova does not require a trusted setup, making it more practical and secure for various applications.
5. Constant-Sized Verifier Circuit: The verifier circuit is constant-sized and its size is dominated by two group scalar multiplications, providing the smallest verifier circuit in the context of recursive proof composition.
6. Simplified and Efficient Recursive Proof System:

    Nova stands out for its simplicity and efficiency:
   - Minimal Verifier Circuit: The verifier circuit has a constant size, approximately 10,000 multiplication gates.
   - Novel Folding Scheme: Built on a novel folding scheme that reduces the task of verifying two NP statements to verifying one

---
## Applications
Nova's efficient and scalable IVC system can be applied to a wide variety of use cases, including:
- Rollups and succinct blockchains
- Verifiable delay functions (VDFs)
- Incrementally [verifiable state machines](https://eprint.iacr.org/2020/758.pdf)
- Proofs of (virtual) machine executions (e.g., EVM, RISC-V)
## Library Details

This repository provides **nova-snark**, a Rust implementation of Nova over elliptic curve cycles. Supported cycles:
1. **Pallas/Vesta**
2. **BN254/Grumpkin**
3. **secp/secq**

## Commitment Schemes in Nova
At its core, Nova relies on a commitment scheme for vectors.  
Compressing IVC proofs using Spartan involves interpreting commitments to vectors as commitments to multilinear polynomials and proving evaluations of committed polynomials.  

Our code implements two commitment schemes and evaluation arguments:  

- **Pedersen Commitments:** Utilizes an IPA-based evaluation argument and is supported on all three curve cycles.  
- **HyperKZG Commitments:** Utilizes an evaluation argument and is supported on curves with pairings, such as BN254.  

For more details on using HyperKZG, please see the test `test_ivc_nontrivial_with_compression`.  
The HyperKZG instantiation requires a universal trusted setup (the so-called "powers of tau").  
In the `setup` method in `src/provider/hyperkzg.rs`, one can load group elements produced in an existing KZG trusted setup (that was created for other proof systems based on univariate polynomials such as Plonk or variants), but the library does not currently do so (please see [this issue](https://github.com/microsoft/Nova/issues/270)).  


### SNARK Implementation for Compressing IVC Proofs 
We also implement a SNARK, based on  [Spartan](https://eprint.iacr.org/2019/550.pdf), to compress IVC proofs produced by Nova. There are two variants:
1. **Non-Preprocessing Variant:** This variant does not use any preprocessing.
2. **Preprocessing Variant:** This variant uses preprocessing of circuits to ensure that the verifier's runtime does not depend on the size of the step circuit.

For zero-knowledge proofs, IVC proofs are folded with random instances before compression as described in the [HyperNova paper](https://eprint.iacr.org/2023/573.pdf).

---

## Front-End Integration

A front-end is a tool that converts a high-level program into an intermediate representation (e.g., a circuit) that can be used to prove the execution of the program on concrete inputs. Nova supports three ways to write high-level programs in a form that can be proven.
1. **Native APIs**: The native APIs of Nova accept circuits expressed with Bellman-style circuits. For examples, see [minroot.rs](https://github.com/microsoft/Nova/blob/main/examples/minroot.rs) or [sha256.rs](https://github.com/microsoft/Nova/blob/main/benches/sha256.rs).
2. **Circom**: A DSL and a compiler that transforms high-level programs expressed in its language into a circuit. Middleware exists to convert the output of Circom into a form suitable for proving with Nova. See [Nova Scotia](https://github.com/nalinbhardwaj/Nova-Scotia) and [Circom Scotia](https://github.com/lurk-lab/circom-scotia). In the future, we will add examples in the Nova repository to demonstrate how to use these tools with Nova.

---

## Testing
To run tests (we recommend the release mode to drastically shorten run times):
```text
cargo test --release
```

To run an example:
```text
cargo run --release --example minroot
```
### Running Tests
Run all tests in release mode:
```bash
cargo test --release
```

## Acknowledgments
- **Research Contributors**: We would like to acknowledge the contributions of Abhiram Kothapalli, Srinath Setty, Ioanna Tzialla, and Wilson Nguyen for their foundational work on Nova and recursive zero-knowledge proofs.
- **Cryptographic Libraries**: Thanks to the cryptographic research community for their continuous improvements in commitment schemes, elliptic curve cryptography, and other cryptographic primitives used in Nova.
- **Open Source Community**: Special thanks to the open-source community for their contributions to the development of libraries and tools that made Nova possible, particularly the contributors to the Rust ecosystem and cryptographic libraries like `bellman` and `spartan`.
- **Microsoft**: The project is supported by the Microsoft open-source ecosystem, and we acknowledge their contribution through the adoption of their CLA (Contributor License Agreement) for this project.

---
## References
The following paper, which appeared at CRYPTO 2022, provides details of the Nova proof system and a proof of security:

[Nova: Recursive Zero-Knowledge Arguments from Folding Schemes](https://eprint.iacr.org/2021/370) \
Abhiram Kothapalli, Srinath Setty, and Ioanna Tzialla \
CRYPTO 2022

For efficiency, our implementation of the Nova proof system is instantiated over a cycle of elliptic curves. The following paper specifies that instantiation and provides a proof of security:

[Revisiting the Nova Proof System on a Cycle of Curves](https://eprint.iacr.org/2023/969) \
Wilson Nguyen, Dan Boneh, and Srinath Setty \
IACR ePrint 2023/969

The zero-knowledge property is achieved using an idea described in the following paper:

[HyperNova: Recursive arguments for customizable constraint systems](https://eprint.iacr.org/2023/573) \
Abhiram Kothapalli and Srinath Setty \
CRYPTO 2024

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Additional guidelines
This codebase implements a sophisticated cryptographic protocol, necessitating proficiency in cryptography, mathematics, security, and software engineering. Given the inherent complexity, the introduction of subtle bugs is a pervasive concern, rendering the acceptance of substantial contributions exceptionally challenging. Consequently, external contributors are kindly urged to submit incremental, easily reviewable pull requests (PRs) that encapsulate well-defined changes.

Our preference is to maintain code that is not only simple, but also easy to comprehend and maintain. This approach facilitates the auditing of code for correctness and security. To achieve this objective, we may prioritize code simplicity over minor performance enhancements, particularly when such improvements entail intricate, challenging-to-maintain code that disrupts abstractions.

In the event that you propose performance-related changes through a PR, we anticipate the inclusion of reproducible benchmarks demonstrating substantial speedups across a range of typical circuits. This rigorous benchmarking ensures that the proposed changes meaningfully enhance the performance of a diverse set of applications built upon Nova. Each performance enhancement will undergo a thorough, case-by-case evaluation to ensure alignment with our commitment to maintaining codebase simplicity.

Lastly, should you intend to submit a substantial PR, we kindly request that you initiate a GitHub issue outlining your planned changes, thereby soliciting feedback prior to committing substantial time to the implementation of said changes.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
