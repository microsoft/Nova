# Nova: High-speed recursive arguments from folding schemes

Nova is a high-speed recursive SNARK (a SNARK is type cryptographic proof system that enables a prover to prove a mathematical statement to a verifier with a short proof and succinct verification, and a recursive SNARK enables producing proofs that prove statements about prior proofs). 

> [!IMPORTANT]
> This project accepts external contributions, but please read [this](#additional-guidelines) section before contributing.

More precisely, Nova achieves [incrementally verifiable computation (IVC)](https://iacr.org/archive/tcc2008/49480001/49480001.pdf), a powerful cryptographic primitive that allows a prover to produce a proof of correct execution of a "long running" sequential computations in an incremental fashion. For example, IVC enables the following: The prover takes as input a proof $\pi_i$ proving the first $i$ steps of its computation and then update it to produce a proof $\pi_{i+1}$ proving the correct execution of the first $i + 1$ steps. Crucially, the prover's work to update the proof does not depend on the number of steps executed thus far, and the verifier's work to verify a proof does not grow with the number of steps in the computation. IVC schemes including Nova have a wide variety of applications such as Rollups, verifiable delay functions (VDFs), succinct blockchains, incrementally verifiable versions of [verifiable state machines](https://eprint.iacr.org/2020/758.pdf), and, more generally, proofs of (virtual) machine executions (e.g., EVM, RISC-V). 

A distinctive aspect of Nova is that it is the simplest recursive proof system in the literature, yet it provides the fastest prover. Furthermore, it achieves the smallest verifier circuit (a key metric to minimize in this context): the circuit is constant-sized and its size is about 10,000 multiplication gates. Nova is constructed from a simple primitive called a *folding scheme*, a cryptographic primitive that reduces the task of checking two NP statements into the task of checking a single NP statement. 

## Details of the library
This repository provides `nova-snark,` a Rust library implementation of Nova over a cycle of elliptic curves. Our code supports three curve cycles: (1) Pallas/Vesta, (2) BN254/Grumpkin, and (3) secp/secq. 

At its core, Nova relies on a commitment scheme for vectors. Compressing IVC proofs using Spartan relies on interpreting commitments to vectors as commitments to multilinear polynomials and prove evaluations of committed polynomials. Our code implements two commitment schemes and evaluation arguments: 
1. Pedersen commitments with IPA-based evaluation argument (supported on all three curve cycles), and
2. HyperKZG commitments and evaluation argument (supported on curves with pairings e.g., BN254).
    
For more details on using  HyperKZG, please see the test `test_ivc_nontrivial_with_compression`. The HyperKZG instantiation requires a universal trusted setup (the so-called "powers of tau"). In the `setup` method in `src/provider/hyperkzg.rs`, one can load group elements produced in an existing KZG trusted setup (that was created for other proof systems based on univariate polynomials such as Plonk or variants), but the library does not currently do so (please see [this](https://github.com/microsoft/Nova/issues/270) issue). 

We also implement a SNARK, based on [Spartan](https://eprint.iacr.org/2019/550.pdf), to compress IVC proofs produced by Nova. There are two variants, one that does *not* use any preprocessing and another that uses preprocessing of circuits to ensure that the verifier's run time does not depend on the size of the step circuit.

## Supported front-ends
A front-end is a tool to take a high-level program and turn it into an intermediate representation (e.g., a circuit) that can be used to prove executions of the program on concrete inputs. There are three supported ways to write high-level programs in a form that can be proven with Nova.

1. bellpepper: The native APIs of Nova accept circuits expressed with bellpepper, a Rust toolkit to express circuits. See [minroot.rs](https://github.com/microsoft/Nova/blob/main/examples/minroot.rs) or [sha256.rs](https://github.com/microsoft/Nova/blob/main/benches/sha256.rs) for examples.

2. Circom: A DSL and a compiler to transform high-level program expressed in its language into a circuit. There exist middleware to turn output of circom into a form suitable for proving with Nova. See [Nova Scotia](https://github.com/nalinbhardwaj/Nova-Scotia) and [Circom Scotia](https://github.com/lurk-lab/circom-scotia). In the future, we will add examples in the Nova repository to use these tools with Nova.

3. [Lurk](https://github.com/lurk-lab/lurk-rs): A Lisp dialect and a universal circuit to execute programs expressed in Lurk. The universal circuit can be proven with Nova.

In the future, we plan to support [Noir](https://noir-lang.org/), a Rust-like DSL and a compiler to transform those programs into an IR. See [this](https://github.com/microsoft/Nova/issues/275) GitHub issue for details.

## Tests and examples
By default, we enable the `asm` feature of an underlying library (which boosts performance by up to 50\%). If the library fails to build or run, one can pass `--no-default-features` to `cargo` commands noted below.

To run tests (we recommend the release mode to drastically shorten run times):
```text
cargo test --release
```

To run an example:
```text
cargo run --release --example minroot
```

## References
The following paper, which appeared at CRYPTO 2022, provides details of the Nova proof system and a proof of security:

[Nova: Recursive Zero-Knowledge Arguments from Folding Schemes](https://eprint.iacr.org/2021/370) \
Abhiram Kothapalli, Srinath Setty, and Ioanna Tzialla \
CRYPTO 2022

For efficiency, our implementation of the Nova proof system is instantiated over a cycle of elliptic curves. The following paper specifies that instantiation and provides a proof of security:

[Revisiting the Nova Proof System on a Cycle of Curves](https://eprint.iacr.org/2023/969) \
Wilson Nguyen, Dan Boneh, and Srinath Setty \
IACR ePrint 2023/969

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

## Acknowledgments
See the contributors list [here](https://github.com/microsoft/Nova/graphs/contributors)