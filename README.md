# Nova: Recursive SNARKs without trusted setup

Nova is a high-speed recursive SNARK (a SNARK is type cryptographic proof system that enables a prover to prove a mathematical statement to a verifier with a short proof and succinct verification, and a recursive SNARK enables producing proofs that prove statements about prior proofs). 

More precisely, Nova achieves [incrementally verifiable computation (IVC)](https://iacr.org/archive/tcc2008/49480001/49480001.pdf), a powerful cryptographic primitive that allows a prover to produce a proof of correct execution of a "long running" sequential computations in an incremental fashion. For example, IVC enables the following: The prover takes as input a proof $\pi_i$ proving the the first $i$ steps of its computation and then update it to produce a proof $\pi_{i+1}$ proving the correct execution of the first $i + 1$ steps. Crucially, the prover's work to update the proof does not depend on the number of steps executed thus far, and the verifier's work to verify a proof does not grow with the number of steps in the computation. IVC schemes including Nova have a wide variety of applications such as Rollups, verifiable delay functions (VDFs), succinct blockchains, incrementally verifiable versions of [verifiable state machines](https://eprint.iacr.org/2020/758.pdf), and, more generally, proofs of (virtual) machine executions (e.g., EVM, RISC-V). 

A distinctive aspect of Nova is that it is the simplest recursive proof system in the literature, yet it provides the fastest prover. Furthermore, it achieves the smallest verifier circuit (a key metric to minimize in this context): the circuit is constant-sized and its size is about 10,000 multiplication gates. Nova is constructed from a simple primitive called a *folding scheme*, a cryptographic primitive that reduces the task of checking two NP statements into the task of checking a single NP statement. 

## Tests and examples
This repository provides `nova-snark,` a Rust library implementation of Nova on a cycle of elliptic curves. The code currently supports Pallas/Vesta (i.e., Pasta curves) and BN254/Grumpkin elliptic curve cycles. One can use Nova with other elliptic curve cycles (e.g., secp/secq) by providing an implementation of Nova's traits for those curves (e.g., see `src/provider/mod.rs`).

We also implement a SNARK, based on [Spartan](https://eprint.iacr.org/2019/550.pdf), to compress IVC proofs produced by Nova.

To run tests (we recommend the release mode to drastically shorten run times):
```text
cargo test --release
```

To run example:
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

## Acknowledgments
See the contributors list [here](https://github.com/microsoft/Nova/graphs/contributors)

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

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
