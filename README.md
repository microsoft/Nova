# Nova: Recursive SNARKs without trusted setup

Nova is a high-speed recursive SNARK (a SNARK is type cryptographic proof system that enables a prover to prove a mathematical statement to a verifier with a short proof and succinct verification, and a recursive SNARK enables producing proofs that prove statements about prior proofs). The details of Nova are described in our [paper](https://eprint.iacr.org/2021/370). Recursive SNARKs including Nova have a wide variety of applications such as constructions of verifiable delay functions (VDFs), succinct blockchains, and incrementally verifiable versions of [verifiable state machines](https://eprint.iacr.org/2020/758.pdf). A distinctive aspect of Nova is that it is the simplest recursive proof system in the literature. Furthermore, it achieves the smallest verifier circuit (a key metric to minimize in this context): the circuit is constant-sized and its size is dominated by two group scalar multiplications.

This repository provides `libnova,` a Rust library library implementation of Nova. The current release implements the core building blocks in Nova, and future releases will use cycles of elliptic curves to support recursive composition of proofs.

To run tests, run:
```rust
cargo test
```

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
