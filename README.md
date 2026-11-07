# Dynark: Making Groth16 Dynamic

Reference implementation of [Dynark: Making Groth16 Dynamic](https://eprint.iacr.org/2025/1897).

Given a Groth16 proof for a witness `z`, Dynark produces a proof for a witness
`z'` that differs from `z` in `d` entries using `O(d)` group operations, instead
of re-running the prover. Two schemes are implemented:

- **SemiDynark** updates relative to a fixed base witness, after an `O(n log n)`
  preprocessing of that witness.
- **FullyDynark** updates the current witness, and spreads the cost of
  re-preprocessing across updates.

Dynark keeps the Groth16 prover key. Keys and proofs are the types of the
unmodified [`ark-groth16`](https://crates.io/crates/ark-groth16) crate, and both
fresh and updated proofs are checked by its verifier.

The crate also implements two faster Groth16 provers from the paper. They
use the same keys and produce the same proofs as the standard prover:

- **FasterGroth16** computes the quotient with 4 DFTs over the domain instead
  of 6 over a coset, at the cost of a second size-`n` MSM. In our measurements
  this is no faster than the 6-DFT prover on dense witnesses (an MSM costs
  30-50 DFTs); it only gains when `A * z`, `B * z`, `C * z` are mostly zero.
- **SparseGroth16** takes time that depends only on the number of nonzero
  entries in the assignment and in `A * z` and `B * z`.

## Layout

| Module | Paper |
|---|---|
| `poly` | Circular convolution; sparse convolution; stepwise DFT |
| `decomposition` | The paper's quotient decomposition. An update changes the proof's quotient by two parts: a **projection term** and a **cross term**. `basis` holds the precomputed vectors both parts use. `projection` computes the projection term, and the per-witness data it needs: in one go, step by step for FullyDynark, or with the Feist-Khovratovich method as a baseline. `cross` computes the cross term. |
| `groth16` | Groth16's setup, prover and verifier (the paper's `G`, `P`, `V`), with provers taking 7 DFTs (as arkworks) and 6 (as snarkjs). Converts R1CS to polynomials slightly differently from arkworks; see below |
| `faster_groth16` | FasterGroth16 |
| `sparse_groth16` | SparseGroth16 |
| `semi_dynamic` | SemiDynark: preprocessing `K` and update `U` |
| `fully_dynamic` | FullyDynark: two checkpoints, with background rebuilds |
| `synthetic_r1cs` | The synthetic R1CS used in the evaluation |
| `circuit` | Running Dynark on `ark-relations` circuits |

Each module uses only the modules above it in this table.

Before proving, Groth16 turns the R1CS constraints into polynomials (a QAP).
Dynark does this slightly differently from `ark-groth16`'s
`LibsnarkReduction`: the evaluation domain is sized to the number of
constraints, and no extra rows are added for the public inputs. Keys from
`ark-groth16`'s own setup therefore can't be used with Dynark's prover; use
`groth16::setup` instead.

## Building and testing

```sh
cargo build --release
cargo test
```

The tests check fresh proofs on several real circuits and curves, and check
semi-dynamic and fully-dynamic updates on the synthetic workload. Tests are
built with optimizations (see `[profile.test]` in `Cargo.toml`).

## Examples and benchmarks

Two short examples show the API on a small circuit, in a few seconds:

```sh
cargo run --release --example semi_dynamic
cargo run --release --example fully_dynamic
```

The benchmarks are in `benches/`, one target per kind of measurement:
`setup`, `preprocessing` (ours against Feist-Khovratovich), `provers`,
`updates` and `cross_term`. For example:

```sh
cargo bench --bench updates -- --log-n 10..22:2 --log-d 0 --serial
```

Each run writes a CSV with one row per measured run, and a Markdown report
with the command, commit, machine and a summary, to `results/`. See
[`benches/README.md`](benches/README.md) for the options.

## Acknowledgement

The Groth16 setup and prover build on [arkworks' implementation of
Groth16](https://github.com/arkworks-rs/groth16).

## License

Licensed under either the [Apache License 2.0](LICENSE-APACHE) or the [MIT
license](LICENSE-MIT), at your option.
