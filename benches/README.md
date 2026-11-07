# Benchmarks

Each benchmark is a separate target, run with `cargo bench --bench <name>`;
options go after `--`. For example:

```sh
cargo bench --bench updates -- --log-n 10..22:2 --log-d 0 --serial
```

## Output

Every run writes two files to `results/` (or `--out-dir`), named after the
benchmark and its start time in UTC:

- `<name>-<time>.csv`: one row per measured run, with the parameters as
  columns, then `run` and `seconds`. Build tables and figures from this file.
- `<name>-<time>.md`: the command, commit, machine and thread count, and a
  summary table with the number of runs, the median and the mean. The same
  summary is printed to the terminal.

A new run never overwrites an earlier one.

## Options

Every benchmark accepts:

| Option | Meaning | Default |
|---|---|---|
| `--log-n` | domain sizes, as powers of two | depends on the benchmark |
| `--repeat` | runs per measurement | 3 |
| `--threads` | size of the thread pool | all cores |
| `--out-dir` | where to write the results | `results` |

Lists can be written as `10,12,14`, as a range `10..14`, or as a range with a
step `10..22:2`. Ranges include both ends.

All benchmarks use the synthetic R1CS of the paper's evaluation, with as many
variables as constraints and an instance of just the constant one.

## The benchmarks

### `setup`

The Groth16 setup, and deriving the updating key from the prover key.

Columns: `log_n`, `phase` (`groth16_setup` or `updating_key`). Default
`--log-n 12,14,16`.

### `preprocessing`

Preprocessing `K`, the projection data of a witness: ours (circular
convolutions) against the Feist-Khovratovich baseline (linear convolutions).
Both are timed in two phases:

- `one_time`: depends only on the circuit. Ours computes the quotient basis
  (`ell_bar`, `diag`, the DFTs of `w` and `w_prime`); FK computes the DFT of
  the zero-padded `h_query`.
- `per_witness`: the Lagrange coefficients of `A * z` and `B * z`, and their
  projection data.

The two methods are checked to give the same result.

Columns: `method` (`ours` or `fk`), `phase`, `log_n`, `threads`. Default
`--log-n 12,14,16`.

| Option | Meaning |
|---|---|
| `--methods ours,fk` | which methods to run |
| `--serial-max-log-n M` | also measure single-threaded for `log_n <= M` |

### `provers`

The Groth16 provers, which all produce the same proof:

- `groth16_6dft`: 6 DFTs, committing to the quotient's evaluations on a coset,
  as snarkjs does. This is the baseline.
- `groth16_7dft`: 7 DFTs, as arkworks does.
- `faster_groth16`: FasterGroth16, 4 DFTs over the domain.
- `sparse_groth16`: SparseGroth16, time depends only on the number of nonzero
  entries.

Each is timed in two phases: `one_time`, the per-circuit precomputation it
needs beyond the prover key (`CosetBasis` for `groth16_6dft`,
`FasterGroth16Key` and `SparseGroth16Key` for the other two; `groth16_7dft`
needs none), and `prove`,
one proof.

Two workloads: `dense`, the synthetic R1CS, and `sparse`, where only
`--nonzero` of the `n / 4` inputs are nonzero.

Columns: `workload`, `prover`, `phase`, `log_n`, `threads`. Default `--log-n 12,14,16`.

| Option | Meaning |
|---|---|
| `--workloads dense,sparse` | which workloads to run |
| `--provers groth16_6dft,groth16_7dft,faster_groth16,sparse_groth16` | which provers to run |
| `--nonzero K` | nonzero inputs in the sparse workload (default 64) |
| `--serial` | also measure single-threaded |

### `updates`

Updating a proof after `d` entries of the witness change, against proving from
scratch.

- `groth16`: proving the whole witness from scratch with the 6-DFT prover
  (`prove`), and verifying it (`verify`). Its one-time `CosetBasis` is timed
  in `provers`, not here.
- `semi`: one SemiDynark update of `d` entries from the base witness,
  `--repeat` times.
- `fully`: consecutive FullyDynark updates of `d` entries. By default there are
  `max(repeat, 2^((log n + 6) / 2) / d)` of them, about one rebuild cycle, so
  their mean includes the cost of rebuilding. Use the mean, not the median,
  for `fully`. Each update is recorded as `update`, and split into `rebuild`
  (the background rebuild steps) and `apply` (the rest). Needs even `log n`;
  odd sizes are skipped.

Every proof is checked to verify.

Columns: `scheme`, `log_n`, `d` (empty for `groth16`), `threads`, `phase`.
Default `--log-n 10,12,14,16`.

| Option | Meaning |
|---|---|
| `--log-d` | update sizes `d`, as powers of two (default `0`); sizes with `d >= n / 2` are skipped |
| `--schemes groth16,semi,fully` | which schemes to run |
| `--fully-runs K` | number of `fully` updates per `d` |
| `--half-budget` | spend `(d + 1) / 2` rebuild steps per `fully` update instead of `d` |
| `--serial` | also measure single-threaded: every `groth16` and `semi` run is repeated on one thread, and one random `fully` update in every `--serial-every` (default 32) runs on one thread |

### `cross_term`

The cross term of an update, with its inner sums computed `pairwise`, by
`sparse_conv`, or by whichever `auto` picks. Two shapes: `semi` changes the
same `d` rows of `A * z` and `B * z`; `fully` crosses `acc` accumulated rows
with `d` new ones.

Columns: `shape`, `rows_a`, `rows_b`, `method`, `log_n`, `threads`. Uses the
first `--log-n` (default 15).

| Option | Meaning |
|---|---|
| `--log-d` | update sizes, as powers of two (default `0..13`) |
| `--log-acc` | accumulated sizes for the `fully` shape (default `11,13`) |
