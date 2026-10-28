# Lyapunov speed experiment

## decision

Recommend [the minimal scalar-distance patch](recommended.patch) in the nearest-neighbour loop. It removes one temporary array per candidate pair while preserving ascending-index tie selection, arithmetic order, Theiler exclusion and the existing divergence reduction. The patch is **a recommendation only**; this worker did not edit `complexeva/`.

Keep the KD-tree and parallel implementations as measured experimental alternatives. KD-tree has the strongest large-N speedups on continuous low-dimensional fixtures, while large Theiler windows and repeated points can make it slower than the original. The parallel kernel retains quadratic work and introduces thread-pool scheduling and oversubscription concerns. An unconditional KD-tree replacement or automatic backend threshold is not supported by this experiment.

## profile-first evidence

The initial production run covered N=500, 1000, 2000, 5000, 10000 and 20000 before candidate implementation. Its pyinstrument profile recorded only eight samples because the compiled kernel is opaque to Python sampling. The report locates time in the public function but cannot identify native sub-stages.

The next run split the original kernel into `neighbours_original` and `follow_divergence`, preserving the original operations. At every N, all divergence values matched the production `_divergence` exactly. Search occupied 95.02%, 97.66%, 98.67%, 99.59%, 99.79% and **99.87%** of native stage time, respectively. An additional interpreted N=500 diagnostic exposes the repeated NumPy reductions; its timings are not used for throughput comparisons.

Evidence:

- `../.results/profile/baseline.{html,txt,json}`
- `../.results/profile/stages.{html,txt,json}`
- `../.results/tables/stages.csv`

These initial diagnostic runs were exploratory and unlocked. Every final timing below comes from the separately lock-held comparison.

## protocol

- Runtime: `emergence`, Python 3.14.3, NumPy 2.4.3, SciPy 1.17.1, Numba 0.64.0; macOS arm64, 16 logical CPUs.
- Numba parallel candidate: four threads. KD-tree: one query worker, `leafsize=16`, exact `eps=0` queries. No `fastmath` or parallel floating-point reduction.
- Compilation/startup is measured separately and excluded from throughput. All four methods are interleaved in seeded shuffled order, with three repetitions per workload. Reported values are medians, not best-of-run values.
- Final run held the shared atomic benchmark lock for **139 seconds**. Metadata records `shared_timing_lock_held=true` and the start time `2026-10-28T22:06:29.604088+00:00`.
- The lock coordinates the speed workers only; it does not isolate the computer. This run has substantial scheduling variability. Raw repetitions and min/max values are retained, and ratios should not be interpreted as machine-independent guarantees.
- Candidate end-to-end paths include current public argument validation, embedding, default-window FFT, search, unchanged divergence reduction and line fitting. The recorded original package source is retained in `../.results/manifests/package-source.py`.

## final timings

Gaussian signals, `m=2`, `tau=1`, `trajectory_len=20`, inferred Theiler window; **seconds**, medians of three lock-held repetitions:

| N | Original | Scalar scan | Parallel scan, 4 threads | KD-tree |
|---:|---:|---:|---:|---:|
| 500 | 0.004709 | 0.000689 | 0.000506 | 0.000778 |
| 1000 | 0.018648 | 0.001489 | 0.003091 | 0.001365 |
| 2000 | 0.097393 | 0.005628 | 0.019926 | 0.002740 |
| 5000 | 0.668064 | 0.047509 | 0.037301 | 0.006852 |
| 10000 | 3.349287 | 0.132800 | 0.067944 | 0.014277 |
| 20000 | 11.719267 | 0.422251 | 0.174498 | 0.031605 |

At N=20000, these correspond to **27.75×**, **67.16×** and **370.81×** speedups. The original ranged from 9.897 to 11.926 seconds; KD-tree ranged from 0.029425 to 0.035444 seconds.

Logistic-map input at N=20000 (`m=1`, `trajectory_len=8`) gave 9.767813 seconds original, 0.310666 scalar, 0.205328 parallel and 0.015298 KD-tree: **31.44×**, **47.57×** and **638.50×**. The original ranged from 8.135 to 14.788 seconds, illustrating the shared-host uncertainty.

Stress workloads at N=2000, also **seconds**, three lock-held repetitions:

| Workload | Original | Scalar | Parallel | KD-tree |
|---|---:|---:|---:|---:|
| Constant, explicit window=5 | 0.174427 | 0.094926 | 0.012415 | 0.360003 |
| Quantized ties, window=7 | 0.250867 | 0.009518 | 0.008193 | 0.008413 |
| Monotone, window=700 | 0.086459 | 0.003281 | 0.015577 | 0.895941 |
| Gaussian, m=16 | 0.209694 | 0.011879 | 0.026245 | 0.129237 |
| Gaussian, m=32 | 0.323997 | 0.053720 | 0.018760 | 0.063941 |

The wide-window KD-tree result is **10.36× slower** than the original. Constant-data KD-tree is **2.06× slower**. Scalar scanning was faster than the original in every measured workload, without a new global threading policy. Small-input parallel timings are sensitive to scheduling and can exceed the scalar candidate.

Full evidence: `../.results/tables/timings.csv` and `../.results/tables/timings-raw.csv`.

## numerical fidelity

All **34 differential fixtures × 3 candidates = 102** comparisons passed `atol=1e-12, rtol=0`. Every nearest-neighbour index matched exactly, maximum observed finite divergence error was **0.0**, and maximum final-exponent error was **0.0**. All **204 timed results** also passed.

Coverage includes logistic and Henon maps, sine, constants, quantized and periodic ties, near ties, exact Theiler feasibility boundaries, wide windows, zero exclusion, strided/reversed inputs, `tau`, `dt`, `fit_offset`, dimensions 1–32, ten random seeds, very small/large finite magnitudes, and squared-distance underflow/overflow. Matching infinities are accepted explicitly. Further tests cover tied initial distances with different future trajectories and exclusion of points lacking a full future trajectory.

The KD-tree candidate:

1. Builds the tree only over the first `n_traj` vectors.
2. Excludes every `abs(i-j) <= min_tsep` candidate.
3. Recomputes squared distances with the original scalar accumulation order and picks the lowest index on ties.
4. Doubles k until a strict distance gap confirms that a tie was not truncated; a floating-point margin includes near-boundary candidates.
5. Bounds query matrices to 65536 candidate entries per batch and handles SciPy's `tree.n` missing-neighbour sentinel, including overflowed distances.

This is empirical equivalence over the stated inputs, not a formal floating-point proof for every possible array. Duplicate-heavy data and large exclusions can force all-pairs work; dimensionality also weakens tree pruning.

Evidence: `../.results/tables/verification.csv` and the experiment's tests.

## verification and patch scope

- `micromamba run -n emergence python -m pytest -q -p no:cacheprovider tests experiments/exp_002_lyap_speed/tests`: **311 passed in 13.03 seconds**, including 40 experiment tests.
- `micromamba run -n emergence ruff check --no-cache complexeva tests experiments/exp_002_lyap_speed`: passed.
- Experiment convention gate: passed with three project experiments.
- `git apply --check experiments/exp_002_lyap_speed/docs/recommended.patch`: passed; no patch was applied.
- Import safety and a real failing CLI workload are tested. Failure preserves a nonzero exit and writes the default profile; `failure-exit.*` is intentionally empty of samples because validation fails immediately.
- Active LSP inspection found no language-server errors. Six auxiliary numeric-parse warnings refer to conversions of internally computed NumPy/Python numeric values, not unvalidated strings; no catch-all handlers were added.

The recommendation changes only the search distance calculation. It leaves the current public validation/docstrings/type hints, embedding, Theiler default, `dt`, fitting, result convention and serial divergence accumulation untouched. Complexity remains O(N²m); the experiment does not claim the minimal patch solves the asymptotic scaling problem.
