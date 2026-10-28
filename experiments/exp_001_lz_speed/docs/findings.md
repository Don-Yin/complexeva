# Lempel-Ziv speed findings

## Decision

Recommend replacing only `lempel_ziv_core` with the suffix-automaton implementation in [recommended.patch](recommended.patch), subject to accepting its memory and regular-input tradeoffs below. The patch is tested but **not applied** to the main checkout. It preserves input validation, median/mean/diff binarization, the exact SciPy Hilbert transform, raw integer counts, and normalization. No dependency is added. Packed64 remains an experimental comparison, not a second production implementation.

The final candidate uses native-width `numpy.intp` state indices, avoiding an int32-only input-length limit. All candidate timings and memory estimates below refer to that final version. The measured workload is N=300–1,000,000.

## Measurement protocol

- Apple M4 Max, arm64; Python 3.14.3 in `emergence`; NumPy 2.4.3, SciPy 1.17.1, Numba 0.64.0.
- Input at each N: `numpy.random.default_rng(n).standard_normal(n)`. The Hilbert case uses `abs(scipy.signal.hilbert(x))`; the main core/pipeline comparisons use median binarization.
- N: 300, 1,000, 3,000, 10,000, 30,000, 100,000, 300,000, 1,000,000.
- Profile first: the original N=300/10,000/30,000 exploratory profile is preserved as `.results/profile/initial-baseline.html`. It located the dominant cost in the counting core. Its unlocked timings are not used in the final table.
- Every final timed row holds `/tmp/complexeva-bench.lock`, acquired with atomic mkdir, released in `finally` including handled SIGTERM. Lock waits are outside the timed interval. The longest observed row was 145.84 seconds, below the approximately three-minute hold budget.
- Profiles remain on. Core/pipeline rows use three samples through N=10,000, otherwise one; N<=1,000 uses 30 inner calls per sample. Preprocessing uses three samples. Timings are medians, with minima and repeat counts retained in CSV.
- Cold calls use a separate seeded 300-bit int32 input and precede steady-state timing. No candidate enables Numba disk caching; `NUMBA_CACHE_DIR` points to scratch.
- This is a shared-machine experiment, not a dedicated-host latency study. Even with the agreed lock, nonbenchmark processes and profiler overhead affect timings. For example, the N=10,000 raw baseline core had a 14.44 ms minimum and a 29.63 ms median. Large-N rows are single measurements; interpret speedups as approximate orders of magnitude, not confidence intervals.

## Authoritative evidence

All paths below are relative to this experiment:

- `.results/tables/timings.csv`: use its **production** core/pipeline rows as the baseline.
- `.results/tables/candidates-timings.csv`: use its candidate rows and preprocessing comparisons for the final recommendation.
- `.results/tables/candidates-first-calls.csv`: final cold-call measurements.
- `.results/tables/candidates-verification.json`: exhaustive/seeded verification.
- `.results/profile/run.html` / `run.txt`: full baseline comparison; 741.195 s wall time, including 175.23 s spent waiting for the shared lock. The CSV excludes those waits.
- `.results/profile/candidates.html` / `candidates.txt`: final native-index candidates; 21.053 s wall time.
- `.results/logs/locked-run.log`, `locked-candidates.log`, `tests.log`, `ruff.log`, `patch-tests.log`, `patch-ruff.log`.

The maturity worker added finite/real input validation while this experiment was running. The final candidate-only pass therefore reuses the same validator and includes the Hilbert input check. It also uses native-width indices instead of the initial int32 state arrays. All earlier candidate rows in `timings.csv` are superseded; they must not be used for the final recommendation. Production baseline rows remain valid. A cross-run audit compared all 72 final candidate core/pipeline count cells with their production rows and found zero differences.

## Warm core timings

Milliseconds; every displayed value comes from lock-held rows. The production columns are from `timings.csv`, candidate columns from `candidates-timings.csv`.

| N | Raw production | Raw packed64 | Raw automaton | Hilbert production | Hilbert automaton |
|---:|---:|---:|---:|---:|---:|
| 300 | 0.00850 | 0.000972 | 0.00230 | 0.01004 | 0.00208 |
| 1,000 | 0.07542 | 0.00655 | 0.00671 | 0.08284 | 0.00639 |
| 3,000 | 1.398 | 0.05558 | 0.03754 | 1.280 | 0.03404 |
| 10,000 | 29.632 | 0.5025 | 0.1639 | 12.997 | 0.1666 |
| 30,000 | 187.005 | 3.9192 | 0.7659 | 111.934 | 0.7589 |
| 100,000 | 1,734.061 | 107.491 | 4.458 | 1,430.832 | 3.176 |
| 300,000 | 11,763.766 | 383.877 | 14.671 | 11,020.842 | 15.560 |
| 1,000,000 | 145,844.379 | 3,905.641 | 92.163 | 112,058.177 | 98.345 |

At N=1,000,000, the exact raw/Hilbert counts are **50,770 / 48,051**. The automaton core is approximately **1,582x / 1,139x** faster than production for these inputs. Packed64 is approximately 37x faster on the raw case but retains quadratic scaling.

### End-to-end calls at N=1,000,000

| API path | Production | Validation-matched automaton | Approximate speedup |
|---|---:|---:|---:|
| `lempel_ziv(x)` | 126.361 s | 119.810 ms | 1,055x |
| `lempel_ziv_hilbert(x)` | 110.182 s | 199.370 ms | 553x |

These are measured full calls, not sums of separately timed stages. The candidate wrapper applies the same validation/transform/binarization on the benchmark's numeric inputs; the proposed patch itself retains the production public wrappers unchanged. Short-input ratios are more sensitive to Python/profiler overhead and scheduling, so they are not the basis of the recommendation.

## Preprocessing and cold calls

At N=1,000,000 in the final candidate pass:

| Stage | Raw | Hilbert |
|---|---:|---:|
| SciPy Hilbert transform | — | 15.625 ms |
| Envelope absolute value | — | 1.949 ms |
| Median binarization, including production validation and int32 storage | 7.216 ms | 8.980 ms |
| Mean binarization | 0.826 ms | 0.844 ms |
| Diff binarization | 1.384 ms | 1.393 ms |

The old counting core costs over 100 seconds at this length. Removing the int32 cast saves at most about 0.4 ms in these final preprocessing comparisons, so the recommended patch deliberately leaves binarization unchanged. Changing FFT length or Hilbert arithmetic was not attempted because exact thresholded counts must be preserved.

Final first-call / warmed-call times on the same 300-bit input:

| Core | First call | Warm call | First minus warm |
|---|---:|---:|---:|
| Production | 363.622 ms | 7.400 us | 363.615 ms |
| Packed64 | 409.502 ms | 0.897 us | 409.502 ms |
| Automaton | 837.734 ms | 2.093 us | 837.732 ms |

These are first invocations of fresh signatures in one process, including compilation. Production runs first and also pays Numba runtime initialization; later candidates reuse that runtime. They are not independently cold-process startup comparisons. The profile records approximately 1.609 s in `_compile_for_args` during these initial calls. The automaton trades a larger one-time compilation cost for much faster warm execution; disk-cache behavior was not measured.

## Why the counts agree

The automaton stores the earliest ending position for each state. A matched substring of length `matched` has appeared at a previous starting position exactly when:

```python
first_end[state] - matched + 1 < position
```

The previous **start**, rather than previous end, is essential: it permits self-overlapping copies. At each phrase start, parsing advances to the first substring without such a prior start, or to the end of the input. The latter is still counted as the final copied component. Empty and one-symbol inputs retain the existing count of 1. Construction and parsing are linear in input length for the fixed binary alphabet.

Packed64 precomputes 64-bit lookahead words and uses XOR-prefix comparisons. Zero-XOR cases are checked against the actual remaining symbols; it is not a probabilistic hash method. It removes much branching but still scans preceding positions for each phrase.

## Verification

- All 32,766 nonempty binary words through length 14.
- 1,000 seeded random strings, lengths 2–4,096, with probabilities 0.01, 0.1, 0.5, 0.9 and 0.99.
- 112 periodic/constant cases at lengths around 64-bit boundaries and up to 10,000.
- Total: **33,878 binary cases, zero integer mismatches**, for both candidates.
- **210 numeric API configurations**: all three binarization methods, raw/Hilbert, rounded/constant/reversed/float32 inputs, and strict normalization equality. Boolean storage is also checked against int32 counts.
- Full timed N sweep and structured controls: exact count equality, including N=1,000,000. A separate audit matched the final candidate-only rows to the earlier production rows.
- Experiment regressions also cover empty input, noncontiguous/read-only arrays, retained validation, import without profiling, writing a profile when work fails, nonzero entry exits, and lock release on failure.
- `micromamba run -n emergence python -m pytest -q -p no:cacheprovider tests experiments/exp_001_lz_speed/tests`: **292 passed**, 4.43 s.
- Applying the recommended change only in an isolated copy, then running the same tests: **292 passed**, 4.53 s.
- `ruff check --no-cache complexeva tests experiments/exp_001_lz_speed`: passed for both copies.
- `git apply --check experiments/exp_001_lz_speed/docs/recommended.patch`: passed; the patch remains unapplied.

## Tradeoffs and scope

- The automaton allocates five native indices per state and reserves at most 2N states: **80N bytes on this 64-bit platform**, approximately 80 MB at N=1,000,000, excluding input and Python/Numba overhead. This is an allocation bound from the implementation, not a measured RSS claim. Packed64 uses 8N bytes; the old core uses constant extra memory.
- At N=1,000,000, constant input takes production 0.489 ms versus automaton 3.946 ms; alternating input takes 0.436 ms versus 5.879 ms. Both preserve counts 2 and 3. The absolute regression is about 3.5–5.4 ms, but the relative regression is material. No hidden fallback or data-dependent dispatch was added.
- The current recommendation is a single core replacement. It does not recommend changing validation, string handling, normalization, binarization dtype, or Hilbert arithmetic.
