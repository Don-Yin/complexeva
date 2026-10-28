# HFD speed findings

## Recommendation

Adopt the **regression-only** candidate (`slope_only`) after parent review: preserve the public validation and `compute_L_x`, and compute the centered OLS slope in the same Numba call instead of constructing a two-column design matrix and invoking `numpy.linalg.lstsq`.

This is a recommendation, **not an implemented package change**. All candidate implementations remain experiment-local because comparing these alternatives is the deliverable. The slightly faster allocation-light loop is numerically eligible, but its incremental gain is small at the default `k_max=10`; it is not needed for the simplest patch. Do not adopt NumPy vectorization or unconditional `prange`.

## Runtime and method

Final timing run: 2026-10-28, Apple M4 Max (16 logical CPUs), macOS arm64, Python 3.14.3, NumPy 2.4.3, Numba 0.64.0, SciPy 1.17.1, environment `emergence`.

- Synthetic contiguous float64 white noise; deterministic seed `20261028 + N`.
- Requested N = 300, 3,000, 30,000, 1,000,000; default `k_max=10`, plus sensitivity at 64.
- Full public calls, including the current shared finite-input check and float64 conversion, not just private kernels.
- JIT warm-up and verification precede timing. No profiler is active during final timing. Import, cold compilation and data generation are excluded.
- Eleven shuffled rounds of batched calls. Batch calibration targets 50 ms, with at least three calls per batch. Every raw sample is retained.
- BLAS thread limits are one; parallel candidates explicitly use 1, 2 or 4 Numba threads. `OMP_WAIT_POLICY=PASSIVE` and `KMP_BLOCKTIME=0` avoid intentionally busy-waiting workers.
- The final timing run held `/tmp/complexeva-bench.lock`; the full lock hold, including profiling and checks, was 79 seconds. The lock serializes cooperating speed benchmarks, not every OS process.
- `paired_speedup` is the median of the eleven within-round `current / candidate` ratios. It is **not** the ratio of the two marginal timing medians.

Evidence: [environment](../.results/json/benchmark-environment.json), [source snapshot](../.results/json/benchmark-higuchi-source.txt), [raw samples](../.results/json/benchmark.json), [CSV](../.results/tables/benchmark.csv), [lock log](../.results/logs/lock.log), [benchmark log](../.results/logs/benchmark.log).

## Default k_max=10: median microseconds per complete call

| N | Current | Loop + lstsq | Regression only | Loop + OLS | NumPy + lstsq | Parallel 2 | Parallel 4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 11.854 | 11.367 | 3.196 | 2.974 | 85.814 | 36.332 | 71.077 |
| 3,000 | 30.204 | 30.809 | 22.508 | 20.865 | 124.351 | 62.757 | 77.671 |
| 30,000 | 232.428 | 225.520 | 222.472 | 220.703 | 605.930 | 457.251 | 381.511 |
| 1,000,000 | 18,759.951 | 16,091.194 | 18,769.180 | 13,984.911 | 78,923.556 | 12,111.185 | 12,951.933 |

The recommended regression-only candidate has paired speedups **3.64×, 1.36×, 1.06×, 1.00×**. Allocation-light loops plus OLS have **3.94×, 1.45×, 1.08×, 0.93×**. The latter figure illustrates why ratios of medians should not be substituted for paired comparisons under drift.

**Noise limit:** even under the benchmark lock, the N=1,000,000 current batches ranged from **7.175 to 54.320 ms** and regression-only batches from **7.102 to 39.590 ms**. These large-N absolute numbers are local observations, not isolated-hardware throughput estimates. No large-N single-thread speedup is claimed for the recommended patch. Small-N serial comparisons are much more stable: at N=300 the current range was 10.739–14.594 µs, versus 3.081–3.432 µs for regression-only.

## k_max=64 sensitivity

Median complete-call times in microseconds:

| N | Current | Regression only | Loop + OLS | Parallel 4 | Parallel 4 paired speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 25.324 | 25.059 | 20.034 | 412.744 | 0.08× |
| 3,000 | 112.029 | 98.276 | 89.087 | 87.699 | 1.38× |
| 30,000 | 1,357.463 | 1,331.166 | 1,280.502 | 619.934 | 2.19× |
| 1,000,000 | 55,969.708 | 55,793.792 | 49,241.847 | 23,475.111 | 3.08× |

Parallelism can help a larger workload, but a hardware-specific dispatch threshold and process-level oversubscription policy have not been established. This experiment does not justify changing the public API or silently consuming multiple threads for every call. NumPy vectorization was slower than current at every measured N/k pair.

## Profile evidence

Profiling was performed before alternatives were implemented, and refreshed against the maturity track's updated public wrapper before final timing. Full reports are retained, not just a screenshot or summary.

- [N=300](../.results/profile/baseline-n300.txt): 0.603 s inside `hfd`; `lstsq` accounts for 0.273 s (**45.3%**), `column_stack` plus `ones_like` for 0.076 s (**12.6%**), and shared input validation for 0.100 s (**16.6%**). Removing the general regression path targets an observed cost, without weakening validation.
- [N=3,000](../.results/profile/baseline-n3000.txt): 0.125 of 0.627 s (**19.9%**) is in `lstsq`; 0.388 s is charged to the public wrapper's self/native region.
- [N=30,000](../.results/profile/baseline-n30000.txt): 0.537 of 0.564 s (**95.2%**) is charged to that self/native region.
- [N=1,000,000](../.results/profile/baseline-n1000000.txt): 81 samples, all charged to `hfd`. Pyinstrument cannot expand the compiled Numba loop here; this is not evidence that any particular inner instruction is the bottleneck.
- Matching HTML files live beside these text reports. [run.html](../.results/profile/run.html) records actual validation/candidate work by default; [baseline.json](../.results/json/baseline.json) retains the separately timed production/kernel decomposition. Its instrumented timings are **not** the final benchmark table.

## Numerical fidelity and boundaries

[Validation evidence](../.results/json/validation.json) contains **216 input/parameter cases × 8 implementations = 1,728 comparisons** (including the current implementation as a control).

Each implementation has 200 finite-result cases and 16 matching-NaN cases. Nonfinite disagreement fails rather than being skipped. The strict finite comparison is `abs(candidate - current) < 1e-12`, with no relative tolerance.

| Family | Maximum absolute error |
| --- | ---: |
| Allocation-light loops with unchanged lstsq | 0 |
| Vectorized NumPy with unchanged lstsq | 1.4210854715202004e-14 |
| Centered OLS, serial or parallel lengths | 1.376676550535194e-13 |

The largest OLS difference occurred for the signal scaled by `1e120`, N=3,000, `k_max=150`. The suite also covers float32/int16 public inputs, negative/strided views, near-constant large offsets, `1e-120` scaling, ramps, sine, white noise, Brownian paths, plateaus, constants, and small/boundary-valid `k_max`.

During parallel development the maturity track changed public input conversion to float64. The gate caught 112 mismatches when old candidates still computed float32 differences directly, then exited nonzero before timing. Candidates now reuse `as_finite_1d` and `check_positive_int` exactly; narrow-integer special handling was removed because it became unreachable. The failed comparison and the earlier contended exploratory timings remain explicitly excluded audit products. This is not a blanket claim of bit-identical results for all possible floating-point inputs.

## Recommended package edit

The following is the tested regression-only candidate, with private package names and the package's existing non-caching `@njit` policy. Keep `compute_L_x`, the existing public validation, annotations and docstrings unchanged. No `fastmath` or parallel reduction is needed.

```python
@njit
def _centered_slope(x, lengths):
    """Return the OLS slope with an intercept using centered sums."""
    mean_x = numpy.mean(x)
    mean_l = numpy.mean(lengths)
    numerator = 0.0
    denominator = 0.0
    for index in range(len(x)):
        centered_x = x[index] - mean_x
        numerator += centered_x * (lengths[index] - mean_l)
        denominator += centered_x * centered_x
    return numerator / denominator


@njit
def _hfd_core(X, k_max):
    """Compute production lengths and the slope in one native call."""
    x, lengths = compute_L_x(X, k_max)
    return _centered_slope(x, lengths)
```

After the existing checks in `hfd`, replace the `compute_L_x` / `column_stack` / `lstsq` block with:

```python
return float(_hfd_core(X, k_max))
```

The public wrapper must still validate and convert input before calling this core. The experiment's measured `slope_only_core` and `slope` have the same arithmetic; cache flags affect warm-up, which was excluded from the timings. Retain the literal-formula/known-dimension tests and add the strict candidate/reference fidelity cases when adopting the patch. This worker did not modify package code.

## Verification

- Project experiments gate passed at the beginning of each invocation and after implementation.
- The final combined production/experiment run passed: **275 tests in 4.85 s** (271 production tests and 4 experiment infrastructure checks). The experiment checks cover an intentionally wrong candidate being rejected, profile output on failure, explicit `--no-profile`, and import safety. Experiment imports use the full project namespace to avoid collisions with sibling experiments' `src` packages.
- `ruff check --no-cache .` passed. The active diagnostic sweep found no errors; eight numeric-cast heuristic warnings were reviewed as conversions of timing/mathematical results, not untrusted text parsing.
- A real default-profile validation run passed all 1,728 comparisons and wrote nonempty HTML/text output. The real boundary mismatch demonstrated nonzero failure propagation before timing. The production source still matches the source snapshot used for final timing.
- Canonical evidence: [final combined pytest](../.results/logs/pytest-final.log), [production pytest](../.results/logs/pytest.log), [experiment pytest](../.results/logs/pytest-experiment.log), [ruff](../.results/logs/ruff.log), [profiled validation](../.results/logs/profile-validate.log), [profile sizes/source check](../.results/logs/final-products.log), [final gate](../.results/logs/gate-final.log). No commit, push or tag was made.
