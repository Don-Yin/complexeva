# exp_000_hfd_speed

## question

Can warmed `hfd` calls become faster without changing their output by `1e-12` or more? Compare allocation-light Numba loops, vectorized NumPy, parallelism over `k`, and removal of the general-purpose least-squares solver.

## inputs

- The current `complexeva.higuchi.hfd` and its shared input validation; the package is not modified by this experiment.
- Reproducible white noise at N = 300, 3,000, 30,000, 1,000,000, with `k_max=10` and 64.
- Fidelity cases additionally cover Brownian paths, sine, ramps, plateaus, strides, reversed views, float32/int16 inputs, extreme scaling, constants and `k_max` boundaries.
- Python environment: `emergence`.

## outputs

- `.results/profile/`: actual-work HTML/text profiles, including one production baseline for each N and default `run.html`.
- `.results/json/`: raw timing samples, source snapshots, environment, and strict fidelity checks.
- `.results/tables/benchmark.csv`: final warmed timings collected while holding the shared benchmark lock.
- `.results/logs/`: benchmark, lock, hardware, pytest and ruff evidence.
- [Findings and recommended patch](docs/findings.md). The `*-exploratory*` and `*-boundary-mismatch*` products are excluded from final performance conclusions.

## intermediates

`.intermediates/numba-cache/` is disposable private JIT scratch; `.intermediates/pytest/` holds test scratch. Neither is a shared experiment artifact.

## how to run

From the repository root:

```bash
export NUMBA_CACHE_DIR="$PWD/experiments/exp_000_hfd_speed/.intermediates/numba-cache"
export NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
micromamba run -n emergence python experiments/exp_000_hfd_speed/run.py --stage baseline
micromamba run -n emergence python experiments/exp_000_hfd_speed/run.py --stage validate
# Run final timing only while owning /tmp/complexeva-bench.lock; release it on exit.
# Do not run other speed workers' benchmarks concurrently.
micromamba run -n emergence python experiments/exp_000_hfd_speed/run.py \
  --stage benchmark --no-profile --repeats 11 --target-seconds 0.05
micromamba run -n emergence python -m pytest -q -p no:cacheprovider \
  tests experiments/exp_000_hfd_speed/test_experiment.py \
  --basetemp experiments/exp_000_hfd_speed/.intermediates/pytest
micromamba run -n emergence ruff check --no-cache .
```

The default stage is `all`, with actual-work profiling enabled. Profiled timings are deliberately written to `benchmark-profiled.*`, never to the final unprofiled timing table. Importing the entry performs no experiment work and starts no profiler. A fidelity failure exits nonzero before timing begins.

## findings

On an Apple M4 Max, the conservative regression-only change (`slope_only`) gave paired speedups of **3.64× / 1.36× / 1.06× / 1.00×** at the four requested sizes with `k_max=10`. It retains the production length kernel and replaces the Python/SVD regression with centered OLS inside Numba. The largest observed absolute difference was **1.38e-13**, below the strict **1e-12** gate, over 216 inputs and 1,728 comparisons. Do not enable parallelism by default: it loses on short signals and remains noisy on this shared machine. No speed patch was applied to `complexeva/`.
