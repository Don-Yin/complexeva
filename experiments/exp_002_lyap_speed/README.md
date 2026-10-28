# exp_002_lyap_speed

## question

Which nearest-neighbour search accelerates `lyapunov_exponent` while preserving the Theiler exclusion, lowest-index tie rule, divergence trajectory and final result to absolute tolerance `1e-12`?

## inputs

The live `complexeva/lyapunov_exponent.py` baseline and deterministic synthetic signals. Main sizes are 500, 1000, 2000, 5000, 10000 and 20000; Gaussian signals use seed `20260429 + N`, and the logistic map starts at 0.3. No external dataset or new dependency is needed.

## outputs

- `.results/profile/`: initial baseline/stage profiles, the most recent `run.html`, and a failure-exit profile proving cleanup on error.
- `.results/tables/`: baseline/stage measurements, 102 fidelity checks, 204 raw timing rows and 68 timing summaries.
- `.results/manifests/`: environment, lock metadata and the package source at final measurement time.
- `.results/logs/`: tests, lint, profiling and benchmark logs.
- [Findings and limitations](docs/findings.md); [recommended patch](docs/recommended.patch), **not applied** to the package.

## how to run

Use the `emergence` Python 3.14 environment; set `NUMBA_CACHE_DIR` to an existing disposable scratch directory. The parallel comparison uses four threads. Every ordinary run profiles by default; the explicit `--no-profile` below prevents profiling overhead in the final timings.

```bash
export NUMBA_NUM_THREADS=4
micromamba run -n emergence python experiments/exp_002_lyap_speed/run.py --phase baseline
micromamba run -n emergence python experiments/exp_002_lyap_speed/run.py --phase stages
# Preserve run.{html,txt,json} under a descriptive name before another profiled run.
(
  mkdir /tmp/complexeva-bench.lock || exit 1
  trap 'rmdir /tmp/complexeva-bench.lock' EXIT
  export COMPLEXEVA_BENCH_LOCK_HELD=1
  micromamba run -n emergence python experiments/exp_002_lyap_speed/run.py --phase compare --repeats 3 --no-profile
)
micromamba run -n emergence python -m pytest -q -p no:cacheprovider tests experiments/exp_002_lyap_speed/tests
micromamba run -n emergence ruff check --no-cache complexeva tests experiments/exp_002_lyap_speed
```

The test suite intentionally runs one failing workload and replaces `run.html` with its failure profile; the successful baseline and stage reports are retained separately. The baseline is the live package, so applying the recommended patch changes what later runs call `original`; the captured source describes the recorded comparison.

## findings

- Nearest-neighbour search accounted for 99.87% of native stage time at N=20000, verified by an exact split rather than inferring from a Numba-opaque sampling profile.
- All 34 differential fixtures × 3 candidates and all 204 timed results passed; maximum observed finite-value error was **0.0**, and neighbour indices matched exactly.
- At N=20000, final lock-held Gaussian medians were 11.719 s original, 0.422 s scalar, 0.174 s parallel and 0.0316 s KD-tree. Shared-machine scheduling variability remains substantial; consult the recorded min/max values.
- KD-tree is fastest for the large low-dimensional continuous fixtures, but was 10.36× slower than the baseline in the wide-Theiler-window stress test. It is not a safe unconditional performance replacement.
- Recommend the minimal scalar-distance patch first: no temporary distance arrays, new dependencies, implicit threading, API change or altered reduction order. Its complexity remains quadratic. Keep the KD-tree and parallel implementations here as explicitly requested comparison candidates.
- Verification: **311 tests passed**, ruff passed, experiment gate passed, and `git apply --check` accepted the unapplied recommendation.
