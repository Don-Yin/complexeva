# exp_001_lz_speed

## question

Can `lempel_ziv` and `lempel_ziv_hilbert` count exactly the same components faster, from N=300 to N=1,000,000?

## inputs

Seeded Gaussian signals (`numpy.random.default_rng(n)`), their exact SciPy Hilbert envelopes, exhaustive short binary words, and constant/periodic controls. Production functions are imported from `complexeva/`; this experiment does not change them.

## outputs

- `.results/tables/timings.csv`: lock-held production baseline and initial candidates.
- `.results/tables/candidates-timings.csv`: final candidates with the maturity track's input validation retained.
- `.results/tables/*first-calls.csv`, `*verification.json`, `*environment.json`: cold-call and equality evidence.
- `.results/profile/run.html` and `candidates.html`: actual-work profiles; text versions are beside them.
- `.results/logs/`: benchmark, pytest, and ruff evidence, including an isolated application of the proposed patch.
- [Findings](docs/findings.md) and [recommended patch](docs/recommended.patch): durable review material.

## intermediates

None. Set `NUMBA_CACHE_DIR` to a disposable scratch directory. The patch was tested in an isolated copy; only the patch and verification logs were promoted here.

## how to run

```bash
NUMBA_CACHE_DIR="$(mktemp -d /tmp/complexeva-lz-numba.XXXXXX)" \
  micromamba run -n emergence python experiments/exp_001_lz_speed/run.py
# Default: profile/run.html; --no-profile is the explicit opt-out.
# Each timed row acquires /tmp/complexeva-bench.lock and releases it on failure.

# Rerun candidates without repeating the slow production baseline:
NUMBA_CACHE_DIR="$(mktemp -d /tmp/complexeva-lz-numba.XXXXXX)" \
  micromamba run -n emergence python experiments/exp_001_lz_speed/run.py --candidates-only

micromamba run -n emergence python -m pytest -q -p no:cacheprovider \
  tests experiments/exp_001_lz_speed/tests
micromamba run -n emergence ruff check --no-cache complexeva tests experiments/exp_001_lz_speed
```

## findings

The core loop dominates long random signals, not binarization or Hilbert. At N=1,000,000 the suffix-automaton core took 92 ms (raw) / 98 ms (Hilbert bits), versus 145.84 s / 112.06 s in production, with identical counts. It needs approximately 80 MB of working arrays at this length and is slower on trivial periodic inputs (about 4–6 ms instead of 0.5 ms). The smaller packed64 alternative still has quadratic scaling. The recommended patch replaces only the core, leaves validation/transforms/normalization unchanged, and adds no dependency. It is **not applied** to the main checkout.

Both candidates passed 33,878 binary cases and 210 numeric API configurations. Existing tests plus experiment regressions: **292 passed** in both the main checkout and the patched isolated copy; ruff passed for both. The final implementation uses native-width state indices. See the findings for timing limitations, first-call costs, and memory tradeoffs.
