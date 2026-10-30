# changelog

## 0.4.1 (2026-10-30)

- `hfd`, `dfa`, `permutation_entropy`, `sample_entropy` and numeric `lempel_ziv` accept a batch with time on the last axis, e.g. (channels, time), and return one value per signal, identical bit for bit to the 1D call. `hfd_hilbert`, `lempel_ziv_hilbert`, string input, `lyapunov_exponent` and `compression_complexity` stay 1D.
- `dfa` fits its slope with a centred two-pass least squares in compiled code instead of `numpy.polyfit`; against a 60-digit reference the worst error falls from 1.6e-9 to 3.0e-11.
- `permutation_entropy` (orders 2-7) computes entropy in compiled code; values may change in the last bits (at most 5e-14).
- Every memory layout gives the same result; batches are processed row by row without copying the whole input.
- Leaner: a batch call uses less peak heap than a loop of antropy calls, on every measure and shape benchmarked. `dfa` no longer stores the profile; `sample_entropy` uses int32 tree indices and caps concurrent trees at 12 MiB; `lempel_ziv` uses a packed-bit occurrence index instead of a suffix automaton (about 4x less peak memory per signal at N = 100k; worst-case time is no longer linear); `hfd` and `permutation_entropy` batches run serially.

## 0.4.0 (2026-10-29)

- Added type hints, a PEP 561 `py.typed` marker, and complete parameter/return docstrings for the existing measures.
- Replaced assertion-based input checks with `ValueError` messages naming invalid shapes or values; numeric measures validate finite real 1D input, and Lempel-Ziv rejects nonbinary strings and unknown numeric binarization methods.
- float32/integer inputs are now computed in float64; results can differ in the last digits. `compression_complexity` is exempt and preserves the original dtype and bytes.
- `lyapunov_exponent` validates embedding, trajectory, fitting and sampling parameters; constant signals require an explicit `min_tsep` rather than failing during the default period estimate.
- `compression_complexity` now rejects non-1D arrays and empty byte representations with informative `ValueError` messages; NaN and infinity remain valid raw-byte input.

- Added `sample_entropy(x, order=2, tolerance=None)`: sample entropy in nats, with strict Chebyshev matches, no self-matches and default tolerance 0.2 × population standard deviation.
- Added `permutation_entropy(x, order=3, delay=1, normalize=False)`: ordinal-pattern entropy in bits, optional normalization by log2(order!), and deterministic temporal ordering for ties.
- Added `dfa(x, nvals=None)`: DFA-1 scaling exponent with non-overlapping windows, linear detrending and configurable window sizes.
- New measures are exported at the top level and have known-answer tests checked against antropy/nolds; neither reference package is a dependency.

- Faster, with unchanged results: `lempel_ziv` counts with a suffix automaton in linear time (N = 100000: about 1.7 s to 5 ms for the core count; identical integers); `lyapunov_exponent` computes neighbour distances without temporary arrays (about 28x at N = 20000; identical output); `hfd` fits its slope natively instead of with `lstsq` (about 3x at N = 300; differences below 1e-14); `permutation_entropy` counts ordinal patterns in a fixed histogram for orders up to 7 (about 450x at N = 100000); `sample_entropy` counts template matches with an exact dual tree on long series (N = 100000: about 8.8 s to 0.26 s); `lyapunov_exponent` prunes its exact neighbour search by sorting (N = 100000: about 9 s to 0.09 s); `dfa` and `hfd_hilbert` allocate less (1.2-1.9x faster).
- Further speed and memory work: `dfa` fits every window size in one compiled kernel (N = 1000: about 105 to 48 µs); `sample_entropy` counts without branches (N = 100000: about 263 to 144 ms); `lyapunov_exponent` skips blocks inside the Theiler window, so trending series are no longer quadratic (noisy line, N = 100000: 2.5 s to 0.15 s); `lempel_ziv` uses about 32 instead of 80 MB per million samples and rejects series of 2**30 samples or more. `dfa` now matches an exact reference to within κ ulp instead of reproducing NumPy's summation order; slopes on ill-conditioned adjacent window sizes can differ from 0.3.x by up to about 1e-10.
- Faster start: Numba kernels are cached on disk and `scipy.signal` is no longer imported (import 0.56 to 0.20 s). Read-only installs should set `NUMBA_CACHE_DIR`.

## 0.3.0 (2026-10-28)

Numerical results of three measures change. To reproduce earlier results, pin `complexeva==0.2.3`.

- Breaking: measures are imported from the top level (`from complexeva import hfd`); `complexeva.sequence` is removed.
- Breaking: removed unexported helpers (`hfd_hilbert_envelope/transform/frequency`, `lempel_ziv_hilbert_envelope/transform/frequency`, `time_series_to_binary`) and the unused `**kwargs` of `lempel_ziv`.

- `hfd`, `hfd_hilbert`: default `k_max` is now 10 (was 64), which gives lower error and better discrimination on series of a few hundred points.
- `hfd`: raises `ValueError` when `k_max` is outside 2 to len(X) // 2 (previously `ZeroDivisionError` or NaN on short series).
- `hfd`: now returns the standard Higuchi fractal dimension D; 0.2.3 returned D - 1 (missing `/k` in L_m(k)) and skipped the last increment of each subsequence.
- `lempel_ziv`: now counts the final unfinished component as in Kaspar & Schuster (1987); 0.2.3 undercounted by 1 on most inputs. New `normalize=True` option returns c*log2(n)/n.
- `lyapunov_exponent`: rewritten as the Rosenstein et al. (1993) largest Lyapunov exponent with a Theiler window; 0.2.3 averaged one-step log distances and depended on signal amplitude. New options `min_tsep`, `trajectory_len`, `fit_offset`.
