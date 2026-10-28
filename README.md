# complexeva

Complexity measures for 1D time series, implemented with NumPy, SciPy and Numba.

## install

```bash
pip install complexeva
```

Requires Python 3.14 or later.

## example

```python
import numpy as np
import complexeva

x = np.random.default_rng(0).standard_normal(2000)  # white noise

complexeva.hfd(x)                          # about 2.0
complexeva.lempel_ziv(x, normalize=True)   # about 1.0
complexeva.compression_complexity(x)       # about 0.96
complexeva.lyapunov_exponent(x)            # largest Lyapunov exponent
```

## measures

| function | what it returns | main parameters | reference |
|---|---|---|---|
| `hfd(X, k_max=10)` | Higuchi fractal dimension, 1 (smooth) to 2 (white noise) | `k_max`: largest time interval, 2 to len(X) // 2; report it with results | Higuchi 1988, Physica D 31:277 |
| `hfd_hilbert(X, k_max=10)` | `hfd` of the Hilbert envelope (instantaneous amplitude) | as `hfd` | |
| `lempel_ziv(x, method="median", normalize=False)` | Lempel-Ziv complexity: integer count, or c·log2(n)/n (about 1 for random input) when `normalize=True` | `method`: binarise by `"median"`, `"mean"` or `"diff"` (rising step); a `"0101..."` string is used as is | Kaspar & Schuster 1987, Phys. Rev. A 36:842 |
| `lempel_ziv_hilbert(x, method="median", normalize=False)` | `lempel_ziv` of the Hilbert envelope | as `lempel_ziv` | |
| `compression_complexity(data)` | zlib-compressed size / raw size of the array bytes, 0 to about 1 | depends on the array dtype | |
| `lyapunov_exponent(data, tau=1, m=2, dt=1.0, min_tsep=None, trajectory_len=20, fit_offset=0)` | largest Lyapunov exponent in units of 1/`dt`; positive suggests chaos, about 0 for periodic signals | `tau`, `m`: delay embedding; `min_tsep`: Theiler window (default: mean period); `trajectory_len`: steps fitted | Rosenstein et al. 1993, Physica D 65:117 |
| `sample_entropy(x, order=2, tolerance=None)` | sample entropy in nats; lower values indicate more regularity; NaN if no length-order templates match, infinity if none extend | `order`: template length; `tolerance`: strict Chebyshev threshold, default 0.2 × population std; no self-matches | Richman & Moorman 2000, Am. J. Physiol. 278:H2039 |
| `permutation_entropy(x, order=3, delay=1, normalize=False)` | ordinal-pattern entropy in bits, or 0 to 1 when normalized by log2(order!) | `order`: pattern length (≥ 2); `delay`: spacing in samples; ties retain temporal order | Bandt & Pompe 2002, Phys. Rev. Lett. 88:174102 |
| `dfa(x, nvals=None)` | DFA-1 scaling exponent: about 0.5 for white noise and 1.5 for Brownian motion, not a fractal dimension | `nvals`: at least two distinct integer window sizes, 4 to len(x) − 1; default geometric scales 4 to len(x)/10; non-overlapping windows, linear detrending | Peng et al. 1994, Phys. Rev. E 49:1685 |

All functions take a 1D array. The new entropy and DFA functions require finite,
real values. Sample entropy uses a strict distance threshold: a constant series
has NaN at its default zero tolerance, or 0 with a positive tolerance. DFA returns
NaN for a constant series; its default scales require at least 58 samples. Report
the template/delay/tolerance or fitted window sizes with results. The new measures
are checked against antropy/nolds, which are not runtime or test dependencies.
Results changed in 0.3.0; see [CHANGELOG.md](CHANGELOG.md).

## synthetic-data checks

For fractional Brownian motion (fBm), the theoretical graph dimension is
**D = 2 − H**. HFD tracks this relation without rescaling its output. Each point
below uses 32 independently sampled paths of 8,192 samples, with `k_max=10`;
error bars show ±1 standard deviation across paths, not a confidence interval.
The paths use circulant embedding of fractional Gaussian noise, followed by a
cumulative sum.

![Higuchi fractal dimension follows the theoretical fBm dimension from 1.9 to 1.1 as the Hurst exponent increases.](docs/figures/hfd-fbm.png)

With the same sample length and median binarization, normalized Lempel–Ziv
complexity is low for a periodic sine wave and close to 1 for Gaussian white
noise. The noise bar summarizes 32 trials; the sine wave has period 128 samples.
The dashed line is the asymptotic random-sequence limit, not an upper bound.
These are finite-sample sanity checks, not guarantees for every signal or parameter.

![Normalized Lempel–Ziv complexity is low for a sine wave and near one for white noise, with signal excerpts alongside.](docs/figures/lz-sine-noise.png)

Regenerate both figures from a repository checkout (fixed RNG seed 0):

```bash
python -m pip install -e . matplotlib
python examples/validate.py
```

The script writes small, headless PNGs to `docs/figures/`; use `--output-dir PATH`
to write them elsewhere. Matplotlib is only needed for these plots, not to use
the package.

## licence

MIT
