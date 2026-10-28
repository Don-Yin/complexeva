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

All functions take a 1D array. Results changed in 0.3.0; see [CHANGELOG.md](CHANGELOG.md).

## licence

MIT
