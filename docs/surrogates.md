# Surrogate tests on the CPU

A surrogate is a synthetic copy of a time series that keeps selected features of the original. Use `surrogate_test` to ask whether a measure differs from what those features alone would predict.

```python
from functools import partial

import numpy
from complexeva import sample_entropy, surrogate_test

signals = numpy.random.default_rng(0).standard_normal((8, 400))
measure = partial(sample_entropy, order=2)
result = surrogate_test(measure, signals, n=99, seed=42, alternative="less")
print(result.observed, result.surrogate_mean, result.delta, result.pvalue)
```

The callable receives one 1D `float64` signal at a time and must return one real number without modifying the signal. Each result field is a float for a single signal, or an array with the input's leading shape for a batch. `delta` is `observed - surrogate_mean`; it is not a standardized effect size.

## Choose what the surrogates preserve

| Method | Preserves | Use when |
| --- | --- | --- |
| `method="phase"` (default) | Mean and Fourier magnitudes up to roundoff. Interior phases are randomized; the zero-frequency bin and the even-length Nyquist bin are fixed. | The reference is a stationary random-phase signal. The original distribution of sample amplitudes need not be preserved. |
| `method="iaaft"` | The exact sorted `float64` samples and approximate Fourier magnitudes. | The reference should also keep the original distribution of sample amplitudes. |

IAAFT means **iterative amplitude-adjusted Fourier transform**. Each copy starts as an independent time permutation. Iteration alternates matching Fourier magnitudes and remapping ranks to the original sorted samples. The final remapping preserves the samples, not the exact spectrum. `max_iter` limits the number of iterations (default 100); the algorithm stops earlier when the copies stop changing.

```python
from complexeva import iaaft_surrogates, phase_randomized_surrogates

phase_copies = phase_randomized_surrogates(signals, n=10, seed=42)  # (8, 10, 400)
iaaft_copies = iaaft_surrogates(signals, n=10, seed=42)             # (8, 10, 400)
```

Both generators insert the copy axis just before time. A fixed integer `seed` reproduces a call on the same input; passing a NumPy random generator instead advances its state.

## Read the p-value

- `alternative="less"`: `(1 + count(surrogate_scores <= observed)) / (n + 1)`.
- `alternative="greater"`: the same calculation with `>=`.
- `alternative="two-sided"` (default): twice the smaller one-sided value, capped at 1.

Ties count in either tail. The smallest one-sided p-value is `1 / (n + 1)`. Any NaN score makes that signal's p-value NaN. Infinite scores are ranked as extreme values, but can make the mean or difference undefined. Batch p-values are not adjusted for multiple tests.

Choose the measure, null and tail before looking at the result. A small p-value is evidence against the chosen surrogate reference, not proof of a specific nonlinear mechanism. Trends, changing statistics over time, and discontinuities where the Fourier transform joins the endpoints can make the reference unsuitable. Check spectral agreement before interpreting a test with IAAFT.

Methods: [Theiler et al. (1992), the method of surrogate data](https://doi.org/10.1016/0167-2789(92)90102-S) and [Schreiber & Schmitz (1996), improved surrogate data](https://doi.org/10.1103/PhysRevLett.77.635).
