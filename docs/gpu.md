# Optional GPU backend

Install `pip install 'complexeva[gpu]'`. This adds PyTorch; importing `complexeva` on its own does not load it or change CPU behavior.

```python
import numpy
from complexeva import gpu

signals = numpy.random.default_rng(0).standard_normal((32, 400))
dimension = gpu.hfd(signals, k_max=10)
```

Higuchi fractal dimension (`hfd`) and the shuffle test below accept one signal or a batch with time on the last axis. Set `device="cuda"`, `device="mps"` (Apple Metal) or `device="cpu"` to select where PyTorch runs. By default, PyTorch tensor inputs stay on their current device. For other inputs the order is CUDA, Apple Metal, then CPU.

`hfd` returns a Python float for one signal or a NumPy array with one value per signal. Undefined fits give NaN. PyTorch uses float32 on Apple Metal and float64 on CUDA and the CPU, both for Higuchi and for generating shuffled signals. Use `device="cpu"` if you need float64 and do not have CUDA.

Speed depends on batch size, device and data transfers. In the repository's local Apple Metal timings, neither standalone sample entropy nor Lempel–Ziv beat the package's CPU implementations at the tested batch sizes. Use `complexeva.sample_entropy` and `complexeva.lempel_ziv` for those measures.

## Spectrum-preserving shuffle test

```python
result = gpu.spectrum_shuffle_test(signals, order=2, n_shuffles=39, seed=0)
print(result.observed.shape, result.shuffled.shape)  # (32,), (32, 39)
print(result.pvalue.shape)  # (32,)
```

PyTorch generates the copies by randomizing Fourier phases. The transform preserves the mean and Fourier magnitudes up to roundoff, with the zero-frequency bin and the even-length Nyquist bin unchanged. The sample amplitude distribution can change. The test then uses `complexeva.sample_entropy` on the CPU to score the original samples and each copy. The precision of the transform can affect the copies and their scores.

The lower-tail p-value is `(1 + count(shuffled <= observed)) / (n_shuffles + 1)`, with ties included. A NaN original or surrogate score yields a NaN p-value. See [CPU surrogate tests](surrogates.md) for other measures, amplitude-adjusted surrogates and interpretation.
