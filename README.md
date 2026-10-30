# complexeva

Fast, low-memory complexity measures for 1D time series (EEG, MEG, physiology).

```bash
pip install complexeva  # Python 3.14+
```

```python
import numpy
import complexeva

signal = numpy.random.randn(1000)   # one signal, shape (time,)
complexeva.hfd(signal)              # one float
```

## batches

Pass every signal at once with **time on the last axis**; you get one value per signal, in the shape of the other axes. Each value is identical to calling the measure on that signal alone.

```python
eeg = numpy.random.randn(64, 1000)              # (channels, time)
complexeva.hfd(eeg)                             # shape (64,)

study = numpy.random.randn(20, 64, 1000)        # (subjects, channels, time)
complexeva.sample_entropy(study)                # shape (20, 64)

recording = numpy.random.randn(1000, 64)        # time first? put time last
complexeva.dfa(recording.T)                     # shape (64,)

windows = numpy.lib.stride_tricks.sliding_window_view(signal, 200)[::10]  # (windows, 200), no copy
complexeva.permutation_entropy(windows)         # one value per window
```

Batches work for `hfd`, `dfa`, `permutation_entropy`, `sample_entropy` and numeric `lempel_ziv`; the rest take one signal. A non-finite sample raises an error naming its full index; a constant signal gives NaN in its place. `NUMBA_NUM_THREADS` limits the threads.

## measures

| | measure | input | paper |
|---|---|---|---|
| <img src="docs/figures/icons/hfd.png" width="120" alt=""> | Higuchi fractal dimension<br>`hfd`, `hfd_hilbert` | (…, time) numeric; `hfd_hilbert` 1D | [Higuchi 1988](https://doi.org/10.1016/0167-2789(88)90081-4); [Varley 2020](https://doi.org/10.1371/journal.pone.0223812) |
| <img src="docs/figures/icons/lempel_ziv.png" width="120" alt=""> | Lempel–Ziv complexity<br>`lempel_ziv`, `lempel_ziv_hilbert` | (…, time) numeric; Hilbert and binary string 1D | [Kaspar & Schuster 1987](https://doi.org/10.1103/PhysRevA.36.842); [Schartner 2017](https://doi.org/10.1038/srep46421) |
| <img src="docs/figures/icons/sample_entropy.png" width="120" alt=""> | sample entropy<br>`sample_entropy` | (…, time) numeric | [Richman & Moorman 2000](https://doi.org/10.1152/ajpheart.2000.278.6.H2039) |
| <img src="docs/figures/icons/permutation_entropy.png" width="120" alt=""> | permutation entropy<br>`permutation_entropy` | (…, time) numeric | [Bandt & Pompe 2002](https://doi.org/10.1103/PhysRevLett.88.174102) |
| <img src="docs/figures/icons/dfa.png" width="120" alt=""> | detrended fluctuation analysis<br>`dfa` | (…, time) numeric | [Peng 1994](https://doi.org/10.1103/PhysRevE.49.1685) |
| <img src="docs/figures/icons/lyapunov_exponent.png" width="120" alt=""> | largest Lyapunov exponent<br>`lyapunov_exponent` | 1D numeric | [Rosenstein 1993](https://doi.org/10.1016/0167-2789(93)90009-P) |
| <img src="docs/figures/icons/compression_complexity.png" width="120" alt=""> | compression complexity<br>`compression_complexity` | 1D array | [Cilibrasi & Vitányi 2005](https://doi.org/10.1109/TIT.2005.844059) |

## speed and memory vs other packages

![Ratio of each other package's time and peak heap to complexeva's, per measure at N = 1k, 10k and 100k; above 1 favours complexeva.](docs/figures/speed-memory-ratio-table.png)

Batches, against a loop of antropy calls (`NUMBA_NUM_THREADS` sets the thread count):

![Ratio of an antropy loop's time and peak heap to complexeva's batch call for five measures at three shapes, with 16 threads; above 1 favours complexeva.](docs/figures/batch-ratio-table.png)

## validation

![Higuchi fractal dimension follows the theoretical fBm dimension from 1.9 to 1.1 as the Hurst exponent increases.](docs/figures/hfd-fbm.png)

![Normalized Lempel–Ziv complexity is low for a sine wave and near one for white noise, with signal excerpts alongside.](docs/figures/lz-sine-noise.png)

## potential icons

| A | B | C |
|---|---|---|
| <img src="docs/figures/logo/option-a.png" width="200" alt="icon option A"> | <img src="docs/figures/logo/option-b.png" width="200" alt="icon option B"> | <img src="docs/figures/logo/option-c.png" width="200" alt="icon option C"> |

## licence

MIT
