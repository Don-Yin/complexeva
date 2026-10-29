# complexeva

Fast, low-memory complexity measures for 1D time series (EEG, MEG, physiology).

```bash
pip install complexeva  # Python 3.14+
```

```python
import complexeva
complexeva.hfd(signal)  # any 1D array
```

## measures

| | measure | input | paper |
|---|---|---|---|
| <img src="docs/figures/icons/hfd.png" width="120" alt=""> | Higuchi fractal dimension<br>`hfd`, `hfd_hilbert` | 1D numeric | [Higuchi 1988](https://doi.org/10.1016/0167-2789(88)90081-4); [Varley 2020](https://doi.org/10.1371/journal.pone.0223812) |
| <img src="docs/figures/icons/lempel_ziv.png" width="120" alt=""> | Lempel–Ziv complexity<br>`lempel_ziv`, `lempel_ziv_hilbert` | 1D numeric or binary string | [Kaspar & Schuster 1987](https://doi.org/10.1103/PhysRevA.36.842); [Schartner 2017](https://doi.org/10.1038/srep46421) |
| <img src="docs/figures/icons/sample_entropy.png" width="120" alt=""> | sample entropy<br>`sample_entropy` | 1D numeric | [Richman & Moorman 2000](https://doi.org/10.1152/ajpheart.2000.278.6.H2039) |
| <img src="docs/figures/icons/permutation_entropy.png" width="120" alt=""> | permutation entropy<br>`permutation_entropy` | 1D numeric | [Bandt & Pompe 2002](https://doi.org/10.1103/PhysRevLett.88.174102) |
| <img src="docs/figures/icons/dfa.png" width="120" alt=""> | detrended fluctuation analysis<br>`dfa` | 1D numeric | [Peng 1994](https://doi.org/10.1103/PhysRevE.49.1685) |
| <img src="docs/figures/icons/lyapunov_exponent.png" width="120" alt=""> | largest Lyapunov exponent<br>`lyapunov_exponent` | 1D numeric | [Rosenstein 1993](https://doi.org/10.1016/0167-2789(93)90009-P) |
| <img src="docs/figures/icons/compression_complexity.png" width="120" alt=""> | compression complexity<br>`compression_complexity` | 1D array | [Cilibrasi & Vitányi 2005](https://doi.org/10.1109/TIT.2005.844059) |

## speed and memory vs other packages

![Ratio of each other package's time and peak heap to complexeva's, per measure at N = 1k, 10k and 100k; above 1 favours complexeva.](docs/figures/speed-memory-ratio-table.png)

## validation

![Higuchi fractal dimension follows the theoretical fBm dimension from 1.9 to 1.1 as the Hurst exponent increases.](docs/figures/hfd-fbm.png)

![Normalized Lempel–Ziv complexity is low for a sine wave and near one for white noise, with signal excerpts alongside.](docs/figures/lz-sine-noise.png)

## potential icons

| A | B | C |
|---|---|---|
| <img src="docs/figures/logo/option-a.png" width="200" alt="icon option A"> | <img src="docs/figures/logo/option-b.png" width="200" alt="icon option B"> | <img src="docs/figures/logo/option-c.png" width="200" alt="icon option C"> |

## licence

MIT
