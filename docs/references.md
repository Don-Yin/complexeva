# References

Papers consulted while developing 0.3.0. Decisions that rely on them are recorded in [design.md](design.md). Local PDFs are in `docs/papers/<slug>.pdf` and are git-ignored, because most of them are under publisher copyright. The Lempel & Ziv, Kaspar & Schuster and Rosenstein papers have no local copy.

## Method definitions

| Paper | DOI | Used for |
| --- | --- | --- |
| Higuchi 1988, *Physica D* 31: 277-283. Approach to an irregular time series on the basis of the fractal theory | [10.1016/0167-2789(88)90081-4](https://doi.org/10.1016/0167-2789(88)90081-4) | `hfd`: the curve length `L_m(k)` (section 2) matches the implementation exactly |
| Lempel & Ziv 1976, *IEEE Trans. Inf. Theory* 22: 75-81. On the complexity of finite sequences | [10.1109/TIT.1976.1055501](https://doi.org/10.1109/TIT.1976.1055501) | `lempel_ziv`: complexity definition |
| Kaspar & Schuster 1987, *Phys. Rev. A* 36: 842-848. Easily calculable measure for the complexity of spatiotemporal patterns | [10.1103/PhysRevA.36.842](https://doi.org/10.1103/PhysRevA.36.842) | `lempel_ziv`: counting algorithm, including the final unfinished component, and the `c log2(n) / n` normalisation |
| Rosenstein, Collins & De Luca 1993, *Physica D* 65: 117-134. A practical method for calculating largest Lyapunov exponents from small data sets | [10.1016/0167-2789(93)90009-P](https://doi.org/10.1016/0167-2789(93)90009-P) | `lyapunov_exponent`: the algorithm and the Theiler window |

## Choice of `k_max` for `hfd`

| Paper | DOI | Finding relevant to `k_max` |
| --- | --- | --- |
| Accardo, Affinito, Carrozzi & Bouquet 1997, *Biol. Cybern.* 77: 339-350. Use of the fractal dimension for the analysis of electroencephalographic time series | [10.1007/s004220050394](https://doi.org/10.1007/s004220050394) | Weierstrass and fBm segments, N = 50-1000, `k_max` 3-10: the best estimate was at `k_max` = 6 |
| Esteller, Vachtsevanos, Echauz & Litt 2001, *IEEE Trans. Circuits Syst. I* 48: 177-183. A comparison of waveform fractal dimension algorithms | [10.1109/81.904882](https://doi.org/10.1109/81.904882) | Higuchi was the most accurate of the Higuchi, Katz and Petrosian methods on synthetic data, but the most sensitive to noise; it gives no recommendation for `k_max` |
| Porcaro et al. 2020, *Int. J. Neural Syst.* Characterisation of haemodynamic activity in resting state networks by fractal analysis | [10.1142/s0129065720500616](https://doi.org/10.1142/s0129065720500616) | fMRI, 50-volume windows: k = 2-25 was scanned and `k_max` = 12 was used |
| Porcaro et al. 2020, *J. Headache Pain*. Haemodynamic activity characterization of resting state networks by fractal analysis and thalamocortical morphofunctional integrity in chronic migraine | [10.1186/s10194-020-01181-8](https://doi.org/10.1186/s10194-020-01181-8) | Same fMRI approach as above |
| Gomolka et al. 2018, *Front. Physiol.* Higuchi fractal dimension of heart rate variability during percutaneous auricular vagus nerve stimulation in healthy and diabetic subjects | [10.3389/fphys.2018.01162](https://doi.org/10.3389/fphys.2018.01162) | 100-sample windows, `k_max` 2-50 scanned: `k_max` = 5 gave the best group separation |
| Wanliss & Wanliss 2021 preprint (published 2022, *Nonlinear Dyn.*). Efficient calculation of fractal properties via the Higuchi method | [10.21203/rs.3.rs-756319/v1](https://doi.org/10.21203/rs.3.rs-756319/v1), [10.1007/s11071-022-07353-2](https://doi.org/10.1007/s11071-022-07353-2) | A formula for `k_max` as a function of N that minimises mean error on fBm with N >= 1000; variance is not part of the criterion, so it was not adopted |
| Wanliss et al. 2021, *Int. J. Res. Granthaalayah*. Optimization of the Higuchi method | [10.29121/granthaalayah.v9.i11.2021.4393](https://doi.org/10.29121/granthaalayah.v9.i11.2021.4393) | A related cubic-in-ln(N) formula for N from 1000 to 200000, again fitted on mean error only |
| Moaveninejad et al. 2025, *Information Fusion*. Fractal dimension and clinical neurophysiology fusion to gain a deeper brain signal understanding: a systematic review | [10.1016/j.inffus.2025.102936](https://doi.org/10.1016/j.inffus.2025.102936) | Recommends a large K based on Weierstrass signals with lambda = 10; our check shows that, for those signals, the estimate oscillates as K changes (see design.md) |
| Yoder et al. 2024, *medRxiv*. Fractal dimension distributions of resting-state EEG improve detection of dementia and Alzheimer's disease compared to traditional fractal analysis | [10.1101/2024.06.10.24308727](https://doi.org/10.1101/2024.06.10.24308727) | Uses the Wanliss formula (`k_max` = 108 on full EEG recordings, 25 on 1 s windows) |
| Armonaite et al. 2024, *Commun. Appl. Ind. Math.* Insights on neural signal analysis with Higuchi fractal dimension | [10.2478/caim-2024-0009](https://doi.org/10.2478/caim-2024-0009) | A description of the method; it gives no guidance on `k_max` |
| Varley et al. 2020, *PLOS ONE*. Fractal dimension of cortical functional connectivity networks and severity of disorders of consciousness | [10.1371/journal.pone.0223812](https://doi.org/10.1371/journal.pone.0223812) | Context for `hfd_hilbert` (HFD of Hilbert-transformed BOLD); the value of `k_max` is not stated |
