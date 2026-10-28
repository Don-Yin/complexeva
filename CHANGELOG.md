# changelog

## 0.3.0 (unreleased)

Numerical results of three measures change. To reproduce earlier results, pin `complexeva==0.2.3`.

- Breaking: measures are imported from the top level (`from complexeva import hfd`); `complexeva.sequence` is removed.
- Breaking: removed unexported helpers (`hfd_hilbert_envelope/transform/frequency`, `lempel_ziv_hilbert_envelope/transform/frequency`, `time_series_to_binary`) and the unused `**kwargs` of `lempel_ziv`.

- `hfd`, `hfd_hilbert`: default `k_max` is now 10 (was 64), which gives lower error and better discrimination on series of a few hundred points.
- `hfd`: raises `ValueError` when `k_max` is outside 2 to len(X) // 2 (previously `ZeroDivisionError` or NaN on short series).
- `hfd`: now returns the standard Higuchi fractal dimension D; 0.2.3 returned D - 1 (missing `/k` in L_m(k)) and skipped the last increment of each subsequence.
- `lempel_ziv`: now counts the final unfinished component as in Kaspar & Schuster (1987); 0.2.3 undercounted by 1 on most inputs. New `normalize=True` option returns c*log2(n)/n.
- `lyapunov_exponent`: rewritten as the Rosenstein et al. (1993) largest Lyapunov exponent with a Theiler window; 0.2.3 averaged one-step log distances and depended on signal amplitude. New options `min_tsep`, `trajectory_len`, `fit_offset`.
