# Batches

Batches work for every measure except `lyapunov_exponent`, `compression_complexity`, the `_hilbert` variants and binary-string `lempel_ziv`, which take one signal. `multiscale_entropy` adds a last axis of scales, and `hjorth` returns a (mobility, complexity) pair of arrays. A non-finite sample raises an error naming its full index; a constant signal gives NaN in its place. `NUMBA_NUM_THREADS` limits the threads.
