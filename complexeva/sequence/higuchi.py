#!/Users/donyin/miniconda3/envs/rotation-1/bin/python


import numpy
from numba import njit


@njit
def compute_L_x(X, k_max=16):
    N = len(X)
    L = numpy.zeros(k_max)
    x = numpy.zeros(k_max)
    for k in range(1, k_max + 1):
        Lk = numpy.zeros(k)
        for m in range(k):
            Lmk = 0.0
            n_max = (N - m) // k
            for i in range(1, n_max):
                Lmk += numpy.abs(X[m + i * k] - X[m + (i - 1) * k])
            Lmk *= (N - 1) / (n_max * k)
            Lk[m] = Lmk
        L[k - 1] = numpy.log(Lk.mean())
        x[k - 1] = numpy.log(1.0 / k)
    return x, L


def hfd(X, k_max=16):
    """expect a 1d numpy array as input"""
    x, L = compute_L_x(X, k_max)
    A = numpy.column_stack((x, numpy.ones_like(x)))
    beta, _, _, _ = numpy.linalg.lstsq(A, L, rcond=None)
    return beta[0]


if __name__ == "__main__":
    signal = numpy.random.rand(1000)  # shape: (1000,)
    complexity = hfd(signal)
    print(complexity)
