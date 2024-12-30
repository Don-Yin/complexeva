#!/Users/donyin/miniconda3/envs/rotation-1/bin/python


import numpy
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from donware import inspect_package
from numba import njit


def hfd_matlab_equivalent(data, k_max=16):
    """
    [translated from MATLAB code]

    [output]
    - produce values between 1 and 2
    """
    N = len(data)
    L = numpy.zeros(k_max)
    x = numpy.zeros(k_max)
    y = numpy.zeros(k_max)

    for k in range(1, k_max + 1):
        Lk = numpy.zeros(k)
        for m in range(1, k + 1):
            norm_factor = (N - 1) / (numpy.round((N - m) / k) * k)
            X = numpy.sum(numpy.abs(numpy.diff(data[m - 1 :: k])))
            Lk[m - 1] = X * norm_factor / k

        y[k - 1] = numpy.log(numpy.sum(Lk) / k)
        x[k - 1] = numpy.log(1 / k)

    D = numpy.polyfit(x, y, 1)
    HFD = D[0]

    return HFD


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
    x, L = compute_L_x(X, k_max)
    A = numpy.column_stack((x, numpy.ones_like(x)))
    beta, _, _, _ = numpy.linalg.lstsq(A, L, rcond=None)
    return beta[0]
