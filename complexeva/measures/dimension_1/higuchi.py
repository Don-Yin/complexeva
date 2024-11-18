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
                Lmk += abs(X[m + i * k] - X[m + (i - 1) * k])
            Lmk *= (N - 1) / (n_max * k)
            Lk[m] = Lmk
        L[k - 1] = numpy.log(Lk.mean())
        x[k - 1] = numpy.log(1.0 / k)
    return x, L


def hfd_pyeeg(X, k_max=16):
    x, L = compute_L_x(X, k_max)
    A = numpy.column_stack((x, numpy.ones_like(x)))
    beta, _, _, _ = numpy.linalg.lstsq(A, L, rcond=None)
    return beta[0]


def _hfd_pyeeg(X, k_max=16):
    """
    [from PyEEG package]
    Higuchi Fractal Dimension of a time series X. kmax is an HFD parameter

    [output]
    - for some reasons produce values between 0 and 1

    [modifications]
    - kmax should be inclusive, Kmax + 1
    """
    L = []
    x = []
    N = len(X)
    for k in range(1, k_max + 1):
        Lk = []
        for m in range(0, k):
            Lmk = 0
            for i in range(1, int(numpy.floor((N - m) / k))):
                Lmk += abs(X[m + i * k] - X[m + i * k - k])
            Lmk = Lmk * (N - 1) / numpy.floor((N - m) / float(k)) / k
            Lk.append(Lmk)
        L.append(numpy.log(numpy.mean(Lk)))
        x.append([numpy.log(float(1) / k), 1])

    (p, _, _, _) = numpy.linalg.lstsq(x, L, rcond=None)
    return p[0]
