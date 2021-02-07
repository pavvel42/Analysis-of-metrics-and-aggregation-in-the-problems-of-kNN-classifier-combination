#import statistics https://www.geeksforgeeks.org/python-statistics-harmonic_mean/
import numpy as np
from scipy.stats import hmean

def amean(a, axis=0, dtype=None):
    if not isinstance(a, np.ndarray):
        a = np.array(a, dtype=dtype)
    if isinstance(a, np.ma.MaskedArray):
        size = a.count(axis)
    else:
        if axis is None:
            a = a.ravel()
            size = a.shape[0]
        else:
            size = a.shape[axis]
    with np.errstate(divide='ignore'):
        return np.sum(a, axis=axis, dtype=dtype) / size


def qmean(a, axis=0, dtype=None):
    if not isinstance(a, np.ndarray):
        a = np.array(a, dtype=dtype)
    if isinstance(a, np.ma.MaskedArray):
        size = a.count(axis)
    else:
        if axis is None:
            a = a.ravel()
            size = a.shape[0]
        else:
            size = a.shape[axis]
    with np.errstate(divide='ignore'):
        return np.sqrt(np.sum(a ** 2, axis=axis, dtype=dtype) / size)


def gmean(a, axis=0, dtype=None):
    if not isinstance(a, np.ndarray):
        # if not an ndarray object attempt to convert it
        a = np.array(a, dtype=dtype)

    if isinstance(a, np.ma.MaskedArray):
        size = a.count(axis)
    else:
        if axis is None:
            a = a.ravel()
            size = a.shape[0]
        else:
            size = a.shape[axis]

    wk = 1 / size
    with np.errstate(divide='ignore'):
        return np.prod(a ** wk, axis=axis, dtype=dtype)

# Testy agregacji z wynikami pliku excel
# T = np.array([[0.8, 0.2, 0.7], [1, 0.05, 0.1], [1, 1, 1], [0.3, 0.3, 0.3], [0.5, 0.5, 0.5], [0.1, 0.9, 0.6],
#               [0.2, 0.8, 0.7], [0.3, 0.7, 0.8], [0.4, 0.6, 0.9], [0.5, 0.5, 1], [0.6, 0.4, 0.1], [0.7, 0.3, 0.2],
#               [0.8, 0.2, 0.3], [0.9, 0.1, 0.4], [1, 0.5, 0.5], [1, 3, 5], [2, 6, 5]])
#
# print("[numpy] Arithmetic mean is % s " % (
#     np.mean(T, axis=0)))  # https://numpy.org/devdocs/reference/generated/numpy.mean.html#numpy.mean
# print("[scipy.stats] Arithmetic mean is % s " % (amean(T)))
# print("[scipy.stats] Geometric mean is % s " % (
#     gmean(T)))  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gmean.html#scipy.stats.gmean
# print("[scipy.stats] Harmonic mean is % s " % (
#     hmean(T)))  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hmean.html#scipy.stats.hmean
# print("[scipy.stats] Quadratic mean is % s " % (qmean(T)))
#
# for i in T:
#     print("Row ", i)
#     print("[numpy] Arithmetic mean is % s " % (np.mean(i, axis=0)))
#     print("[scipy.stats] Arithmetic mean is % s " % (amean(i)))
#     print("[scipy.stats] Geometric mean is % s " % (gmean(i)))
#     print("[scipy.stats] Harmonic mean is % s " % (hmean(i)))
#     print("[scipy.stats] Quadratic mean is % s " % (qmean(i)))
