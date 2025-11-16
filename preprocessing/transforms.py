import numpy as np
from scipy.signal import savgol_filter
from numpy.polynomial import Polynomial


def apply_savgol_first_derivative(X, window_length=15, polyorder=2, delta=1):
    return savgol_filter(X, window_length, polyorder, deriv=1, delta=delta, axis=1)


def apply_savgol_second_derivative(X, window_length=15, polyorder=2, delta=1):
    return savgol_filter(X, window_length, polyorder, deriv=2, delta=delta, axis=1)


def baseline_correction_poly(X, degree=2):
    n_samples, n_vars = X.shape
    x_axis = np.arange(n_vars)
    X_corrected = np.zeros_like(X)

    for i in range(n_samples):
        y = X[i, :]
        coefs = Polynomial.fit(x_axis, y, deg=degree).convert().coef
        baseline = np.polyval(coefs[::-1], x_axis)
        X_corrected[i, :] = y - baseline

    return X_corrected


def mean_center(X):
    mean_vec = X.mean(axis=0)
    return X - mean_vec


def msc(X):
    ref = X.mean(axis=0)
    X_corr = np.zeros_like(X)

    for i in range(X.shape[0]):
        y = X[i, :]
        a, b = np.polyfit(ref, y, 1)
        X_corr[i, :] = (y - b) / a

    return X_corr


def snv(X):
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    return (X - means) / stds


def area_normalization(X):
    areas = X.sum(axis=1, keepdims=True)
    return X / areas
