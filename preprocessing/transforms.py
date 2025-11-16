import numpy as np
from scipy.signal import savgol_filter


def apply_savgol_first_derivative(
    X: np.ndarray,
    window_length: int,
    polyorder: int,
) -> np.ndarray:
    return savgol_filter(X, window_length=window_length, polyorder=polyorder, deriv=1, axis=1)


def apply_savgol_second_derivative(
    X: np.ndarray,
    window_length: int,
    polyorder: int,
) -> np.ndarray:
    return savgol_filter(X, window_length=window_length, polyorder=polyorder, deriv=2, axis=1)


def baseline_correction_poly(X: np.ndarray, degree: int) -> np.ndarray:
    n_samples, n_features = X.shape
    x_axis = np.arange(n_features)
    X_corrected = np.empty_like(X)
    for i in range(n_samples):
        coeffs = np.polyfit(x_axis, X[i, :], degree)
        baseline = np.polyval(coeffs, x_axis)
        X_corrected[i, :] = X[i, :] - baseline
    return X_corrected


def mean_center(X: np.ndarray) -> np.ndarray:
    mean = np.mean(X, axis=0, keepdims=True)
    return X - mean


def msc(X: np.ndarray) -> np.ndarray:
    reference = np.mean(X, axis=0)
    X_corrected = np.empty_like(X)
    for i in range(X.shape[0]):
        fit = np.polyfit(reference, X[i, :], 1, full=False)
        slope, intercept = fit
        X_corrected[i, :] = (X[i, :] - intercept) / slope
    return X_corrected


def snv(X: np.ndarray) -> np.ndarray:
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


def area_normalization(X: np.ndarray) -> np.ndarray:
    area = np.sum(np.abs(X), axis=1, keepdims=True)
    area = np.where(area == 0, 1.0, area)
    return X / area
