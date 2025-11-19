import numpy as np
from scipy.signal import savgol_filter


def apply_savgol_first_derivative(
    X: np.ndarray,
    window_length: int,
    polyorder: int,
) -> np.ndarray:
    """Replica exata do pré-tratamento com Savitzky–Golay de 1ª derivada."""

    return savgol_filter(X, window_length, polyorder, deriv=1, axis=1)


def apply_savgol_second_derivative(
    X: np.ndarray,
    window_length: int,
    polyorder: int,
) -> np.ndarray:
    """Replica exata do pré-tratamento com Savitzky–Golay de 2ª derivada."""

    return savgol_filter(X, window_length, polyorder, deriv=2, axis=1)


def baseline_correction_poly(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """Aplica correção de baseline via ajuste polinomial por linha."""

    x = np.arange(X.shape[1])
    corrected = np.array([row - np.polyval(np.polyfit(x, row, degree), x) for row in X])
    return corrected


def mean_center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0)


def msc(X: np.ndarray) -> np.ndarray:
    ref = np.mean(X, axis=0)
    corrected_rows = []
    for row in X:
        slope, intercept = np.polyfit(ref, row, 1)
        corrected_rows.append((row - intercept) / slope)
    return np.array(corrected_rows)


def snv(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


def area_normalization(X: np.ndarray) -> np.ndarray:
    area = np.trapz(X, axis=1).reshape(-1, 1)
    area = np.where(area == 0, 1.0, area)
    return X / area
