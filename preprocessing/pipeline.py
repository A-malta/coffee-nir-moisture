from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing.loaders import save_dataset
from preprocessing.transforms import (
    apply_savgol_first_derivative,
    apply_savgol_second_derivative,
    area_normalization,
    baseline_correction_poly,
    mean_center,
    msc,
    snv,
)
from preprocessing.utils import build_preprocessed_df


def _normalize_target_column(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if target_column in df.columns:
        return df

    normalized_target = target_column.strip().lower().replace(" ", "_")
    for col in df.columns:
        normalized = col.strip().lower().replace(" ", "_")
        if normalized.startswith(normalized_target):
            return df.rename(columns={col: target_column})
    raise ValueError(
        f"Coluna alvo '{target_column}' não encontrada. Colunas disponíveis: {list(df.columns)}"
    )


def _clean_sample_column(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "", regex=False)
    )


def _ensure_original_moisture_column(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if "Moisture W.B" in df.columns:
        return df
    if target_column not in df.columns:
        raise ValueError("Coluna de umidade não encontrada após normalização")
    df["Moisture W.B"] = df[target_column]
    return df


def load_and_merge(input_spectra: Path, input_moisture: Path, target_column: str) -> pd.DataFrame:
    df_spectra = pd.read_csv(input_spectra)
    df_moist = pd.read_csv(input_moisture)
    df_spectra["sample"] = _clean_sample_column(df_spectra["sample"])
    df_moist["sample"] = _clean_sample_column(df_moist["sample"])
    df_moist = _normalize_target_column(df_moist, target_column)
    df_moist = _ensure_original_moisture_column(df_moist, target_column)
    moisture_cols = ["sample", "Moisture W.B"]
    if target_column != "Moisture W.B":
        moisture_cols.append(target_column)
    df = pd.merge(df_spectra, df_moist[moisture_cols], on="sample", how="left")
    return df


def prepare_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    datasets_dir = output_dir / "datasets"
    plots_dir = output_dir / "plots"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return datasets_dir, plots_dir


def _slugify(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_").lower()
    return normalized or "preprocessado"


def _coerce_wavelengths(columns: pd.Index | list[str]) -> np.ndarray:
    try:
        return columns.astype(float)
    except Exception:
        return np.arange(len(columns))


def _plot_spectra(X_proc: np.ndarray, wavelengths: np.ndarray, y: np.ndarray, title: str, save_path: Path) -> None:
    if np.isnan(y).all():
        vmin, vmax = 0.0, 1.0
    else:
        vmin = np.nanmin(y)
        vmax = np.nanmax(y)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
    denom = vmax - vmin if vmax != vmin else 1.0

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(X_proc.shape[0]):
        value = 0.5 if not np.isfinite(y[i]) else (y[i] - vmin) / denom
        ax.plot(
            wavelengths,
            X_proc[i, :],
            color=plt.cm.coolwarm(np.clip(value, 0, 1)),
            alpha=0.8,
        )

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Umidade (%)")
    ax.grid(False)
    ax.set_xlabel("Comprimento de onda (nm)")
    ax.set_ylabel("Absorbância")
    ax.set_title(f"Espectros - {title}")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def apply_all_transforms(
    X: np.ndarray,
    window_length: int,
    polyorder: int,
    baseline_degree: int,
) -> dict[str, np.ndarray]:
    return {
        "Bruto": X.copy(),
        "Savitzky–Golay 1ª Derivada": apply_savgol_first_derivative(X, window_length, polyorder),
        "Savitzky–Golay 2ª Derivada": apply_savgol_second_derivative(X, window_length, polyorder),
        "Baseline Correction": baseline_correction_poly(X, baseline_degree),
        "Mean Centering": mean_center(X),
        "MSC": msc(X),
        "SNV": snv(X),
        "Area Normalization": area_normalization(X),
    }


def _prepare_feature_matrix(df: pd.DataFrame, target_column: str) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    meta_columns = {"sample", "replicate", "Moisture W.B"}
    if target_column != "Moisture W.B":
        meta_columns.add(target_column)

    spectral_cols = [col for col in df.columns if col not in meta_columns]
    X = df[spectral_cols].to_numpy(dtype=float)
    y = df["Moisture W.B"].to_numpy(dtype=float)
    return X, y, pd.Index(spectral_cols)


def save_outputs(
    outputs: dict[str, np.ndarray],
    df_raw: pd.DataFrame,
    spectral_cols: pd.Index,
    datasets_dir: Path,
    plots_dir: Path,
    generate_plots: bool,
    y: np.ndarray,
) -> None:
    spectral_cols_list = list(spectral_cols)
    wavelengths = _coerce_wavelengths(pd.Index(spectral_cols_list))
    for name, Xp in outputs.items():
        df_out = build_preprocessed_df(df_raw, spectral_cols_list, Xp)
        slug = _slugify(name)
        save_dataset(df_out, datasets_dir / f"dados_{slug}.csv")
        if generate_plots:
            _plot_spectra(Xp, wavelengths, y, name, plots_dir / f"{slug}.png")


def generate_all_preprocessed_datasets(
    input_spectra: Path,
    input_moisture: Path,
    output_dir: Path,
    window_length: int,
    polyorder: int,
    baseline_degree: int,
    start_col_index: int,
    generate_plots: bool,
    target_column: str,
):
    datasets_dir, plots_dir = prepare_output_dirs(output_dir)
    df = load_and_merge(input_spectra, input_moisture, target_column)
    X, y, spectral_cols = _prepare_feature_matrix(df, target_column)
    outputs = apply_all_transforms(X, window_length, polyorder, baseline_degree)
    save_outputs(outputs, df, spectral_cols, datasets_dir, plots_dir, generate_plots, y)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    input_spectra = project_root / "data" / "raw" / "dados_brutos.csv"
    input_moisture = project_root / "data" / "raw" / "moisture.csv"
    output_dir = project_root / "output" / "preprocessed"
    target_column = "moisture"

    generate_all_preprocessed_datasets(
        input_spectra=input_spectra,
        input_moisture=input_moisture,
        output_dir=output_dir,
        window_length=15,
        polyorder=2,
        baseline_degree=2,
        start_col_index=2,
        generate_plots=True,
        target_column=target_column,
    )

