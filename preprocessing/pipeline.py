from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing.loaders import save_dataset
from preprocessing.transforms import (
    apply_savgol_first_derivative,
    apply_savgol_second_derivative,
    baseline_correction_poly,
    mean_center,
    msc,
    snv,
    area_normalization,
)
from preprocessing.utils import get_spectral_columns, build_preprocessed_df


def load_and_merge(input_spectra: Path, input_moisture: Path) -> pd.DataFrame:
    df_spectra = pd.read_csv(input_spectra)
    df_moist = pd.read_csv(input_moisture)
    df = df_spectra.merge(df_moist, on="sample")
    return df


def plot_spectra(X, cols, title, save_path):
    try:
        wl = np.array([float(c) for c in cols])
    except Exception:
        wl = np.arange(X.shape[1])
    plt.figure(figsize=(8, 5))
    for i in range(X.shape[0]):
        plt.plot(wl, X[i], alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def generate_all_preprocessed_datasets(
    input_spectra: Path,
    input_moisture: Path,
    output_dir: Path,
    window_length: int,
    polyorder: int,
    baseline_degree: int,
    start_col_index: int,
    generate_plots: bool,
):
    datasets_dir = output_dir / "datasets"
    plots_dir = output_dir / "plots"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(input_spectra, input_moisture)
    spectral_cols = get_spectral_columns(df, start_col_index)
    X = df[spectral_cols].values

    X1 = apply_savgol_first_derivative(X, window_length, polyorder)
    X2 = apply_savgol_second_derivative(X, window_length, polyorder)
    Xb = baseline_correction_poly(X, baseline_degree)
    Xm = mean_center(X)
    Xmsc = msc(X)
    Xsnv = snv(X)
    Xa = area_normalization(X)

    outputs = {
        "savgol_1d": X1,
        "savgol_2d": X2,
        "baseline": Xb,
        "mean_center": Xm,
        "msc": Xmsc,
        "snv": Xsnv,
        "area_norm": Xa,
    }

    for name, Xp in outputs.items():
        df_out = build_preprocessed_df(df_raw=df, spectral_cols=spectral_cols, X_processed=Xp)
        save_dataset(df_out, datasets_dir / f"dados_{name}.csv")
        if generate_plots:
            plot_spectra(Xp, spectral_cols, name, plots_dir / f"{name}.png")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    input_spectra = project_root / "data" / "raw" / "dados_brutos.csv"
    input_moisture = project_root / "data" / "raw" / "moisture.csv"
    output_dir = project_root / "output" / "preprocessed"
    generate_all_preprocessed_datasets(
        input_spectra=input_spectra,
        input_moisture=input_moisture,
        output_dir=output_dir,
        window_length=15,
        polyorder=2,
        baseline_degree=2,
        start_col_index=2,
        generate_plots=True,
    )
