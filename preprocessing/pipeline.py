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


def load_and_merge(input_spectra: Path, input_moisture: Path, target_column: str) -> pd.DataFrame:
    df_spectra = pd.read_csv(input_spectra)
    df_moist = pd.read_csv(input_moisture)
    df_moist = _normalize_target_column(df_moist, target_column)
    df = df_spectra.merge(df_moist, on="sample")
    return df


def prepare_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    datasets_dir = output_dir / "datasets"
    plots_dir = output_dir / "plots"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return datasets_dir, plots_dir


def apply_all_transforms(X: np.ndarray, window_length: int, polyorder: int, baseline_degree: int) -> dict[str, np.ndarray]:
    return {
        "savgol_1d": apply_savgol_first_derivative(X, window_length, polyorder),
        "savgol_2d": apply_savgol_second_derivative(X, window_length, polyorder),
        "baseline": baseline_correction_poly(X, baseline_degree),
        "mean_center": mean_center(X),
        "msc": msc(X),
        "snv": snv(X),
        "area_norm": area_normalization(X),
    }


def _build_axis(cols, n_features):
    numerical_axis: list[float] = []
    for c in cols:
        try:
            numerical_axis.append(float(c))
        except (TypeError, ValueError):
            return np.arange(n_features), "Índice das features"
    if numerical_axis:
        start_label, end_label = cols[0], cols[-1]
        axis_label = (
            f"Características espectrais ({start_label} – {end_label})"
            if start_label != end_label
            else f"Característica espectral {start_label}"
        )
        return np.asarray(numerical_axis), axis_label
    return np.arange(n_features), "Características espectrais"


def plot_spectra(X, cols, title, save_path):
    wl, axis_label = _build_axis(cols, X.shape[1])
    plt.figure(figsize=(8, 5))
    for i in range(X.shape[0]):
        plt.plot(wl, X[i], alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.title(title)
    plt.xlabel(axis_label)
    plt.ylabel("Intensidade pré-processada")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_outputs(
    outputs: dict[str, np.ndarray],
    df_raw: pd.DataFrame,
    spectral_cols: list[str],
    datasets_dir: Path,
    plots_dir: Path,
    generate_plots: bool
) -> None:
    for name, Xp in outputs.items():
        df_out = build_preprocessed_df(df_raw, spectral_cols, Xp)
        save_dataset(df_out, datasets_dir / f"dados_{name}.csv")
        if generate_plots:
            plot_spectra(Xp, spectral_cols, name, plots_dir / f"{name}.png")


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
    spectral_cols = get_spectral_columns(df, start_col_index)
    spectral_cols = [col for col in spectral_cols if col != target_column]
    if not spectral_cols:
        raise ValueError(
            "Nenhuma coluna espectral foi encontrada após remover a coluna alvo. "
            "Verifique o parâmetro 'start_col_index' e o nome da coluna alvo."
        )
    X = df[spectral_cols].values
    outputs = apply_all_transforms(X, window_length, polyorder, baseline_degree)
    save_outputs(outputs, df, spectral_cols, datasets_dir, plots_dir, generate_plots)


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

