from preprocessing.loaders import load_raw_dataset, save_dataset
from preprocessing.transforms import (
    apply_savgol_first_derivative, apply_savgol_second_derivative,
    baseline_correction_poly, mean_center, msc, snv, area_normalization
)
from preprocessing.utils import get_spectral_columns, build_preprocessed_df
import os


def generate_all_preprocessed_datasets(
    input_csv,
    output_dir,
    window_length=15,
    polyorder=2,
    baseline_degree=2
):
    os.makedirs(output_dir, exist_ok=True)

    df_raw = load_raw_dataset(input_csv)
    spectral_cols = get_spectral_columns(df_raw)
    X = df_raw[spectral_cols].to_numpy(dtype=float)

    preprocessors = {
        "savgol_1d": lambda X: apply_savgol_first_derivative(X, window_length, polyorder),
        "savgol_2d": lambda X: apply_savgol_second_derivative(X, window_length, polyorder),
        "baseline": lambda X: baseline_correction_poly(X, baseline_degree),
        "mean_center": mean_center,
        "msc": msc,
        "snv": snv,
        "area_norm": area_normalization
    }

    for name, func in preprocessors.items():
        X_proc = func(X)
        df_proc = build_preprocessed_df(df_raw, spectral_cols, X_proc, name)
        output_path = os.path.join(output_dir, f"dados_{name}.csv")
        save_dataset(df_proc, output_path)
        print(f" Salvo: {output_path}")


if __name__ == "__main__":
    generate_all_preprocessed_datasets(
        input_csv="data/raw/dados_brutos.csv",
        output_dir="output/preprocessed",
        window_length=15,
        polyorder=2,
        baseline_degree=2
    )
