from pathlib import Path

from preprocessing.pipeline import generate_all_preprocessed_datasets
from modeling.train_all_models import main as run_modeling_pipeline


def run_preprocessing(project_root: Path) -> None:
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


def main() -> None:
    project_root = Path(__file__).resolve().parent
    run_preprocessing(project_root)
    run_modeling_pipeline()


if __name__ == "__main__":
    main()
