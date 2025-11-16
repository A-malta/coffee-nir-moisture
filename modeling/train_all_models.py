from pathlib import Path

from modeling.models import load_dataset, ALGORITHMS, train_and_save_all_models_for_algorithm
from preprocessing.kennard_stone import kennard_stone


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_DATASETS_DIR = PROJECT_ROOT / "output" / "preprocessed" / "datasets"
MODELS_ROOT_DIR = PROJECT_ROOT / "output" / "models"


def main():
    csv_paths = sorted(PREPROCESSED_DATASETS_DIR.glob("dados_*.csv"))
    if not csv_paths:
        return
    for csv_path in csv_paths:
        dataset_name = csv_path.stem.replace("dados_", "")
        X, y = load_dataset(csv_path)
        X_np = X.values
        n_val = max(1, int(0.2 * len(X_np)))
        val_idx, train_idx = kennard_stone(X_np, n_val)
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        for algo_name in ALGORITHMS.keys():
            output_dir = MODELS_ROOT_DIR / algo_name / dataset_name
            train_and_save_all_models_for_algorithm(
                algo_name=algo_name,
                X_train=X_train,
                X_test=X_val,
                y_train=y_train,
                y_test=y_val,
                output_dir=output_dir,
                cv_splits=5,
                random_state=42,
            )


if __name__ == "__main__":
    main()
