from pathlib import Path
from modeling.models import (
    load_dataset,
    ALGORITHMS,
    train_and_save_all_models_for_algorithm,
)
from preprocessing.kennard_stone import kennard_stone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_DATASETS_DIR = PROJECT_ROOT / "output" / "preprocessed" / "datasets"
MODELS_ROOT_DIR = PROJECT_ROOT / "output" / "models"


def get_preprocessed_csv_paths(pattern="dados_*.csv"):
    return sorted(PREPROCESSED_DATASETS_DIR.glob(pattern))


def extract_dataset_name(csv_path, prefix="dados_"):
    return csv_path.stem.replace(prefix, "")


def split_train_val_kennard_stone(X, y, val_ratio=0.2):
    X_np = X.to_numpy()
    n_val = max(1, int(val_ratio * len(X_np)))
    val_idx, train_idx = kennard_stone(X_np, n_val)
    return (
        X.iloc[train_idx],
        X.iloc[val_idx],
        y.iloc[train_idx],
        y.iloc[val_idx],
    )


def load_and_prepare_dataset(csv_path, target_column, val_ratio=0.2):
    dataset_name = extract_dataset_name(csv_path)
    X, y = load_dataset(csv_path, target_column)
    X_train, X_val, y_train, y_val = split_train_val_kennard_stone(X, y, val_ratio)
    return dataset_name, X_train, X_val, y_train, y_val


def train_all_algorithms_for_dataset(
    dataset_name,
    X_train,
    X_val,
    y_train,
    y_val,
    algo_names=None,
    cv_splits=5,
    random_state=42,
):
    algo_names = algo_names or ALGORITHMS.keys()

    for algo_name in algo_names:
        output_dir = MODELS_ROOT_DIR / algo_name / dataset_name
        train_and_save_all_models_for_algorithm(
            algo_name=algo_name,
            X_train=X_train,
            X_test=X_val,
            y_train=y_train,
            y_test=y_val,
            output_dir=output_dir,
            cv_splits=cv_splits,
            random_state=random_state,
        )


def main(val_ratio=0.2, cv_splits=5, random_state=42, target_column="moisture"):
    csv_paths = get_preprocessed_csv_paths()
    if not csv_paths:
        return

    for csv_path in csv_paths:
        dataset_name, X_train, X_val, y_train, y_val = load_and_prepare_dataset(
            csv_path,
            target_column,
            val_ratio,
        )
        train_all_algorithms_for_dataset(
            dataset_name,
            X_train,
            X_val,
            y_train,
            y_val,
            cv_splits=cv_splits,
            random_state=random_state,
        )


if __name__ == "__main__":
    main()

