from pathlib import Path
from modeling.models import (
    load_dataset,
    ALGORITHMS,
    train_and_save_all_models_for_algorithm,
)
from preprocessing.kennard_stone import kennard_stone
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_DATASETS_DIR = PROJECT_ROOT / "output" / "preprocessed" / "datasets"
MODELS_ROOT_DIR = PROJECT_ROOT / "output" / "models"


def get_preprocessed_csv_paths(pattern="dados_*.csv"):
    return sorted(PREPROCESSED_DATASETS_DIR.glob(pattern))


def extract_dataset_name(csv_path, prefix="dados_"):
    return csv_path.stem.replace(prefix, "")


def split_train_test_val_ks(X, y, val_ratio=0.1, test_ratio=0.1):
    X_np = X.to_numpy()
    n_total = len(X_np)
    
    # 1. Select Validation Set (from Total)
    n_val = max(1, int(val_ratio * n_total))
    val_idx, remaining_idx = kennard_stone(X_np, n_val)
    
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]
    
    X_remaining = X.iloc[remaining_idx]
    y_remaining = y.iloc[remaining_idx]
    X_remaining_np = X_remaining.to_numpy()
    
    # 2. Select Test Set (from Remaining)
    # We want test_ratio of the TOTAL.
    n_test = max(1, int(test_ratio * n_total))
    
    # Ensure we don't ask for more than we have
    if n_test >= len(X_remaining):
        raise ValueError("Not enough samples for Test set after Validation split")
        
    test_idx_rel, train_idx_rel = kennard_stone(X_remaining_np, n_test)
    
    # Map relative indices back to original dataframe is tricky if we use iloc on X_remaining directly
    # simpler to just use the subsets returned by iloc
    X_test = X_remaining.iloc[test_idx_rel]
    y_test = y_remaining.iloc[test_idx_rel]
    
    X_train = X_remaining.iloc[train_idx_rel]
    y_train = y_remaining.iloc[train_idx_rel]
    
    return X_train, X_test, X_val, y_train, y_test, y_val


def load_and_prepare_dataset(csv_path, target_column, val_ratio=0.1, test_ratio=0.1):
    dataset_name = extract_dataset_name(csv_path)
    X, y = load_dataset(csv_path, target_column)
    X_train, X_test, X_val, y_train, y_test, y_val = split_train_test_val_ks(X, y, val_ratio, test_ratio)
    return dataset_name, X_train, X_test, X_val, y_train, y_test, y_val


def train_all_algorithms_for_dataset(
    dataset_name,
    X_train,
    X_test,
    X_val,
    y_train,
    y_test,
    y_val,
    algo_names=None,
    cv_splits=5,
    random_state=42,
):
    algo_names = algo_names or ALGORITHMS.keys()

    for algo_name in tqdm(algo_names, desc=f"Training on {dataset_name}", leave=False):
        output_dir = MODELS_ROOT_DIR / algo_name / dataset_name
        train_and_save_all_models_for_algorithm(
            algo_name=algo_name,
            X_train=X_train,
            X_test=X_test,
            X_val=X_val,
            y_train=y_train,
            y_test=y_test,
            y_val=y_val,
            output_dir=output_dir,
            cv_splits=cv_splits,
            random_state=random_state,
        )


def main(val_ratio=0.1, test_ratio=0.1, cv_splits=5, random_state=42, target_column="moisture"):
    csv_paths = get_preprocessed_csv_paths()
    if not csv_paths:
        return

    for csv_path in tqdm(csv_paths, desc="Datasets"):
        dataset_name, X_train, X_test, X_val, y_train, y_test, y_val = load_and_prepare_dataset(
            csv_path,
            target_column,
            val_ratio,
            test_ratio,
        )
        train_all_algorithms_for_dataset(
            dataset_name,
            X_train,
            X_test,
            X_val,
            y_train,
            y_test,
            y_val,
            cv_splits=None, # Disable CV
            random_state=random_state,
        )


if __name__ == "__main__":
    main()

