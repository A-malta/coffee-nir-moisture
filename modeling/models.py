from pathlib import Path
import pandas as pd
from joblib import dump
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset(path, target_column):
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {path}")
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_column], errors='ignore')
    y = df[target_column]
    return X, y


def get_param_grid_pls():
    return {"n_components": [2, 4, 8, 12], "scale": [False, True]}


def get_param_grid_rf():
    return {
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True, False],
    }


def get_param_grid_svr():
    return {
        "kernel": ["rbf"],
        "C": [1.0, 10.0, 100.0],
        "gamma": ["scale", "auto"],
        "epsilon": [0.01, 0.1],
    }


def get_param_grid_mlp():
    return {
        "hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "activation": ["relu", "tanh"],
        "alpha": [1e-4, 1e-3],
        "learning_rate_init": [1e-3, 1e-2],
        "max_iter": [5000, 10000],
        "tol": [1e-4, 1e-3],
    }


def make_pls(params):
    return PLSRegression(n_components=params["n_components"], scale=params["scale"])


def make_rf(params):
    return RandomForestRegressor(
        random_state=42,
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        bootstrap=params["bootstrap"],
    )


def make_svr(params):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(
            kernel=params["kernel"],
            C=params["C"],
            gamma=params["gamma"],
            epsilon=params["epsilon"]
        ))
    ])


def make_mlp(params):
    mlp_params = {
        **params,
        "early_stopping": True,
        "n_iter_no_change": 30,
        "random_state": 42,
    }
    mlp_params.setdefault("max_iter", 5000)
    mlp_params.setdefault("tol", 1e-4)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(**mlp_params))
    ])


ALGORITHMS = {
    "pls": {
        "get_param_grid": get_param_grid_pls,
        "make_estimator": make_pls,
    },
    "random_forest": {
        "get_param_grid": get_param_grid_rf,
        "make_estimator": make_rf,
    },
    "svr": {
        "get_param_grid": get_param_grid_svr,
        "make_estimator": make_svr,
    },
    "mlp": {
        "get_param_grid": get_param_grid_mlp,
        "make_estimator": make_mlp,
    },
}





def compute_model_metrics(estimator, X_train, y_train, X_test, y_test, X_val, y_val):
    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)
    y_val_pred = estimator.predict(X_val)
    
    train_rmse = float(mean_squared_error(y_train, y_train_pred, squared=False))
    test_rmse = float(mean_squared_error(y_test, y_test_pred, squared=False))
    val_rmse = float(mean_squared_error(y_val, y_val_pred, squared=False))
    
    return {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "val_rmse": val_rmse,
        "train_mae": float(mean_absolute_error(y_train, y_train_pred)),
        "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
        "val_mae": float(mean_absolute_error(y_val, y_val_pred)),
        "train_bias": float(np.mean(y_train_pred - y_train)),
        "test_bias": float(np.mean(y_test_pred - y_test)),
        "val_bias": float(np.mean(y_val_pred - y_val)),
        "train_rpd": float(np.std(y_train) / train_rmse) if train_rmse > 0 else 0.0,
        "test_rpd": float(np.std(y_test) / test_rmse) if test_rmse > 0 else 0.0,
        "val_rpd": float(np.std(y_val) / val_rmse) if val_rmse > 0 else 0.0,
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "test_r2": float(r2_score(y_test, y_test_pred)),
        "val_r2": float(r2_score(y_val, y_val_pred)),
    }


def train_and_evaluate(estimator, X_train, y_train, X_test, y_test, X_val, y_val):
    estimator.fit(X_train, y_train)
    metrics = compute_model_metrics(estimator, X_train, y_train, X_test, y_test, X_val, y_val)
    return {
        "estimator": estimator,
        **metrics
    }


def generate_param_combinations(grid_func):
    return list(ParameterGrid(grid_func()))





def build_result_row(algo_name, model_id, model_path, params, results):
    row = {
        "algo": algo_name,
        "model_id": model_id,
        "model_path": str(model_path),
        **{k: v for k, v in results.items() if k != "estimator"}
    }
    for k, v in params.items():
        row[f"param_{k}"] = v
    return row


def create_estimator_for_params(algo_cfg, params):
    return algo_cfg["make_estimator"](params)


def train_evaluate_and_save_single(algo_name, params, algo_cfg, idx, X_train, y_train, X_test, y_test, X_val, y_val, output_dir):
    estimator = create_estimator_for_params(algo_cfg, params)
    results = train_and_evaluate(estimator, X_train, y_train, X_test, y_test, X_val, y_val)

    model_path = output_dir / f"{algo_name}_model_{idx:03d}.joblib"
    dump(results["estimator"], model_path)

    return build_result_row(algo_name, idx, model_path, params, results)


def run_param_grid(algo_name, algo_cfg, param_combinations, X_train, y_train, X_test, y_test, X_val, y_val, output_dir):
    rows = []
    for idx, params in enumerate(tqdm(param_combinations, desc=f"Grid Search {algo_name}", leave=False)):
        row = train_evaluate_and_save_single(
            algo_name, params, algo_cfg, idx, X_train, y_train, X_test, y_test, X_val, y_val, output_dir
        )
        rows.append(row)
    return pd.DataFrame(rows)


def train_models_for_param_grid(algo_name, X_train, y_train, X_test, y_test, X_val, y_val, output_dir):
    algo_cfg = ALGORITHMS[algo_name]
    param_combinations = generate_param_combinations(algo_cfg["get_param_grid"])
    df = run_param_grid(
        algo_name,
        algo_cfg,
        param_combinations,
        X_train,
        y_train,
        X_test,
        y_test,
        X_val,
        y_val,
        output_dir,
    )
    return df, len(param_combinations)


def process_results(df, output_dir, algo_name):
    if df.empty:
        return df
    # No sorting or ranking as requested
    df.to_csv(output_dir / "all_models_metrics.csv", index=False)
    return df


def train_and_save_all_models_for_algorithm(
    algo_name,
    X_train,
    X_test,
    X_val,
    y_train,
    y_test,
    y_val,
    output_dir,
):
    if algo_name not in ALGORITHMS:
        raise ValueError(f"Algorithm '{algo_name}' not defined")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    df, param_count = train_models_for_param_grid(
        algo_name, X_train, y_train, X_test, y_test, X_val, y_val, output_dir
    )

    ranked = process_results(df, output_dir, algo_name)
    return ranked


def main():
    target_column = "moisture"
    train_path = "train.csv"
    test_path = "test.csv"

    X_train, y_train = load_dataset(train_path, target_column)
    X_test, y_test = load_dataset(test_path, target_column)

    algo_name = "random_forest"
    output_dir = Path("results") / algo_name

    ranked_results = train_and_save_all_models_for_algorithm(
        algo_name=algo_name,
        X_train=X_train,
        X_test=X_test,
        X_val=X_test, # Just dummy for main
        y_train=y_train,
        y_test=y_test,
        y_val=y_test, # Just dummy for main
        output_dir=output_dir,
    )

    print("Training finished.")
    print(ranked_results.head())


if __name__ == "__main__":
    main()
