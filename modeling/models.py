from pathlib import Path
import json
from typing import Dict, Any

import pandas as pd
from joblib import dump

from sklearn.model_selection import KFold, ParameterGrid, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "moisture"


def load_dataset(path: Path):
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Coluna alvo '{TARGET_COLUMN}' não encontrada em {path}")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def get_param_grid_pls() -> Dict[str, list]:
    return {
        "n_components": [2, 4, 8, 12],
        "scale": [False, True],
    }


def make_pls(params: Dict[str, Any]):
    return PLSRegression(
        n_components=params["n_components"],
        scale=params["scale"],
    )


def get_param_grid_rf() -> Dict[str, list]:
    return {
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True, False],
    }


def make_rf(params: Dict[str, Any]):
    return RandomForestRegressor(
        random_state=42,
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        bootstrap=params["bootstrap"],
    )


def get_param_grid_svr() -> Dict[str, list]:
    return {
        "kernel": ["rbf"],
        "C": [1.0, 10.0, 100.0],
        "gamma": ["scale", "auto"],
        "epsilon": [0.01, 0.1],
    }


def make_svr(params: Dict[str, Any]):
    svr = SVR(
        kernel=params["kernel"],
        C=params["C"],
        gamma=params["gamma"],
        epsilon=params["epsilon"],
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", svr),
        ]
    )


def get_param_grid_mlp() -> Dict[str, list]:
    return {
        "hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "activation": ["relu", "tanh"],
        "alpha": [1e-4, 1e-3],
        "learning_rate_init": [1e-3, 1e-2],
    }


def make_mlp(params: Dict[str, Any]):
    mlp = MLPRegressor(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        activation=params["activation"],
        alpha=params["alpha"],
        learning_rate_init=params["learning_rate_init"],
        max_iter=2000,
        random_state=42,
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", mlp),
        ]
    )


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


def train_and_save_all_models_for_algorithm(
    algo_name: str,
    X_train,
    X_test,
    y_train,
    y_test,
    output_dir: Path,
    cv_splits: int = 5,
    random_state: int = 42,
):
    if algo_name not in ALGORITHMS:
        raise ValueError(f"Algoritmo '{algo_name}' não definido")
    cfg = ALGORITHMS[algo_name]
    param_grid = cfg["get_param_grid"]()
    make_estimator = cfg["make_estimator"]
    output_dir.mkdir(parents=True, exist_ok=True)
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    param_combinations = list(ParameterGrid(param_grid))
    rows = []
    for idx, params in enumerate(param_combinations):
        estimator = make_estimator(params)
        cv_scores = cross_val_score(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        estimator.fit(X_train, y_train)
        y_train_pred = estimator.predict(X_train)
        y_test_pred = estimator.predict(X_test)
        train_rmse = float(mean_squared_error(y_train, y_train_pred, squared=False))
        test_rmse = float(mean_squared_error(y_test, y_test_pred, squared=False))
        train_r2 = float(r2_score(y_train, y_train_pred))
        test_r2 = float(r2_score(y_test, y_test_pred))
        model_filename = f"{algo_name}_model_{idx:03d}.joblib"
        model_path = output_dir / model_filename
        dump(estimator, model_path)
        row = {
            "algo": algo_name,
            "model_id": idx,
            "model_path": str(model_path),
            "cv_mean_neg_mse": cv_mean,
            "cv_std_neg_mse": cv_std,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_r2": train_r2,
            "test_r2": test_r2,
        }
        for k, v in params.items():
            row[f"param_{k}"] = v
        rows.append(row)
    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        results_df = results_df.sort_values(by="test_rmse", ascending=True)
        results_df["rank_by_test_rmse"] = range(1, len(results_df) + 1)
    results_csv_path = output_dir / "all_models_metrics.csv"
    results_df.to_csv(results_csv_path, index=False)
    summary = {
        "algorithm": algo_name,
        "n_models": len(param_combinations),
        "best_by_test_rmse": results_df.iloc[0].to_dict() if not results_df.empty else None,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    return results_df
