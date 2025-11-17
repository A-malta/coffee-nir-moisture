from pathlib import Path
import json
import pandas as pd
from joblib import dump
import numpy as np

from sklearn.model_selection import KFold, ParameterGrid, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset(path, target_column):
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Coluna alvo '{target_column}' não encontrada em {path}")
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
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            activation=params["activation"],
            alpha=params["alpha"],
            learning_rate_init=params["learning_rate_init"],
            max_iter=2000,
            random_state=42,
        ))
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


def train_model(estimator, X_train, y_train):
    estimator.fit(X_train, y_train)
    return estimator


def evaluate_with_cross_validation(estimator, X, y, cv):
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
    return {
        "cv_mean_neg_mse": float(scores.mean()),
        "cv_std_neg_mse": float(scores.std())
    }


def compute_model_metrics(estimator, X_train, y_train, X_test, y_test):
    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)
    return {
        "train_rmse": float(mean_squared_error(y_train, y_train_pred, squared=False)),
        "test_rmse": float(mean_squared_error(y_test, y_test_pred, squared=False)),
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "test_r2": float(r2_score(y_test, y_test_pred)),
    }


def train_and_evaluate(estimator, X_train, y_train, X_test, y_test, cv):
    cv_scores = evaluate_with_cross_validation(estimator, X_train, y_train, cv)
    trained_model = train_model(estimator, X_train, y_train)
    metrics = compute_model_metrics(trained_model, X_train, y_train, X_test, y_test)
    return {
        "estimator": trained_model,
        **cv_scores,
        **metrics
    }


def generate_param_combinations(grid_func):
    return list(ParameterGrid(grid_func()))


def generate_cv(n_splits, random_state):
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def save_model(model, path):
    dump(model, path)


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


def train_evaluate_and_save_single(algo_name, params, algo_cfg, idx, X_train, y_train, X_test, y_test, cv, output_dir):
    estimator = create_estimator_for_params(algo_cfg, params)
    results = train_and_evaluate(estimator, X_train, y_train, X_test, y_test, cv)

    model_path = output_dir / f"{algo_name}_model_{idx:03d}.joblib"
    save_model(results["estimator"], model_path)

    return build_result_row(algo_name, idx, model_path, params, results)


def run_param_grid(algo_name, algo_cfg, param_combinations, X_train, y_train, X_test, y_test, cv, output_dir):
    rows = []
    for idx, params in enumerate(param_combinations):
        row = train_evaluate_and_save_single(
            algo_name, params, algo_cfg, idx, X_train, y_train, X_test, y_test, cv, output_dir
        )
        rows.append(row)
    return pd.DataFrame(rows)


def train_models_for_param_grid(algo_name, X_train, y_train, X_test, y_test, output_dir, cv):
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
        cv,
        output_dir,
    )
    return df, len(param_combinations)


def save_metrics_csv(df, path):
    df.to_csv(path, index=False)


def save_summary(summary_dict, path):
    with open(path, "w") as f:
        json.dump(summary_dict, f, indent=4)


def rank_results_by_rmse(df):
    if df.empty:
        return df
    df = df.sort_values(by="test_rmse")
    df["rank_by_test_rmse"] = range(1, len(df) + 1)
    return df


def build_summary(df, algo_name, param_count):
    return {
        "algorithm": algo_name,
        "n_models": param_count,
        "best_by_test_rmse": df.iloc[0].to_dict() if not df.empty else None,
    }


def process_results(df, output_dir, algo_name, param_count):
    ranked = rank_results_by_rmse(df)
    save_metrics_csv(ranked, output_dir / "all_models_metrics.csv")
    summary = build_summary(ranked, algo_name, param_count)
    save_summary(summary, output_dir / "summary.json")
    return ranked


def train_and_save_all_models_for_algorithm(
    algo_name,
    X_train,
    X_test,
    y_train,
    y_test,
    output_dir,
    cv_splits=5,
    random_state=42,
):
    if algo_name not in ALGORITHMS:
        raise ValueError(f"Algoritmo '{algo_name}' não definido")

    output_dir.mkdir(parents=True, exist_ok=True)
    cv = generate_cv(cv_splits, random_state)

    df, param_count = train_models_for_param_grid(
        algo_name, X_train, y_train, X_test, y_test, output_dir, cv
    )

    ranked = process_results(df, output_dir, algo_name, param_count)
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
        y_train=y_train,
        y_test=y_test,
        output_dir=output_dir,
        cv_splits=5,
        random_state=42,
    )

    print("Treinamento finalizado.")
    print(ranked_results.head())


if __name__ == "__main__":
    main()
