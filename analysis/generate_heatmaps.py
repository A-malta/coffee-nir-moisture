import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "output" / "models"
OUTPUT_DIR = PROJECT_ROOT / "output" / "analysis" / "heatmaps"

ALGORITHMS = ["pls", "random_forest", "svr", "mlp"]

def load_all_metrics():
    data = []
    
    for algo in ALGORITHMS:
        algo_dir = MODELS_DIR / algo
        if not algo_dir.exists():
            continue
            
        for preprocess_dir in algo_dir.iterdir():
            if not preprocess_dir.is_dir():
                continue
                
            metrics_file = preprocess_dir / "all_models_metrics.csv"
            if not metrics_file.exists():
                continue
                
            df = pd.read_csv(metrics_file)
            df["algorithm"] = algo
            df["preprocessing"] = preprocess_dir.name
            data.append(df)
            
    if not data:
        return pd.DataFrame()
        
    return pd.concat(data, ignore_index=True)

def get_metrics(df):
    exclude_cols = {
        "algo", "model_id", "model_path", "preprocessing", 
        "rank_by_test_rmse", "algorithm"
    }
    metrics = [
        col for col in df.columns 
        if col not in exclude_cols 
        and not col.startswith("param_")
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    return metrics

def generate_heatmap(df, algo, metric):
    algo_df = df[df["algorithm"] == algo].copy()
    
    if algo_df.empty:
        return

    y_col = "model_id"
    y_label = "Model ID"
    algo_df[y_col] = pd.to_numeric(algo_df[y_col], errors='coerce')
    algo_df = algo_df.sort_values(y_col, ascending=True)
    
    pivot_df = algo_df.pivot_table(
        index=y_col, 
        columns="preprocessing", 
        values=metric, 
        aggfunc="first"
    )
    
    if pivot_df.empty:
        return

    pivot_df = pivot_df.sort_index(ascending=True)

    n_rows = len(pivot_df)
    height = max(8, n_rows * 0.3)

    plt.figure(figsize=(12, height))
    sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="viridis_r", cbar_kws={'label': metric})
    plt.title(f"Heatmap: {metric} - {algo.upper()}")
    plt.ylabel(y_label)
    plt.xlabel("Preprocessing")
    plt.tight_layout()
    
    save_dir = OUTPUT_DIR / algo
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{metric}.png", dpi=150)
    plt.close()

def main():
    print("Loading metrics...")
    df = load_all_metrics()
    
    if df.empty:
        print("No metrics found.")
        return

    print(f"Found {len(df)} records.")
    
    metrics = get_metrics(df)
    print(f"Identified metrics: {metrics}")

    for algo in tqdm(ALGORITHMS, desc="Algorithms"):
        for metric in tqdm(metrics, desc=f"Metrics for {algo}", leave=False):
            generate_heatmap(df, algo, metric)
            
    print(f"Heatmaps saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
