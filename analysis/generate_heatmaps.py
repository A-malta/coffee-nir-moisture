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

def filter_best_models(df):
    if df.empty:
        return df
        
    idx = df.groupby(["algorithm", "preprocessing"])["test_rmse"].idxmin()
    return df.loc[idx].reset_index(drop=True)

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

def generate_heatmap(df, metric):
    pivot_df = df.pivot_table(
        index="algorithm", 
        columns="preprocessing", 
        values=metric, 
        aggfunc="first"
    )
    
    if pivot_df.empty:
        return

    existing_algos = [algo for algo in ALGORITHMS if algo in pivot_df.index]
    pivot_df = pivot_df.reindex(existing_algos)

    plt.figure(figsize=(14, 6))
    
    fmt = ".4f"
    
    sns.heatmap(pivot_df, annot=True, fmt=fmt, cmap="viridis_r", cbar_kws={'label': metric})
    plt.title(f"Best Model Performance: {metric}")
    plt.ylabel("Algorithm")
    plt.xlabel("Preprocessing")
    plt.tight_layout()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / f"{metric}.png", dpi=150)
    plt.close()

def generate_all_heatmaps():
    print("Loading metrics...")
    df = load_all_metrics()
    
    if df.empty:
        print("No metrics found.")
        return

    print(f"Found {len(df)} total models.")
    
    print("Filtering for best models...")
    best_models_df = filter_best_models(df)
    print(f"Retained {len(best_models_df)} best models.")
    
    metrics = get_metrics(best_models_df)
    print(f"Identified metrics: {metrics}")

    for metric in tqdm(metrics, desc="Generating Heatmaps"):
        generate_heatmap(best_models_df, metric)
            
    print(f"Heatmaps saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_all_heatmaps()
