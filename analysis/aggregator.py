import pandas as pd
from pathlib import Path

def aggregate_and_rank_models(models_root_dir, output_dir):
    models_root_path = Path(models_root_dir)
    output_path = Path(output_dir)
    
    all_metrics_files = list(models_root_path.rglob("all_models_metrics.csv"))
    
    if not all_metrics_files:
        print("No metrics files found to aggregate.")
        return

    df_list = []
    for file_path in all_metrics_files:
        try:
            df = pd.read_csv(file_path)
            parts = file_path.parts
            if len(parts) >= 3:
                dataset_name = parts[-2]
                algo_name = parts[-3]
                df['dataset'] = dataset_name
                df['algorithm'] = algo_name
            
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not df_list:
        print("No valid data found in metrics files.")
        return

    full_df = pd.concat(df_list, ignore_index=True)

    base_cols = ['algorithm', 'dataset', 'model_id']
    metric_cols = [c for c in full_df.columns if c.endswith('_rmse') or c.endswith('_mae') or c.endswith('_r2')]
    param_cols = [c for c in full_df.columns if c.startswith('param_')]
    
    final_cols = base_cols + param_cols + metric_cols
    final_cols = [c for c in final_cols if c in full_df.columns]
    
    report_df = full_df[final_cols]

    rank_train = report_df.sort_values(by="train_rmse", ascending=True)
    rank_train.to_csv(output_path / "ranking_train.csv", index=False)
    print(f"Saved ranking_train.csv to {output_path}")

    rank_test = report_df.sort_values(by="test_rmse", ascending=True)
    rank_test.to_csv(output_path / "ranking_test.csv", index=False)
    print(f"Saved ranking_test.csv to {output_path}")

    rank_val = report_df.sort_values(by="val_rmse", ascending=True)
    rank_val.to_csv(output_path / "ranking_val.csv", index=False)
    print(f"Saved ranking_val.csv to {output_path}")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    MODELS_DIR = PROJECT_ROOT / "output" / "models"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    aggregate_and_rank_models(MODELS_DIR, OUTPUT_DIR)
