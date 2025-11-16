import pandas as pd
import os


def load_raw_dataset(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise IOError(f"Erro ao carregar arquivo: {filepath}\n{e}")


def save_dataset(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        raise IOError(f"Erro ao salvar arquivo: {output_path}\n{e}")
