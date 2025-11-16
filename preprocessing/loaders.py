from pathlib import Path
import pandas as pd


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_dataset(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
