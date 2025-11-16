from typing import Sequence
import numpy as np
import pandas as pd


def get_spectral_columns(df: pd.DataFrame, start_col_index: int) -> Sequence[str]:
    return list(df.columns[start_col_index:])


def build_preprocessed_df(
    df_raw: pd.DataFrame,
    spectral_cols: Sequence[str],
    X_processed: np.ndarray,
) -> pd.DataFrame:
    df_meta = df_raw.drop(columns=list(spectral_cols))
    df_spectral = pd.DataFrame(X_processed, columns=spectral_cols, index=df_raw.index)
    df_out = pd.concat([df_meta, df_spectral], axis=1)
    return df_out
