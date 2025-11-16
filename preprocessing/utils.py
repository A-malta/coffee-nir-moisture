import pandas as pd


def get_spectral_columns(df: pd.DataFrame, start_col_index: int = 1) -> list:
    return df.columns[start_col_index:].tolist()


def build_preprocessed_df(df_raw: pd.DataFrame, spectral_cols: list, X_processed, suffix: str) -> pd.DataFrame:
    df_spec = pd.DataFrame(
        X_processed,
        columns=spectral_cols,
        index=df_raw.index
    )
    return pd.concat([df_meta, df_spec], axis=1)
