import pandas as pd

from ..data.data_path import FEAT_DIR


def export_parquet(file_name: str, df: pd.DataFrame) -> None:
    parquet_path = FEAT_DIR / file_name
    df.to_parquet(parquet_path, index=False)