import numpy as np
import pandas as pd

from databallpy.utils.constants import MISSING_INT


def _insert_missing_rows(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Functions that inserts missing rows based on gaps in timestamp

    Args:
        df (pd.DataFrame): containing tracking data
        col (str): name of column containing timestamps

    Returns:
        pd.DataFrame: contains tracking data with inserted missing rows
    """
    assert (
        col in df.columns
    ), f"Calculations are based on {col} column, which is not in the df"

    dtypes = df.dtypes
    missing = np.where(df[col].diff() > 1)[0]
    all_missing_timestamps = np.array([])
    for start_missing in missing:
        n_missing = int(df[col].diff()[start_missing] - 1)
        start_timestamp = df.loc[start_missing, col] - n_missing
        missing_timestamps = np.arange(start_timestamp, start_timestamp + n_missing)
        all_missing_timestamps = np.concatenate(
            (all_missing_timestamps, missing_timestamps)
        )

    to_add_data = {
        x: [valid_nan_type(dtypes[x])] * len(all_missing_timestamps) for x in df.columns
    }

    to_add_df = pd.DataFrame(to_add_data)
    to_add_df[col] = all_missing_timestamps
    to_add_df[col] = to_add_df[col].astype(dtypes[col])
    df = pd.concat((df, to_add_df)).sort_values(by=col)
    df.reset_index(drop=True, inplace=True)

    if "datetime" in df.columns:
        df["datetime"] = df["datetime"].astype(dtypes["datetime"])

    return df


def valid_nan_type(dtype):
    if "float" in str(dtype):
        return np.nan
    elif "int" in str(dtype):
        return MISSING_INT
    elif "datetime" in str(dtype):
        return pd.to_datetime("NaT")
    return None
