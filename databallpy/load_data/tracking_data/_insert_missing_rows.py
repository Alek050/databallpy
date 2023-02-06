import numpy as np
import pandas as pd


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

    missing = np.where(df[col].diff() > 1)[0]
    for start_missing in missing:
        n_missing = int(df[col].diff()[start_missing] - 1)
        start_timestamp = df.loc[start_missing, col] - n_missing
        to_add_data = {
            col: list(
                np.arange(start_timestamp, start_timestamp + n_missing, dtype=int)
            )
        }
        to_add_df = pd.DataFrame(to_add_data)
        df = pd.concat((df, to_add_df)).sort_values(by=col)

    df.reset_index(drop=True, inplace=True)

    return df
