import numpy as np
import pandas as pd

from databallpy.utils.constants import MISSING_INT


def _insert_missing_rows(
    df: pd.DataFrame, col: str, *, selection: tuple[int, int] | None = None
) -> pd.DataFrame:
    """Functions that inserts missing rows based on gaps in timestamp

    Args:
        df (pd.DataFrame): containing tracking data
        col (str): name of column containing timestamps
        selection (tuple[int, int], optional): the first and last timestamp that
            are loaded, inclusive. All timestamps within this range are inserted,
            timestamps outside of it are left as is. Defaults to None (only fill
            the gaps between the timestamps in the df).

    Returns:
        pd.DataFrame: contains tracking data with inserted missing rows
    """
    assert (
        col in df.columns
    ), f"Calculations are based on {col} column, which is not in the df"

    dtypes = df.dtypes
    outside_selection = None
    if selection is not None:
        in_selection = df[col].between(*selection)
        outside_selection = df[~in_selection]
        df = df[in_selection].reset_index(drop=True)
        all_missing_timestamps = np.setdiff1d(
            np.arange(selection[0], selection[1] + 1), df[col].values
        )
    else:
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
    to_concat = (df, to_add_df)
    if outside_selection is not None:
        to_concat += (outside_selection,)
    df = pd.concat(to_concat).sort_values(by=col)
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
