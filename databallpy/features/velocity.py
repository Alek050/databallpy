import warnings

import pandas as pd


def get_velocity(
    df: pd.DataFrame, input_columns: list, framerate: float
) -> pd.DataFrame:
    """Function that adds velocity columns based on the position columns

    Args:
        df (pd.DataFrame): tracking data
        input_columns (list): columns for which velocity should be calculated
        framerate (float): framerate of the tracking data

    Returns:
        pd.DataFrame: tracking data with the added velocity columns

    Notes:
        - Pandas PerformanceWarning is ignored. It suggests using df.copy() to
        defragment the memory. However, this is not possible in this case, since
        the dataframe is quite big, taking up a uneccessary memory, and the
        df.copy makes the function about twice as slow.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        for input_column in input_columns:
            velocity_column = f"{input_column}_v"
            df[velocity_column] = df[input_column].diff() * framerate

    return df
