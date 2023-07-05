import pandas as pd
from scipy.signal import savgol_filter


def filter_data(
    df: pd.DataFrame,
    input_columns: list,
    kind: str,
    window_length: int = 5,
    poly_order: int = 2,
) -> pd.DataFrame:
    """Function to filter specified columns, available filters are: moving average

    Args:
        df (pd.DataFrame): dataframe with columns that need to be filtered
        input_columns (list): columns that need to be filtered
        kind (str): kind of filter to apply. One of:
                    "moving_average", "savitzky_golay"
        window_length (int, optional): window length to be used by the filter.
                                       Applies to the following filters:
                                       "moving_average",
                                       "savitzky_golay"
                                       Defaults to 5.
        poly_order (int, optional): order of the polynomial used.
                                    Applies to the following filters:
                                    "savitzky_golay"
    Returns:
        pd.DataFrame: dataframe with filtered columns
    """
    # Check input types
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df should be a pd.DataFrame, not a {type(df)}")

    if not isinstance(input_columns, list):
        raise TypeError(f"input_columns should be a list, not a {type(input_columns)}")

    if not all(isinstance(col, str) for col in input_columns):
        raise TypeError("All elements of input_columns should be strings")

    if not isinstance(kind, str):
        raise TypeError(f"kind should be a string, not a {type(kind)}")

    filter_kind_list = ["moving_average", "savitzky_golay"]
    if kind not in filter_kind_list:
        raise TypeError(
            f"kind should be one of {(', '.join(filter_kind_list))}, not {kind}"
        )

    if not isinstance(window_length, int):
        raise TypeError(f"window_length should be an int, not a {type(window_length)}")

    if kind == "moving_average":
        for col in input_columns:
            df[col] = df[col].rolling(window_length).mean()

    if kind == "savitzky_golay":
        for col in input_columns:
            df[col] = savgol_filter(
                df[col], window_length=window_length, polyorder=poly_order
            )

    return df
