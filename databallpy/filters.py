import pandas as pd
from scipy.signal import savgol_filter


def filter_data(
    df: pd.DataFrame,
    input_columns: list,
    filter_type: str,
    window_length: int = 5,
    poly_order: int = 2,
) -> pd.DataFrame:
    """Function to filter specified columns, available filters are:
       moving average, savitzky golay

    Args:
        df (pd.DataFrame): dataframe with columns that need to be filtered
        input_columns (list): columns that need to be filtered
        filter_type (str): type of filter to apply. One of:
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

    if not isinstance(filter_type, str):
        raise TypeError(f"filter_type should be a string, not a {type(filter_type)}")

    filter_type_list = ["moving_average", "savitzky_golay"]
    if filter_type not in filter_type_list:
        raise TypeError(
            "filter_type should be one of "
            f"{(', '.join(filter_type_list))}, not {filter_type}"
        )

    if not isinstance(window_length, int):
        raise TypeError(f"window_length should be an int, not a {type(window_length)}")

    if filter_type == "moving_average":
        for col in input_columns:
            df[col] = df[col].rolling(window_length).mean()

    if filter_type == "savitzky_golay":
        for col in input_columns:
            df[col] = savgol_filter(
                df[col], window_length=window_length, polyorder=poly_order
            )

    return df
