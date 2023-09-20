import numpy as np
import pandas as pd

from databallpy.utils.filters import filter_data
from databallpy.utils.utils import MISSING_INT


def get_velocity(
    df: pd.DataFrame,
    input_columns: list,
    framerate: float,
    filter_type: str = None,
    window: int = 7,
    poly_order: int = 2,
) -> pd.DataFrame:
    """Function that adds velocity columns based on the position columns

    Args:
        df (pd.DataFrame): tracking data
        input_columns (list): columns for which velocity should be calculated
        framerate (float): framerate of the tracking data
        filter_type (str, optional): filter type to use. Defaults to None.
        window (int, optional): window size for the filter. Defaults to 7.
        poly_order (int, optional): polynomial order for the filter. Defaults to 2.

    Returns:
        pd.DataFrame: tracking data with the added velocity columns
    """

    if filter_type not in ["moving_average", "savitzky_golay", None]:
        raise ValueError(
            "filter_type should be one of: 'moving_average', "
            "'savitzky_golay', None, got: {filter_type}"
        )

    res_df = _differentiate(
        df,
        new_name="velocity",
        metric="",
        frame_rate=framerate,
        filter_type=filter_type,
        window=window,
        poly_order=poly_order,
        column_ids=input_columns,
    )

    return res_df


def _differentiate(
    df: pd.DataFrame,
    *,
    new_name: str,
    metric: str = "",
    frame_rate: int = 25,
    filter_type: str = "savitzky_golay",
    window: int = 7,
    max_val: int = MISSING_INT,
    poly_order: int = 2,
    column_ids: list = None,
):
    """
    Function to differentiate the metric in x and y direction and update the df with the
    differentiated values in x and y direction and the absolute magnitude.

    :param df: pandas df with position data in x and y direction of players and ball
    :param metric: str, over what metric to differentiate the value on, note that
    f"{player}_{metrix}x" and y should exist.
    :param new_name: str, name of the magnitude, first letter will be used for the x and
    y dirctions: f"{player}_vx" and f"{player}_velocity" if new_name = "velocity"
    :param Fs: int, sample frequency of the data
    :param filter_type: str, which filter to use:
                        {"moving average", "savitzky_golay", None}
    :param window: int, the window of the filter
    :param max_val: float, max value of the differentiated value, for instance, a speed
    of higher than 12 m/s is very unlikely.
    :param polyorder: int, polynomial for the Savitzky-Golay filter
    :returns: pandas df with added differentiated values
    """

    to_skip = len(metric) + 2
    if column_ids is None:
        column_ids = [x[:-to_skip] for x in df.columns if f"_{metric}x" in x]

    dt = 1.0 / frame_rate

    res_dict = {}
    for player in column_ids:
        diff_x = df[player + f"_{metric}x"].diff() / dt
        diff_y = df[player + f"_{metric}y"].diff() / dt

        # remove outliers
        raw_differentiated = np.linalg.norm([diff_x, diff_y], axis=0)
        if max_val != MISSING_INT:
            diff_x[(raw_differentiated > max_val) & (diff_x > max_val)] = max_val
            diff_x[(raw_differentiated > max_val) & (diff_x < -max_val)] = -max_val
            diff_y[(raw_differentiated > max_val) & (diff_y > max_val)] = max_val
            diff_y[(raw_differentiated > max_val) & (diff_y < -max_val)] = -max_val

        # smoothing the signal
        if filter_type is not None:
            diff_x = filter_data(
                diff_x.values,
                filter_type=filter_type,
                window_length=window,
                polyorder=poly_order,
            )
            diff_y = filter_data(
                diff_y.values,
                filter_type=filter_type,
                window_length=window,
                polyorder=poly_order,
            )

        res_dict[player + f"_{new_name[0]}x"] = np.array(diff_x)
        res_dict[player + f"_{new_name[0]}y"] = np.array(diff_y)
        res_dict[player + f"_{new_name}"] = np.linalg.norm([diff_x, diff_y], axis=0)

    new_df = pd.concat([df, pd.DataFrame(res_dict)], axis=1)
    return new_df
