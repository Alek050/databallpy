import numpy as np
import pandas as pd

from databallpy.utils.filters import filter_data
from databallpy.utils.logging import create_logger

LOGGER = create_logger(__name__)


def get_acceleration(
    df: pd.DataFrame,
    input_columns: list,
    framerate: float,
    filter_type: str = None,
    window: int = 7,
    poly_order: int = 2,
    max_val: float = np.nan,
) -> pd.DataFrame:
    """Function that adds acceleration columns based on the position columns

    Args:
        df (pd.DataFrame): tracking data
        input_columns (list): columns for which acceleration should be calculated
        framerate (float): framerate of the tracking data
        filter_type (str, optional): filter type to use. Defaults to None.
            Options are `moving_average` and `savitzky_golay`.
        window (int, optional): window size for the filter. Defaults to 7.
        poly_order (int, optional): polynomial order for the filter. Defaults to 2.
        max_val (float, optional): maximum value for the acceleration. Defaults to
            np.nan.

    Returns:
        pd.DataFrame: tracking data with the added acceleration columns
    """
    try:
        if filter_type not in ["moving_average", "savitzky_golay", None]:
            raise ValueError(
                "filter_type should be one of: 'moving_average', "
                "'savitzky_golay', None, got: {filter_type}"
            )
        for input_column in input_columns:
            if input_column + "_velocity" not in df.columns:
                raise ValueError(
                    f"Velocity was not found for {input_column} in the DataFrame. "
                    " Please calculate velocity first using get_velocity() function."
                )

        res_df = _differentiate(
            df,
            new_name="acceleration",
            metric="v",
            frame_rate=framerate,
            filter_type=filter_type,
            window=window,
            poly_order=poly_order,
            column_ids=input_columns,
            max_val=max_val,
        )

        return res_df
    except Exception as e:
        LOGGER.exception(f"Found unexpected exception in get_acceleration(): \n{e}")
        raise e


def get_velocity(
    df: pd.DataFrame,
    input_columns: list,
    framerate: float,
    filter_type: str = None,
    window: int = 7,
    poly_order: int = 2,
    max_val: float = np.nan,
) -> pd.DataFrame:
    """Function that adds velocity columns based on the position columns

    Args:
        df (pd.DataFrame): tracking data
        input_columns (list): columns for which velocity should be calculated
        framerate (float): framerate of the tracking data
        filter_type (str, optional): filter type to use. Defaults to None.
            Options are `moving_average` and `savitzky_golay`.
        window (int, optional): window size for the filter. Defaults to 7.
        poly_order (int, optional): polynomial order for the filter. Defaults to 2.
        max_val (float, optional): maximum value for the velocity. Defaults to np.nan.

    Returns:
        pd.DataFrame: tracking data with the added velocity columns
    """
    try:
        if filter_type not in ["moving_average", "savitzky_golay", None]:
            raise ValueError(
                "filter_type should be one of: 'moving_average', "
                f"'savitzky_golay', None, got: {filter_type}"
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
            max_val=max_val,
        )

        return res_df
    except Exception as e:
        LOGGER.exception(f"Found unexpected exception in get_velocity(): \n{e}")
        raise e


def _differentiate(
    df: pd.DataFrame,
    *,
    new_name: str,
    metric: str = "",
    frame_rate: int = 25,
    filter_type: str = "savitzky_golay",
    window: int = 7,
    max_val: int = np.nan,
    poly_order: int = 2,
    column_ids: list = None,
):
    """
     Function that adds the differentiated values to the DataFrame.

    Args:
         df (pandas DataFrame): Position data in the x and y directions of players
             and ball.
         metric (str): the metric to differentiate the value on. Note that
             f"{player}_{metric}x" and f"{player}_{metric}y" should exist.
         new_name (str): the name of the magnitude. The first letter will be used
             for the x and y directions. For example, f"{player}_vx" and
             f"{player}_velocity" if new_name = "velocity".
         frame_rate (int): the sample frequency of the data.
         filter_type (str): the type of filter to use. Options are "moving average",
             "savitzky_golay", or None.
         window (int): the window size of the filter
         max_val (float): The maximim value of the differentiated value. For
             instance, player speeds > 12 m/s are very unlikely.
         poly_order (int): the polynomial order for the Savitzky-Golay filter.

     Returns:
         df (pandas DataFrame): the DataFrame with the added columns.
    """

    try:
        to_skip = len(metric) + 2
        if column_ids is None:
            column_ids = [x[:-to_skip] for x in df.columns if f"_{metric}x" in x]

        dt = 1.0 / frame_rate

        cols_to_drop = np.array([[c + f"_{new_name}", c + f"_{new_name[0]}x", c + f"_{new_name[0]}y"] for c in column_ids]).ravel()
        df.drop(cols_to_drop, axis=1, errors="ignore", inplace=True)

        res_dict = {}
        for player in column_ids:
            gradient_x = np.gradient(df[player + f"_{metric}x"].values, dt)
            gradient_y = np.gradient(df[player + f"_{metric}y"].values, dt)

            # remove outliers
            raw_differentiated = np.linalg.norm([gradient_x, gradient_y], axis=0)
            if not pd.isnull(max_val):
                gradient_x[
                    (raw_differentiated > max_val) & (gradient_x > max_val)
                ] = max_val
                gradient_x[
                    (raw_differentiated > max_val) & (gradient_x < -max_val)
                ] = -max_val
                gradient_y[
                    (raw_differentiated > max_val) & (gradient_y > max_val)
                ] = max_val
                gradient_y[
                    (raw_differentiated > max_val) & (gradient_y < -max_val)
                ] = -max_val

            # smoothing the signal
            if filter_type is not None:
                gradient_x = filter_data(
                    gradient_x,
                    filter_type=filter_type,
                    window_length=window,
                    polyorder=poly_order,
                )
                gradient_y = filter_data(
                    gradient_y,
                    filter_type=filter_type,
                    window_length=window,
                    polyorder=poly_order,
                )

            res_dict[player + f"_{new_name[0]}x"] = np.array(gradient_x)
            res_dict[player + f"_{new_name[0]}y"] = np.array(gradient_y)
            res_dict[player + f"_{new_name}"] = np.linalg.norm(
                [gradient_x, gradient_y], axis=0
            )

        df = pd.concat([df, pd.DataFrame(res_dict)], axis=1)
        return df

    except Exception as e:
        LOGGER.exception(f"Found unexpected exception in _differentiate: \n{e}")
        raise e
