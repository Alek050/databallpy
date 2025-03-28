import warnings

import numpy as np
import pandas as pd

from databallpy.features.filters import _filter_data
from databallpy.utils.logging import logging_wrapper
from databallpy.utils.warnings import deprecated


@logging_wrapper(__file__)
@deprecated(
    "The add_acceleration function is deprecated and will removed in version 0.8.0. Please use Game.TrackingData.add_acceleration() instead."
)
def add_acceleration(
    tracking_data: pd.DataFrame,
    column_ids: str | list[str],
    frame_rate: float,
    filter_type: str = None,
    window_length: int = 25,
    polyorder: int = 2,
    max_acceleration: float = np.inf,
    inplace: bool = False,
) -> pd.DataFrame:
    """Function that adds acceleration columns based on the position columns

    Args:
        tracking_data (pd.DataFrame): tracking data
        column_ids (str | list[str]): columns for which acceleration should be
            calculated
        frame_rate (float): framerate of the tracking data
        filter_type (str, optional): filter type to use. Defaults to None.
            Options are `moving_average` and `savitzky_golay`.
        window_length (int, optional): window size for the filter. Defaults to 25.
        polyorder (int, optional): polynomial order for the filter. Defaults to 2.
        max_acceleration (float, optional): maximum value for the acceleration.
            Defaults to np.inf.
        inplace (bool, optional): whether to modify the DataFrame in place.
            Defaults to False.

    Returns:
        pd.DataFrame: tracking data with the added acceleration columns

    Raises:
        ValueError: if filter_type is not one of `moving_average`, `savitzky_golay`,
            or None.
        ValueError: if velocity was not found in the DataFrame for the input_columns.

    Note:
        The function will delete the columns in input_columns with the acceleration if
        they already exist.
    """

    if isinstance(column_ids, str):
        column_ids = [column_ids]

    if filter_type not in ["moving_average", "savitzky_golay", None]:
        raise ValueError(
            "filter_type should be one of: 'moving_average', "
            f"'savitzky_golay', None, got: {filter_type}"
        )
    for column_id in column_ids:
        if (
            column_id + "_vx" not in tracking_data.columns
            or column_id + "_vy" not in tracking_data.columns
        ):
            raise ValueError(
                f"Velocity was not found for {column_id} in the DataFrame. "
                " Please calculate velocity first using get_velocity() function."
            )

    res_df = _differentiate(
        tracking_data,
        new_name="acceleration",
        metric="v",
        frame_rate=frame_rate,
        filter_type=filter_type,
        window=window_length,
        poly_order=polyorder,
        column_ids=column_ids,
        max_val=max_acceleration,
        inplace=inplace,
    )

    return res_df


@logging_wrapper(__file__)
@deprecated(
    "The add_velocity function is deprecated and will removed in version 0.8.0. Please use Game.TrackingData.add_velocity() instead."
)
def add_velocity(
    tracking_data: pd.DataFrame,
    column_ids: str | list[str],
    frame_rate: float,
    filter_type: str = None,
    window_length: int = 7,
    polyorder: int = 2,
    max_velocity: float = np.inf,
    inplace: bool = False,
) -> pd.DataFrame:
    """Function that adds velocity columns based on the position columns

    Args:
        tracking_data (pd.DataFrame): tracking data
        column_ids (str | list[str]): columns for which velocity should be calculated
        frame_rate (float): framerate of the tracking data
        filter_type (str, optional): filter type to use. Defaults to None.
            Options are `moving_average` and `savitzky_golay`.
        window_length (int, optional): window size for the filter. Defaults to 7.
        polyorder (int, optional): polynomial order for the filter. Defaults to 2.
        max_velocity (float, optional): maximum value for the velocity.
            Defaults to np.inf.
        inplace (bool, optional): whether to modify the DataFrame in place. Defaults

    Returns:
        pd.DataFrame: tracking data with the added velocity columns if inplace is False
            else None.

    Raises:
        ValueError: if filter_type is not one of `moving_average`, `savitzky_golay`,
            or None.

    Note:
        The function will delete the columns in input_columns with the velocity if
        they already exist.
    """

    if isinstance(column_ids, str):
        column_ids = [column_ids]

    if filter_type not in ["moving_average", "savitzky_golay", None]:
        raise ValueError(
            "filter_type should be one of: 'moving_average', "
            f"'savitzky_golay', None, got: {filter_type}"
        )

    res_df = _differentiate(
        tracking_data,
        new_name="velocity",
        metric="",
        frame_rate=frame_rate,
        filter_type=filter_type,
        window=window_length,
        poly_order=polyorder,
        column_ids=column_ids,
        max_val=max_velocity,
        inplace=inplace,
    )

    return res_df


@logging_wrapper(__file__)
def _differentiate(
    df: pd.DataFrame,
    *,
    new_name: str,
    metric: str = "",
    frame_rate: int = 25,
    filter_type: str = "savitzky_golay",
    window: int = 7,
    max_val: float = np.nan,
    poly_order: int = 2,
    column_ids: list[str] | None = None,
    inplace: bool = False,
) -> pd.DataFrame | None:
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
        max_val (float): The maximum value of the differentiated value. For
            instance, player speeds > 12 m/s are very unlikely.
        poly_order (int): the polynomial order for the Savitzky-Golay filter.
        column_ids (list[str] | None): the columns to differentiate. If None, all
            columns with the metric in the name will be used. Defaults to None.
        inplace (bool): whether to modify the DataFrame in place. Defaults to False.

    Returns:
        pd.DataFrame | None: the DataFrame with the added columns if inplace is False,
        otherwise None.
    """

    if not inplace:
        df = df.copy()

    to_skip = len(metric) + 2
    if column_ids is None:
        column_ids = [x[:-to_skip] for x in df.columns if f"_{metric}x" in x]

    dt = 1.0 / frame_rate

    cols_to_drop = np.array(
        [
            [c + f"_{new_name}", c + f"_{new_name[0]}x", c + f"_{new_name[0]}y"]
            for c in column_ids
        ]
    ).ravel()
    df.drop(cols_to_drop, axis=1, errors="ignore", inplace=True)

    res_dict = {}
    for column_id in column_ids:
        gradient_x = np.gradient(df[column_id + f"_{metric}x"].values, dt)
        gradient_y = np.gradient(df[column_id + f"_{metric}y"].values, dt)
        raw_differentiated = np.linalg.norm([gradient_x, gradient_y], axis=0)

        # Scale gradients if magnitude exceeds max_val
        if not pd.isnull(max_val):
            exceed_max = raw_differentiated > max_val
            scale_factor = max_val / raw_differentiated[exceed_max]
            gradient_x[exceed_max] *= scale_factor
            gradient_y[exceed_max] *= scale_factor

        # smoothing the signal
        if filter_type is not None:
            gradient_x = _filter_data(
                gradient_x,
                filter_type=filter_type,
                window_length=window,
                polyorder=poly_order,
            )
            gradient_y = _filter_data(
                gradient_y,
                filter_type=filter_type,
                window_length=window,
                polyorder=poly_order,
            )

        res_dict[column_id + f"_{new_name[0]}x"] = np.array(gradient_x)
        res_dict[column_id + f"_{new_name[0]}y"] = np.array(gradient_y)
        res_dict[column_id + f"_{new_name}"] = np.linalg.norm(
            [gradient_x, gradient_y], axis=0
        )

    new_columns_df = pd.DataFrame(res_dict)

    if inplace:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
            df[new_columns_df.columns] = new_columns_df
        return None
    else:
        return pd.concat([df, new_columns_df], axis=1)
