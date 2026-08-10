import warnings

import numpy as np
import pandas as pd

from databallpy.features.filters import _filter_data
from databallpy.utils.logging import logging_wrapper
from databallpy.utils.warnings import DataBallPyWarning


@logging_wrapper(__file__)
def _differentiate(
    df: pd.DataFrame,
    *,
    new_name: str,
    metric: str = "",
    frame_rate: int | float = 25,
    filter_type: str = "savitzky_golay",
    window: int = 7,
    max_val: float = np.nan,
    poly_order: int = 2,
    column_ids: list[str] | None = None,
    inplace: bool = False,
    allow_overwrite: bool = False,
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

    if allow_overwrite:
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

        for col, values in zip(
            [
                column_id + f"_{new_name[0]}x",
                column_id + f"_{new_name[0]}y",
                column_id + f"_{new_name}",
            ],
            [gradient_x, gradient_y, np.linalg.norm([gradient_x, gradient_y], axis=0)],
        ):
            if col not in df.columns:
                res_dict[col] = values

    if len(res_dict) == 0 and not allow_overwrite:
        warnings.warn(
            message="No values added to the tracking data. Consider setting `allow_overwrite` to True",
            category=DataBallPyWarning,
        )

    new_columns_df = pd.DataFrame(res_dict)

    if inplace:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
            df[new_columns_df.columns] = new_columns_df
        return None
    else:
        return pd.concat([df, new_columns_df], axis=1)
