import numpy as np
import pandas as pd

from databallpy.utils.constants import MISSING_INT


def _add_periods_to_tracking_data(
    timestamps: pd.Series, periods_frames: pd.DataFrame
) -> pd.Series:
    """Function to add periods

    Args:
        tracking_data (pd.DataFrame): The tracking data
        periods_frames (pd.DataFrame): The periods frames

    Returns:
        pd.DataFrame: The tracking data with periods
    """
    period_conditions = [
        (timestamps <= periods_frames.loc[0, "end_frame"]),
        (timestamps > periods_frames.loc[0, "end_frame"])
        & (timestamps < periods_frames.loc[1, "start_frame"]),
        (timestamps >= periods_frames.loc[1, "start_frame"])
        & (timestamps <= periods_frames.loc[1, "end_frame"]),
        (timestamps > periods_frames.loc[1, "end_frame"])
        & (timestamps < periods_frames.loc[2, "start_frame"]),
        (timestamps >= periods_frames.loc[2, "start_frame"])
        & (timestamps < periods_frames.loc[2, "end_frame"]),
        (timestamps > periods_frames.loc[2, "end_frame"])
        & (timestamps < periods_frames.loc[3, "start_frame"]),
        (timestamps >= periods_frames.loc[3, "start_frame"])
        & (timestamps < periods_frames.loc[3, "end_frame"]),
        (timestamps > periods_frames.loc[3, "end_frame"])
        & (timestamps < periods_frames.loc[4, "start_frame"]),
        (timestamps > periods_frames.loc[4, "start_frame"]),
    ]
    period_values = [
        1,
        MISSING_INT,
        2,
        MISSING_INT,
        3,
        MISSING_INT,
        4,
        MISSING_INT,
        5,
    ]

    return np.select(period_conditions, period_values).astype("int64")
