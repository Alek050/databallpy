import numpy as np
import pandas as pd


def _add_periods_to_tracking_data(
    timestamps: pd.Series, periods_frames: pd.DataFrame
) -> pd.Series:
    """Function to add periods

    Args:
        tracking_data (pd.DataFrame): _description_
        periods_frames (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
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
        -999,
        2,
        -999,
        3,
        -999,
        4,
        -999,
        5,
    ]

    return np.select(period_conditions, period_values).astype("int64")
