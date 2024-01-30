from typing import Union

import numpy as np
import pandas as pd

MISSING_INT = -999


def sigmoid(x:np.ndarray, a:float = 1, b:float = 1, c:float = 0, d:float = 0) -> np.ndarray:
    """Sigmoid function in the shape of a / (1 + e^(-b * (x - c)) ) + d. The defaults 
    are initialized so that the the default function is 1 / (1 + e^(-x)).

    Args:
        x (np.ndarray): Input values.
        a (float, optional): Formula constant. Defaults to 1.
        b (float, optional): Formula constant. Defaults to 1.
        c (float, optional): Formula constant. Defaults to 0.
        d (float, optional): Formula constant. Defaults to 0.

    Returns:
        np.ndarray: Values after passing through the sigmoid. Returned array is
            the same shape as the input shape of x
    """
    return a / (1 + np.exp(-b * (x - c))) + d

def _to_int(value) -> int:
    """Function to make a integer of the value if possible, else MISSING_INT (-999)

    Args:
        value (): a variable value

    Returns:
       int: integer if value can be changed to integer, else MISSING_INT (-999)
    """
    try:
        value = _to_float(value)
        return int(value)
    except (TypeError, ValueError):
        return MISSING_INT


def _to_float(value) -> Union[float, int]:
    """Function to make a float of the value if possible, else np.nan

    Args:
        value (): a variable value

    Returns:
        Union[float, int]: integer if value can be changed to integer, else np.nan
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def get_next_possession_frame(
    tracking_data: pd.DataFrame, tracking_data_frame: pd.Series, passer_column_id: str
) -> pd.Series:
    """Function to get the next frame where the ball is in possession of a player
         other than the passer or the ball is out of play.

    Args:
        tracking_data (pd.DataFrame): tracking data
        tracking_data_frame (pd.DataFrame): frame of the tracking data where the pass
            is made.
        passer_column_id (str): column id of the passer

    Returns:
        pd.Series: first frame after the pass of the tracking data where the ball is in
            possession of a player other than the passer or the ball is out of play.
    """

    next_alive_mask = (tracking_data["ball_status"] == "alive") & (
        tracking_data.index > tracking_data_frame.name
    )
    mask_new_possession = (
        (~pd.isnull(tracking_data["player_possession"]))
        & (tracking_data["player_possession"] != passer_column_id)
        & (tracking_data.index > tracking_data_frame.name)
    )
    if len(tracking_data.loc[mask_new_possession]) > 0:
        first_new_possession_idx = tracking_data.loc[mask_new_possession].index[0]
    else:
        first_new_possession_idx = tracking_data.index[-1]

    if len(tracking_data.loc[next_alive_mask]) > 0:
        alive_again_idx = tracking_data.loc[next_alive_mask].index[0]
    else:
        alive_again_idx = tracking_data.index[-1]

    mask_ball_out = (
        (tracking_data["ball_status"] == "dead")
        & (tracking_data.index > tracking_data_frame.name)
        & (tracking_data.index >= alive_again_idx)
    )
    if len(tracking_data.loc[mask_ball_out]) > 0:
        next_ball_out_idx = tracking_data.loc[mask_ball_out].index[0]
    else:
        next_ball_out_idx = tracking_data.index[-1]

    end_pos_frame = tracking_data.loc[min(first_new_possession_idx, next_ball_out_idx)]

    return end_pos_frame
