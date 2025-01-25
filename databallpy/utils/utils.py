import numpy as np
import pandas as pd

from databallpy.utils.constants import MISSING_INT


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


def _to_float(value) -> float | int:
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


def sigmoid(
    x: float | np.ndarray,
    a: float = 0.0,
    b: float = 1.0,
    c: float = 1.0,
    d: float = 1.0,
    e: float = 0.0,
) -> float | np.ndarray:
    """Function to calculate the sigmoid function.
    The base is a + b / (1 + c * np.exp(d * -(x - e))). When only x is given, the
    function will return the sigmoid function with the default values of a=0, b=1, c=1,
    d=1, e=0 resulting in the sigmoid function 1 / (1 + exp(-x)).

    Args:
        x (float | np.ndarray): input value(s) for the sigmoid function
        a (float, optional): The first parameter. Defaults to 0.
        b (float, optional): The second parameter. Defaults to 1.
        c (float, optional): The third parameter. Defaults to 1.
        d (float, optional): The fourth parameter. Defaults to 1.
        e (float, optional): The fifth parameter. Defaults to 0.

    Returns:
        float | np.ndarray: The sigmoid function value(s) for the input value(s) x
    """
    in_exp = np.clip(d * -(x - e), a_min=-700, a_max=700)
    return a + (b / (1 + c * np.exp(in_exp)))


def _values_are_equal_(input1: any, input2: any) -> bool:
    """Function to check if two values are equal. The function can check if two values
    are equal if they are of the same type. The function can check if two values are
    equal if they are of the following types: float, int, str, pd.Timestamp, list,
    tuple, dict, np.ndarray, pd.Series, pd.DataFrame. If the values are of another
    type, a NotImplementedError is raised.

    Args:
        input1 (any): The first value to check
        input2 (any): The second value to check

    Raises:
        NotImplementedError: If the values are of another type than float, int, str,
            pd.Timestamp, list, tuple, dict, np.ndarray, pd.Series, pd.DataFrame

    Returns:
        bool: True if the values are equal, False if the values are not equal
    """
    if isinstance(input1, (float, np.floating, int, np.integer)):
        if not isinstance(input2, (float, np.floating, int, np.integer)):
            return False
        if pd.isnull(input1):
            return pd.isnull(input2)
        return np.isclose(input1, input2, atol=1e-5)

    if isinstance(input1, (str, pd.Timestamp)):
        return input1 == input2

    if isinstance(input1, (list, tuple)):
        if not isinstance(input2, (list, tuple)):
            return False
        if len(input1) != len(input2):
            return False
        return all(_values_are_equal_(i1, i2) for i1, i2 in zip(input1, input2))

    if isinstance(input1, dict):
        if not isinstance(input2, dict):
            return False
        if not set(input1.keys()) == set(input2.keys()):
            return False
        return all(_values_are_equal_(input1[k], input2[k]) for k in input1.keys())

    if isinstance(input1, np.ndarray):
        if not isinstance(input2, np.ndarray):
            return False
        if not input1.shape == input2.shape:
            return False
        if not input1.dtype == input2.dtype:
            return False
        return np.allclose(input1, input2, atol=1e-5)

    if isinstance(input1, (pd.Series, pd.DataFrame)):
        if not isinstance(input2, type(input1)):
            return False
        if not input1.shape == input2.shape:
            return False
        if isinstance(input1, pd.DataFrame):
            if not set(input1.columns) == set(input2.columns):
                return False
            return all(_values_are_equal_(input1[c], input2[c]) for c in input1.columns)
        return input1.sort_index().round(4).equals(input2.sort_index().round(4))

    if isinstance(input1, type(pd.to_datetime("NaT"))):
        return isinstance(input2, type(pd.to_datetime("NaT")))

    known_classes = [
        "ShotEvent",
        "PassEvent",
        "DribbleEvent",
        "TackleEvent",
        "Game",
        "IndividualCloseToBallEvent",
        "IndividualOnBallEvent",
    ]
    if type(input1).__name__ in known_classes:
        return input1 == input2

    if input1 is None:
        return input2 is None

    else:
        raise NotImplementedError(
            f"Equality check for {type(input1)} is not implemented"
        )


def _copy_value_(value: any) -> any:
    """Function to copy a value. The function can copy a value if the value is of the
    following types: float, int, str, pd.Timestamp, list, tuple, dict, np.ndarray,
    pd.Series, pd.DataFrame. If the value is of another type, a NotImplementedError is
    raised.

    Args:
        value (any): The value to copy

    Raises:
        NotImplementedError: If the value is of another type than float, int, str,
            pd.Timestamp, list, tuple, dict, np.ndarray, pd.Series, pd.DataFrame

    Returns:
        any: The copied value
    """
    if isinstance(value, (float, np.floating, int, np.integer, str, pd.Timestamp)):
        return value

    if isinstance(value, (list, tuple)):
        return type(value)(_copy_value_(i) for i in value)

    if isinstance(value, dict):
        return {k: _copy_value_(v) for k, v in value.items()}

    if isinstance(value, np.ndarray):
        return value.copy()

    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.copy()

    if value is None:
        return None

    known_classes = [
        "ShotEvent",
        "PassEvent",
        "DribbleEvent",
        "TackleEvent",
        "Game",
        "IndividualCloseToBallEvent",
        "IndividualOnBallEvent",
    ]
    if type(value).__name__ in known_classes:
        return value.copy()

    else:
        raise NotImplementedError(f"Copying of {type(value)} is not implemented")
