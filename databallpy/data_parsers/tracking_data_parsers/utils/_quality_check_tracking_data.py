import warnings

import numpy as np
import pandas as pd

from databallpy.features.differentiate import _differentiate
from databallpy.utils.constants import MISSING_INT
from databallpy.utils.warnings import DataBallPyWarning


def _quality_check_tracking_data(
    tracking_data: pd.DataFrame, framerate: int, periods: pd.DataFrame
) -> None:
    """Function that does a quality check on the tracking data

    Args:
        tracking_data (pd.DataFrame): tracking data
        framerate (int): framerate of the tracking data
        periods (pd.DataFrame): holds information on start and end of all periods

    Raises:
        DataBallPyWarning: when at least one of these criteria is met:
        1. ball data is not available for more than 1% of all frames
        2. there is a gap in the ball data for more than 1 seconds
        3. ball velocity is unrealistic for more than 1% of all frames
        4. there is a gap in the realistic ball data for more than 1 seconds
        5. the velocity of at least one player is unrealistic for more than
        0.5% of the player's play time
        6. there is a gap in the realistic player velocity for at least 1 seconds
        for at least one player
        7. home player data is not available for more than 30 seconds when the ball
        status is alive
        8. away player data is not available for more than 30 seconds when the ball
        status is alive
        9. ball data is not available for more than 30 seconds when the ball status
        is alive
    """

    _check_missing_ball_data(tracking_data, framerate)
    _check_ball_velocity(tracking_data, framerate)
    _check_player_velocity(tracking_data, framerate, periods)
    allow_syncronise_tracking_and_event_data = _check_missing_player_data(
        tracking_data, framerate
    )

    return allow_syncronise_tracking_and_event_data


def _check_missing_player_data(
    tracking_data: pd.DataFrame, framerate: int, n_seconds: float = 30.0
) -> bool:
    """Function to check if there is data of at least 7 players available when the
    ball status is alive. If this is not the case for 30 seconds of data, a warning
    is raised and false is returned. Else True is returned.

    Args:
        tracking_data (pd.DataFrame): tracking data
        framerate (int): framrate of the tracking data
        n_seconds (float): number of seconds to check for missing data. Defaults to 30.

    Raises:
        DataBallPyWarning: when at least one of these criteria is met:
        1. home player data is not available for more than 30 seconds when the ball
        status is alive
        2. away player data is not available for more than 30 seconds when the ball
        status is alive
        3. ball data is not available for more than 30 seconds when the ball status
        is alive

    Returns:
        bool: wheter to allow syncronise tracking and event data.
    """
    mask_ball_alive = (tracking_data["ball_status"] == "alive") & (
        tracking_data["gametime_td"] != "Break"
    )
    home_players = [
        x for x in tracking_data.columns if "home" in x and ("_x" in x or "_y" in x)
    ]
    away_players = [
        x for x in tracking_data.columns if "away" in x and ("_x" in x or "_y" in x)
    ]
    ball = ["ball_x", "ball_y"]
    missing_home_frames = (
        tracking_data[mask_ball_alive][home_players].notna().sum(axis=1) < 7 * 2
    ).sum()
    missing_away_frames = (
        tracking_data[mask_ball_alive][away_players].notna().sum(axis=1) < 7 * 2
    ).sum()
    missing_ball_frames = (
        tracking_data[mask_ball_alive][ball].notna().sum(axis=1) < 2
    ).sum()

    for side, n_missing in zip(
        ["home players", "away players", "ball"],
        [missing_home_frames, missing_away_frames, missing_ball_frames],
    ):
        if n_missing > n_seconds * framerate:
            warnings.warn(
                DataBallPyWarning(
                    f"{side} data is not available for more than {n_seconds} seconds"
                    " when the ball status is alive. Syncronising tracking and event"
                    " data is disabled due to bad quality of the tracking data."
                    "\nIf you still wish to syncronise the data, please set "
                    "allow_syncronise_tracking_and_event_data to True in your game "
                    "object."
                )
            )
            return False
    return True


def _check_missing_ball_data(tracking_data: pd.DataFrame, framerate: int) -> None:
    """Function that checks the amount of missing ball data

    Args:
        tracking_data (pd.DataFrame): tracking data
        framerate (int): framerate of the tracking data

    Raises:
        DataBallPyWarning: when at least on of these criteria is met:
        1. ball data is not available for more than 1% of all frames
        2. there is a gap in the ball data for more than 1 seconds

    """
    mask_ball_alive = (tracking_data["ball_status"] == "alive") & (
        tracking_data["gametime_td"] != "Break"
    )
    valid_frames = ~np.isnan(
        tracking_data.loc[mask_ball_alive, ["ball_x", "ball_y"]]
    ).any(axis=1)
    sum_valid_frames = sum(valid_frames)
    n_total_frames = len(tracking_data.loc[mask_ball_alive])
    if sum_valid_frames < n_total_frames * 0.99:
        warnings.warn(
            DataBallPyWarning(
                "Ball data is not available for more than 1% of all frames"
            )
        )

    max_nan_sequence = _max_sequence_invalid_frames(valid_frames)
    if max_nan_sequence >= 1 * framerate:
        warnings.warn(
            DataBallPyWarning("There is a gap in the ball data for at least 1 seconds")
        )

    return


def _check_ball_velocity(tracking_data: pd.DataFrame, framerate: int) -> None:
    """Function that checks the ball data by checking it's velocity
    The max ball velocity was set at 50 m/s\N{SUPERSCRIPT TWO}), because this
    coincides with the hardest shot in football.

    Args:
        tracking_data (pd.DataFrame): tracking data
        framerate (int): framerate of the trackin data

    Raises:
        DataBallPyWarning: when at least on of these criteria is met:
        1. ball velocity is unrealistic for more than 1% of all frames
        2. there is a gap in the realistic ball data for more than 1 seconds

    """
    initial_columns = tracking_data.columns
    if "ball_velocity" not in tracking_data.columns:
        tracking_data = _differentiate(
            tracking_data,
            new_name="velocity",
            metric="",
            frame_rate=framerate,
            filter_type=None,
            column_ids=["ball"],
        )
    velocity_ball = tracking_data["ball_velocity"]
    mask_ball_alive = (tracking_data["ball_status"] == "alive") & (
        tracking_data["gametime_td"] != "Break"
    )
    valid_frames = velocity_ball[mask_ball_alive][1:] < 50

    sum_valid_frames = sum(valid_frames)
    n_total_frames = len(valid_frames)

    if sum_valid_frames < n_total_frames * 0.99:
        warnings.warn(
            DataBallPyWarning(
                "Ball velocity is unrealistic (> 50 m/s) for "
                "more than 1% of all frames"
            )
        )

    max_invalid_sequence = _max_sequence_invalid_frames(valid_frames)

    if max_invalid_sequence > 1 * framerate:
        warnings.warn(
            DataBallPyWarning("There is a gap in the ball data for at least 1 second")
        )

    tracking_data = tracking_data[initial_columns]
    return


def _check_player_velocity(
    tracking_data: pd.DataFrame, framerate: int, periods: pd.DataFrame
) -> None:
    """Function that checks all the player tracking data by checking the velocities
    The max player velocity was set at 12 m/s) as this coincides
    with the biggest player velocity recorded in football.

    Args:
        tracking_data (pd.DataFrame): tracking data
        framerate (int): framerate of the tracking data
        periods (pd.DataFrame): holds information on start and end of all periods

    Raises:
        DataBallPyWarning: when at least on of these criteria is met:
        1. the velocity of at least one player is unrealistic for more than
        0.5% of the player's play time
        2. there is a gap in the realistic player velocity for at least 1 seconds
        for at least one player
    """
    initial_columns = tracking_data.columns
    player_columns = [x for x in tracking_data.columns if "home" in x or "away" in x]
    players_column_ids = [x.replace("_x", "") for x in player_columns if "_x" in x]
    tracking_data = _differentiate(
        tracking_data,
        new_name="velocity",
        metric="",
        frame_rate=framerate,
        filter_type=None,
        column_ids=players_column_ids,
    )

    mask_no_break = [False] * len(tracking_data)
    first_frame = periods.loc[0, "start_frame"]
    for _, row in periods.iterrows():
        if row["start_frame"] != MISSING_INT:
            p_start = row["start_frame"] - first_frame
            p_end = row["end_frame"] - first_frame
            mask_no_break[p_start:p_end] = [True] * (p_end - p_start)

    percentages_valid_frames = []
    max_sequences_invalid_frames = []
    for player in players_column_ids:
        velocity_player = tracking_data[f"{player}_velocity"]
        velocity_player = velocity_player[mask_no_break][1:].reset_index(drop=True)
        valid_frames = velocity_player < 12
        sum_valid_frames = sum(valid_frames)
        player_specific_total_frames = (
            valid_frames[::-1].idxmax() - valid_frames.idxmax()
        )
        percentages_valid_frames.append(
            sum_valid_frames
            / np.clip(player_specific_total_frames, a_min=1, a_max=np.inf)
        )
        max_sequences_invalid_frames.append(
            _max_sequence_invalid_frames(valid_frames, False)
        )

    n_players_to_many_invalid_frames = sum(
        [True for x in percentages_valid_frames if x < 0.995]
    )

    if n_players_to_many_invalid_frames > 0:
        warnings.warn(
            DataBallPyWarning(
                f"For {n_players_to_many_invalid_frames} players, the "
                "velocity is unrealistic (speed > 12 m/s) for more than 0.5% of "
                "playing time"
            )
        )

    tracking_data = tracking_data[initial_columns]
    return


def _max_sequence_invalid_frames(
    valid_frames: pd.Series, include_start_finish: bool = True
) -> int:
    """Function that gives the max sequence length of invalid frames.
    Based on include_start_finish,the function ignores the first and last
    sequence of invalid values (when a player doesn't start or ends the game)

    Args:
        valid_frames (pd.Series): series where valid frames are True
        and invalid ones are False
        include_start_finish (bool, optional): whethere to include the first
        and last sequence of invalid data.This can be useful if a player
        doesn't start of finish the game, which leads to a lot of invalid data
        at the end or start. Defaults to True.

    Returns:
        int: length of the longest sequence of invalid frames
    """
    blocks = valid_frames.cumsum()
    sequences = (~valid_frames).groupby(blocks).sum()
    if not include_start_finish:
        sequences = sequences[1:-1]
    return sequences.max()
