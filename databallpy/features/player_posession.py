from typing import Tuple

import numpy as np
import pandas as pd

from databallpy.features.angle import get_smallest_angle
from databallpy.features.differentiate import _differentiate


def get_individual_player_posessions_and_duels(
    tracking_data: pd.DataFrame,
    frame_rate: int,
    pz_radius: float = 1.0,
    dz_radius: float = 1.0,
    bv_threshold: float = 5.0,
    ba_threshold: float = 10.0,
    bd_threshold: float = 0.1,
    min_frames: int = 0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Function to calculate which player has possesion of the ball in accordance with the
    article of Vidal-Codina et al (2022) : "Automatic Event Detection in Football Using
    Tracking Data". The algorithm finds if the ball is within the possession zone (PZ)
    of a player, it losses the ball when the ball is outside of the PZ in a next frame
    and a different player gains the ball in his PZ. The ball is only gained when the
    ball changes direction or speed, to correct for balls that fly over players.

    Note: if tracking and event data is synchronised, the function will also count a
    gain of posession when a player attempts an on ball action: [pass, shot, dribble].
    It is assumed that the player performs the action when there is no duel
    happening at the same time since we have no data here what player performs the
    action.

    :param tracking_data: pandas df with tracking data over which to calculate the
        player posession.
    :param frame_rate: int with the sampling frequency of the tracking data
    :param pz_radius: float with the radius of the posession zone (PZ) in meters.
        Default is 1.0 m
    :param dz_radius: float with the radius of the duel zone (DZ) in meters.
        Default is 1.0 m
    :param bv_threshold: float with the threshold for the ball velocity (in m/s) change
        needed to gain ball posession. Default is 5.0 m/s
    :param ba_threshold: float with the threshold for the ball angle (in degree) change
        needed to gain ball posession. Default is 10.0 degree.
    :param bd_threshold: float with the threshold for the ball displacement (in meters)
        change needed to lose ball posession. Default is 0.1 m.
    :param min_frames: int with the minimum number of frames a player needs to have the
        ball to be considered posession. Default is 0 frames.

    :returns: pd.Series with which player has posession and a pd.Series with duels over
        time.
    """

    if "ball_velocity" not in tracking_data.columns:
        tracking_data = _differentiate(
            tracking_data,
            new_name="velocity",
            metric="",
            frame_rate=frame_rate,
            filter_type=None,
            column_ids=["ball"],
        )

    distances_df = get_distance_between_ball_and_players(tracking_data)
    pz_initial = get_initial_posessions(
        tracking_data, distances_df, pz_radius=pz_radius, min_frames=min_frames
    )
    duels = get_duels(tracking_data, distances_df, dz_radius=dz_radius)

    player_posession = pd.Series(index=tracking_data.index)
    player_posession[:] = None

    # Find intervals of player posessions, can also include intervals with None values.
    shifting_idxs = np.where(pz_initial.values[:-1] != pz_initial.values[1:])[0]
    shifting_idxs = np.concatenate([[-1], shifting_idxs, [len(pz_initial) - 1]])
    posession_start_idxs = shifting_idxs[:-1] + 1
    next_posession_start_idxs = posession_start_idxs[1:]
    next_posession_start_idxs = np.concatenate(
        [next_posession_start_idxs, [len(pz_initial) - 1]]
    )
    posession_end_idxs = shifting_idxs[1:]  # inclusive index, so include it!
    last_valid_idx = None
    valid_gains = get_valid_gains(
        tracking_data,
        posession_start_idxs,
        posession_end_idxs,
        bv_threshold,
        ba_threshold,
        duels=duels,
    )

    for idx, (start_idx, end_idx) in enumerate(
        zip(posession_start_idxs, posession_end_idxs)
    ):

        if pz_initial[start_idx] is None:
            continue

        player_column_id = pz_initial[start_idx]
        if valid_gains[idx]:

            # posession is lost somewhere between last pz index
            # and the next posession of a player
            if pz_initial[next_posession_start_idxs[idx]] is None:
                next_posession_start_idx = next_posession_start_idxs[idx + 1]
            else:
                next_posession_start_idx = next_posession_start_idxs[idx]

            lost_posession_idx = get_lost_posession_idx(
                tracking_data.loc[end_idx:next_posession_start_idx], bd_threshold
            )

            # check if the player is the same as the last player in posession
            if last_valid_idx is not None:
                last_posession_idx = last_valid_idx
                last_player_column_id = pz_initial[last_posession_idx]
                if last_player_column_id == player_column_id:
                    start_idx = last_posession_idx

            # update last_valid_idx
            last_valid_idx = lost_posession_idx

            player_posession[start_idx : lost_posession_idx + 1] = player_column_id

    # Lastly, only apply when ball status is alive
    alive_mask = tracking_data["ball_status"] == "alive"
    player_posession[~alive_mask] = None
    duels[~alive_mask] = None

    return player_posession, duels


def get_initial_posessions(
    tracking_data: pd.DataFrame,
    distances_df: pd.DataFrame,
    pz_radius: float,
    min_frames: int,
) -> pd.Series:
    """Function to calculate the initial posession of the ball. This is when a player is
    within the PZ of the ball. Posession is assigned to a player when he is the closest
    player to the ball and within the PZ of the ball.
    Note: The duel zones are not taken into account in this function.

    Args:
        tracking_data (pd.DataFrame): pandas df with tracking data over which to
        calculate the player posession.
        distances_df (pd.DataFrame): pandas df with the distances between the ball and
            all players.
        pz_radius (float): float with the radius of the posession zone (PZ) in meters.
        min_frames (int): int with the minimum number of frames a player needs to be
            within the PZ to be considered posession.

    Returns:
        pd.Series: pd.Series with which player has posession.
    """

    # find the player_posession player to the ball
    closest_player = distances_df.idxmin(axis=1)
    close_enough = distances_df.min(axis=1) < pz_radius
    initial_posession = np.where(close_enough, closest_player, None)

    # find the players that have posession for at least min_frames
    # embrace yourself for some pretty abstract thinking...
    if min_frames > 0:
        # note that the indexes are the last index where the value is the same as the
        # last sequence of the same value. [0, 0, 1, 1, 1, 2,] will return [1, 4].
        shifting_idxs = np.where(initial_posession[:-1] != initial_posession[1:])[0]
        # add first and last index
        shifting_idxs = np.concatenate(
            [[0], shifting_idxs, [len(initial_posession) - 1]]
        )
        # get the number of frames per value (can also be None values)
        frames_per_value = np.diff(shifting_idxs)
        too_short = frames_per_value < min_frames
        # get the indexes, in shifting_idxs, where the frames_per_value is too short
        too_short_idxs = np.where(too_short)[0]

        # value in shifiting_idxs is the index of the last frame of the sequence
        # before the change, so add 1 to get the first frame of the sequence that
        # is too short
        too_short_start_idxs = shifting_idxs[too_short_idxs] + 1

        # add one to too_short_idx to get the index where the sequence is ending
        # note, this index is the last frame of the sequence, so include it!
        too_short_end_idxs = shifting_idxs[too_short_idxs + 1]
        for start_idx, end_idx in zip(too_short_start_idxs, too_short_end_idxs):
            initial_posession[start_idx : end_idx + 1] = None  # include the end_idx

        # note that some frames_per_value are still < min_frames, these are None values
        # that is allowed, we only want to shorten the posessions that are too short,
        # not the in between intervals.

    return pd.Series(initial_posession, index=tracking_data.index)


def get_duels(
    tracking_data: pd.DataFrame, distances_df: pd.DataFrame, dz_radius: float
) -> pd.Series:
    """Function to calculate the duels of the ball. This is when two opponent
    players are within the DZ of the ball. If more than one player of a team is within
    the duel zone of the ball, the closest player is assigned to the duel.

    Args:
        tracking_data (pd.DataFrame): pandas df with tracking data over which to
            calculate the player posession.
        distances_df (pd.DataFrame): pandas df with the distances between the ball and
            all players.
        dz_radius (float): float with the radius of the duel zone (DZ) in meters.

    Returns:
        pd.Series: pd.Series with which players are in a duel. If 'home_1' and 'home_2'
            are in a duel, the duel the outcome is 'home_1-home_2'.
    """
    home_distances_df = distances_df.filter(like="home")
    away_distances_df = distances_df.filter(like="away")

    # .lt is quicker than using the < operator
    home_close_enough = home_distances_df.lt(dz_radius)
    away_close_enough = away_distances_df.lt(dz_radius)

    valid_duel = home_close_enough.any(axis=1) & away_close_enough.any(axis=1)
    duel = np.where(
        valid_duel,
        home_distances_df.idxmin(axis=1) + "-" + away_distances_df.idxmin(axis=1),
        None,
    )

    return pd.Series(duel, index=tracking_data.index)


def get_distance_between_ball_and_players(tracking_data: pd.DataFrame) -> pd.DataFrame:
    """Function to calculate the distances between the ball and all players.

    Args:
        tracking_data (pd.DataFrame): pandas df with tracking data over which to
            calculate the distances.

    Returns:
        pd.DataFrame: pandas df with the distances between the ball and all players.
    """

    player_column_ids = [
        col[:-2] for col in tracking_data.columns if "_x" in col and "ball" not in col
    ]
    ball_xy = tracking_data[["ball_x", "ball_y"]].values
    distances_df = pd.DataFrame(columns=player_column_ids, index=tracking_data.index)
    for player_column_id in player_column_ids:
        player_xy = tracking_data[
            [f"{player_column_id}_x", f"{player_column_id}_y"]
        ].values
        distances = np.linalg.norm(ball_xy - player_xy, axis=1)
        distances_df[player_column_id] = distances

    return distances_df


def get_lost_posession_idx(tracking_data: pd.DataFrame, bd_threshold: float) -> int:
    """
    Function to check when a player has lost posession of the ball, in accordance with
    the article of Vidal-Codina et al (2022). The posession of the ball is lost when two
    conditions are satisfied: (1) the ball is outside the pz_radius of the player and
    the ball moves above the displacement threshold, and (2), the player is not present
    in the next frame where there is either a posession or a duel. This function checks
    conditions 1, condition 2 is checked in get_individual_player_posession

    Args:
        tracking_data (pd.DataFrame): pandas df with tracking data over which to
            calculate the player posession. Note that the df should be shortened
            to the time the player is outside the pz, untill the next player gets.
            in posession
        bd_threshold (float): float with the threshold for the ball displacement
            (in meters)

    Returns:
        int: int with the index where the player lost posession of the ball.
    """

    ball_displacement_condition_met = tracking_data["ball_velocity"] > bd_threshold
    if ball_displacement_condition_met.sum() == 0:
        return tracking_data.index[-1]
    else:
        return ball_displacement_condition_met.idxmax()


def get_valid_gains(
    tracking_data: pd.DataFrame,
    posession_start_idxs: np.ndarray,
    posession_end_idxs: np.ndarray,
    bv_threshold: float,
    ba_threshold: float,
    duels: pd.Series = None,
) -> np.ndarray:
    """Function to check if, within a given period, a player gains posession of the
    ball. Posession is gained if the ball speed changes at least bs_threshold m/s or
    the ball changes direction (> ba_threshold) between the first and the last
    proposed posession frame.

    If tracking and event data is synchronised, the function will also return true for
    a posession when the player attempts an on ball action: [pass, shot, dribble].
    It is assumed that the player performs the action when there is no duel happening.
    at the same time since we have no data here what player performs the action.

    Args:
        tracking_data (pd.DataFrame): pandas df with tracking data over which to
            calculate the player posession.
        posession_start_idxs (np.ndarray): array with the starting indexes of the
            proposed posessions.
        posession_end_idxs (np.ndarray): array with the ending indexes of the
            proposed posessions.
        bv_threshold (float): minimal velocity change of the ball to gain posession
        ba_threshold (float): minimal angle change of the ball to gain posession
        duels (pd.Series, optional): pd Series with the duels in the match. Is only used
        when tracking and event data is synchronised. Default is None.

    Returns:
        np.ndarray: array with bools with if the player gained posession of the ball
        per posession.
    """
    # check for every posession if the ball angle changes above threshold from first
    # to last frame of the proposed posession
    start_idxs_plus_1 = np.clip(posession_start_idxs + 1, 0, len(tracking_data))
    end_idxs_minus_1 = np.clip(posession_end_idxs - 1, 0, len(tracking_data))

    incomming_vectors = (
        tracking_data.loc[start_idxs_plus_1, ["ball_x", "ball_y"]].values
        - tracking_data.loc[posession_start_idxs, ["ball_x", "ball_y"]].values
    )
    outgoing_vectors = (
        tracking_data.loc[posession_end_idxs, ["ball_x", "ball_y"]].values
        - tracking_data.loc[end_idxs_minus_1, ["ball_x", "ball_y"]].values
    )
    ball_angles = get_smallest_angle(
        incomming_vectors, outgoing_vectors, angle_format="degree"
    )
    ball_angle_above_threshold = ball_angles > ba_threshold

    # check for every posession if the ball speed changes above threshold anywhere
    # in first to last frame of the proposed posession
    ball_speed_change = tracking_data["ball_velocity"].diff().abs() > bv_threshold
    interval_idxs = np.concatenate([[0], posession_end_idxs])
    intervals = [
        (start, end) for start, end in zip(interval_idxs[:-1], interval_idxs[1:])
    ]
    ball_speed_change_above_threshold = np.array(
        [np.any(ball_speed_change[start : end + 1]) for start, end in intervals]
    )

    # check if the player attempts an on ball action
    if duels is not None and "databallpy_event" in tracking_data.columns:
        has_duel_per_frame = ~pd.isnull(duels)
        on_ball_actions = ["pass", "shot", "dribble"]
        has_on_ball_action_per_frame = tracking_data["databallpy_event"].isin(
            on_ball_actions
        )
        has_event = np.array(
            [
                np.any(has_on_ball_action_per_frame[start : end + 1])
                for start, end in intervals
            ]
        )
        has_duel = np.array(
            [np.any(has_duel_per_frame[start : end + 1]) for start, end in intervals]
        )
        has_on_ball_action = np.logical_and(has_event, ~has_duel)
    else:
        has_on_ball_action = np.zeros_like(ball_speed_change_above_threshold)

    # check for valid gain
    valid_gains = np.logical_or(
        ball_angle_above_threshold,
        np.logical_or(ball_speed_change_above_threshold, has_on_ball_action),
    )
    return valid_gains
