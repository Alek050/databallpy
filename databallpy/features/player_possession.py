import numpy as np
import pandas as pd

from databallpy.features import get_smallest_angle
from databallpy.utils.constants import MISSING_INT
from databallpy.utils.warnings import deprecated


@deprecated(
    "The get_individual_player_possession function is deprecated and will removed in version 0.8.0. Please use Game.TrackingData.add_individual_player_possession() instead."
)
def get_individual_player_possession(
    tracking_data: pd.DataFrame,
    pz_radius: float = 1.5,
    bv_threshold: float = 5.0,
    ba_threshold: float = 10.0,
    min_frames_pz: int = 0,
    inplace: bool = False,
) -> None | np.ndarray:
    """Function to calculate the individual player possession based on the tracking
    data. The method uses the methodology of the paper of  Vidal-Codina et al. (2022):
    "Automatic Event Detection in Football Using Tracking Data".


    Args:
        tracking_data (pd.DataFrame): Tracking data with player positions.
        pz_radius (float, optional): The radius of the possession zone constant.
            Defaults to 1.5.
        bv_threshold (float, optional): The ball velocity threshold in m/s.
            Defaults to 5.0.
        ba_threshold (float, optional): The ball angle threshold in degrees.
            Defaults to 10.0.
        min_frames_pz (int, optional): The minimum number of frames that the ball
            has to be in the possession zone to be considered as a possession.
            Defaults to 0.
        inplace (bool, optional): If True, the tracking data will get a new column
            with `player_possession`. Defaults to False.

    Returns:
        None | np.ndarray: If inplace is True, the tracking data will be updated with
            a new column `player_possession`. If inplace is False, the function will
            return the player possession as a np.ndarray.
    """
    if "ball_velocity" not in tracking_data.columns:
        raise ValueError(
            "The tracking data should have a column 'ball_velocity'. Use the "
            "add_velocity function to add the ball velocity."
        )

    distances_df = get_distance_between_ball_and_players(tracking_data)
    initial_possession = get_initial_possessions(pz_radius, distances_df)
    possession_start_idxs, possession_end_idxs = get_start_end_idxs(initial_possession)
    valid_gains = get_valid_gains(
        tracking_data,
        possession_start_idxs,
        possession_end_idxs,
        bv_threshold,
        ba_threshold,
        min_frames_pz,
    )
    valid_gains_start_idxs, ball_losses_idxs = get_ball_losses_and_updated_gain_idxs(
        possession_start_idxs, possession_end_idxs, valid_gains, initial_possession
    )

    possession = np.full(len(tracking_data), None, dtype=object)
    for start, end in zip(valid_gains_start_idxs, ball_losses_idxs):
        possession[start:end] = initial_possession[start]

    alive_mask = tracking_data["ball_status"] == "alive"
    possession[~alive_mask] = None

    if inplace:
        tracking_data["player_possession"] = possession
    else:
        return possession


def get_distance_between_ball_and_players(tracking_data: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized function to calculate the distances between the ball and all players using
    vectorized operations.

    Args:
        tracking_data (pd.DataFrame): DataFrame with tracking data over which to
        calculate the distances.

    Returns:
        pd.DataFrame: DataFrame with the distances between the ball and all players.
    """

    player_columns = [
        col for col in tracking_data.columns if "_x" in col and "ball" not in col
    ]
    ball_x, ball_y = tracking_data["ball_x"].values, tracking_data["ball_y"].values

    distances_df = pd.DataFrame(index=tracking_data.index)
    for col in player_columns:
        player_x, player_y = (
            tracking_data[f"{col}"].values,
            tracking_data[f"{col[:-2]}_y"].values,
        )
        distances = np.sqrt((ball_x - player_x) ** 2 + (ball_y - player_y) ** 2)
        distances_df[col[:-2]] = distances

    return distances_df


def get_initial_possessions(
    pz_radius: float,
    distances_df: pd.DataFrame,
) -> np.ndarray:
    """
    Calculate initial ball possession based on proximity and duration within the
    possession zone (PZ).

    Args:
        pz_radius (float): Radius of the possession zone in meters.
        distances_df (pd.DataFrame): DataFrame with distances between the
        ball and players.

    Returns:
        np.ndarray: Array with the initial possession of the ball.
    """
    filled_distances_df = distances_df.fillna(np.inf)
    closest_player = filled_distances_df.idxmin(axis=1)
    close_enough = filled_distances_df.min(axis=1) < pz_radius
    return np.where(close_enough, closest_player, None)


def get_valid_gains(
    tracking_data: pd.DataFrame,
    possession_start_idxs: np.ndarray,
    possession_end_idxs: np.ndarray,
    bv_threshold: float,
    ba_threshold: float,
    min_frames_pz: int,
) -> np.ndarray:
    """Function to check if, within a given period, a player gains possession of the
    ball. Possession is gained if the ball speed changes at least bs_threshold m/s or
    the ball changes direction (> ba_threshold) between the first and the last
    proposed possession frame.

    Args:
        tracking_data (pd.DataFrame): pandas df with tracking data over which to
            calculate the player possession.
        possession_start_idxs (np.ndarray): array with the starting indexes of the
            proposed possessions.
        possession_end_idxs (np.ndarray): array with the ending indexes of the proposed
            possessions.
        bv_threshold (float): minimal velocity change of the ball to gain possession
        ba_threshold (float): minimal angle change of the ball to gain possession
        min_frames_pz (int): minimal number of frames the ball has to be in the
            possession zone to be considered as a possession.

    Returns:
        np.ndarray: array with bools with if the player gained possession of the ball
        per possession.
    """

    ball_angle_condition = get_ball_angle_condition(
        tracking_data, possession_start_idxs, possession_end_idxs, ba_threshold
    )

    ball_speed_condition = get_ball_speed_condition(
        tracking_data, possession_start_idxs, possession_end_idxs, bv_threshold
    )

    min_frames_condition = (
        possession_end_idxs - possession_start_idxs + 1 >= min_frames_pz
    )

    return np.logical_and(
        min_frames_condition, np.logical_or(ball_angle_condition, ball_speed_condition)
    )


def get_start_end_idxs(pz_initial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Function to get the starting and ending indexes of the proposed possessions
    based on the initial possession of the ball. The proposed possessions are periods
    where the possession of the ball changes.

    Args:
        pz_initial (np.ndarray): The initial possession of the ball.

    Returns:
        tuple[np.ndarray, np.ndarray]: The starting and ending indexes of the proposed
        possessions.
    """

    shifting_idxs = np.where(pz_initial[:-1] != pz_initial[1:])[0]
    shifting_idxs = np.concatenate([[-1], shifting_idxs, [len(pz_initial) - 1]])

    possession_start_idxs = shifting_idxs[:-1] + 1
    possession_end_idxs = shifting_idxs[1:]

    none_idxs = np.where(pd.isnull(pz_initial[possession_start_idxs]))[0]
    possession_start_idxs = np.delete(possession_start_idxs, none_idxs)
    possession_end_idxs = np.delete(possession_end_idxs, none_idxs)

    return possession_start_idxs, possession_end_idxs


def get_ball_speed_condition(
    tracking_data: pd.DataFrame,
    possession_start_idxs: np.ndarray,
    possession_end_idxs: np.ndarray,
    bv_threshold: float,
) -> np.ndarray:
    """Function to check if, within the pz zone period, the ball changes speed
    enough to count as a possession gain based on the ball speed condition.

    Args:
        tracking_data (pd.DataFrame): Tracking data with player positions.
        possession_start_idxs (np.ndarray): The starting indexes of the proposed
            possessions.
        possession_end_idxs (np.ndarray): The ending indexes of the proposed
            possessions.
        bv_threshold (float): The threshold for the ball speed condition in m/s.

    Returns:
        np.ndarray: Array with bools indicating if the ball speed condition is met
            for each proposed possession.
    """
    ball_vel = pd.concat(
        [pd.Series(data=[0]), tracking_data["ball_velocity"]], ignore_index=True
    )
    ball_speed_change = ball_vel.diff().abs()[1:] > bv_threshold
    intervals = [
        (start, end) for start, end in zip(possession_start_idxs, possession_end_idxs)
    ]

    # Prevent index out of bounds
    if intervals[-1][1] == tracking_data.index[-1]:
        intervals[-1] = (intervals[-1][0], intervals[-1][1] - 1)

    return np.array(
        [np.any(ball_speed_change[start : end + 1]) for start, end in intervals]
    )


def get_ball_angle_condition(
    tracking_data: pd.DataFrame,
    possession_start_idxs: np.ndarray,
    possession_end_idxs: np.ndarray,
    ba_threshold: float,
) -> np.ndarray:
    """Function to check if, within the pz zone period, the ball changes direction
    enough to count as a possession gain based on the ball angle condition.

    Args:
        tracking_data (pd.DataFrame): Tracking data with player positions.
        possession_start_idxs (np.ndarray): The starting indexes of the proposed
            possessions.
        possession_end_idxs (np.ndarray): The ending indexes of the proposed
            possessions.
        ba_threshold (float): The threshold for the ball angle condition in degrees.

    Returns:
        np.ndarray: Array with bools indicating if the ball angle condition is met
            for each proposed possession.
    """
    start_idxs_minus_1 = np.clip(possession_start_idxs - 1, 0, tracking_data.index[-1])
    end_idxs_plus_1 = np.clip(possession_end_idxs + 1, 0, tracking_data.index[-1])

    incomming_vectors = (
        tracking_data.loc[possession_start_idxs, ["ball_x", "ball_y"]].values
        - tracking_data.loc[start_idxs_minus_1, ["ball_x", "ball_y"]].values
    )

    outgoing_vectors = (
        tracking_data.loc[end_idxs_plus_1, ["ball_x", "ball_y"]].values
        - tracking_data.loc[possession_end_idxs, ["ball_x", "ball_y"]].values
    )
    ball_angles = get_smallest_angle(
        incomming_vectors, outgoing_vectors, angle_format="degree"
    )

    return ball_angles > ba_threshold


def get_ball_losses_and_updated_gain_idxs(
    possession_start_idxs: np.ndarray,
    possession_end_idxs: np.ndarray,
    valid_gains: np.ndarray,
    initial_possession: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Function to get the ball losses and updated gain indexes based on the
    initial possession of the ball.

    Args:
            possession_start_idxs (np.ndarray): The starting indexes of the
                proposed possessions.
            possession_end_idxs (np.ndarray): The ending indexes of the
                proposed possessions.
            valid_gains (np.ndarray): The valid gains of the ball.
            initial_possession (np.ndarray): The initial possession of the ball.

    Returns:
            tuple[np.ndarray, np.ndarray]: The starting indexes of the valid gains and
            the ball losses.
    """
    ball_losses_idxs = np.full(len(possession_start_idxs), MISSING_INT, dtype=int)
    last_player = None
    for i, (start, end, is_valid_gain) in enumerate(
        zip(possession_start_idxs, possession_end_idxs, valid_gains)
    ):
        player = initial_possession[start]
        if player == last_player:
            ball_losses_idxs[i - 1] = end
        elif is_valid_gain:
            ball_losses_idxs[i] = end
            last_player = player

    valid_gains_start_idxs = possession_start_idxs[
        (ball_losses_idxs != MISSING_INT) & valid_gains
    ]
    ball_losses_idxs = ball_losses_idxs[ball_losses_idxs != MISSING_INT]

    return valid_gains_start_idxs, ball_losses_idxs
