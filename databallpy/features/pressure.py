import math

import numpy as np
import pandas as pd

from databallpy.features.angle import get_smallest_angle


def calculate_variable_dfront(
    td_frame: pd.Series,
    column_id: str,
    max_d_front: int = 9,
    pitch_length: float = 105.0,
) -> float:
    """
    Function to calculate d_front over time according to the article of Herold et al
    2022: "Off-ball behavior in association football: A datadriven model to measure
    changes in individual defensive pressure".

    :param td_frame: pandas Series with tracking data frame of all players
    :param column_id: str, column name of the player of which to calculate the pressure
    :param max_d_front: int, maximal d_front, 9 meters according to the article
    :param pitch_length: float, length (x-direction) of the the pitch
    :returns: float, the pressure on the player
    """

    team = column_id[:4]
    goal_xy = [pitch_length / 2, 0] if team == "home" else [-pitch_length / 2, 0]
    player_xy = [td_frame[column_id + "_x"], td_frame[column_id + "_y"]]
    player_goal_distance = math.dist(goal_xy, player_xy)

    return max_d_front - 0.05 * (pitch_length - player_goal_distance)


def calculate_z(
    td_frame: pd.Series,
    column_id: str,
    opponent_column_id: str,
    pitch_length: float = 105.0,
) -> float:
    """
    Calculates the z value in accordance with the article of Adrienko et al (2016).
    Note that the angle calculation is slightly different here, therefore the formula is
    not z = (1 - cos(phi))/2, but z = (1 + cos(phi))/2. Phi is the angle between the
    direction of the player to the target (goal), and the vector of the opponent to the
    player.

    Args:
        td_frame (pd.Series): Tracking data frame of all players.
        column_id (str): Column name of the player for which to calculate the pressure.
        opponent_column_id (str): Column name of the player which is pressuring the
            player.
        pitch_length (float): Length (x-direction) of the pitch. Defaults to 105.0.

    Returns:
        float: z value of the pressure calculation.
    """
    team = column_id[:4]

    goal_xy = [pitch_length / 2, 0] if team == "home" else [-pitch_length / 2, 0]
    opponent_xy = [
        td_frame[opponent_column_id + "_x"],
        td_frame[opponent_column_id + "_y"],
    ]
    player_xy = [td_frame[column_id + "_x"], td_frame[column_id + "_y"]]

    # create vector between player and the goal
    player_goal_vec = [goal_xy[0] - player_xy[0], goal_xy[1] - player_xy[1]]

    # create vector between the player and the opponent
    player_opponent_vec = [opponent_xy[0] - player_xy[0], opponent_xy[1] - player_xy[1]]

    angles = get_smallest_angle(
        player_goal_vec, player_opponent_vec, angle_format="radian"
    )

    return (1.0 + np.cos(angles)) / 2.0


def calculate_l(d_back: float, d_front: float, z: float) -> float:
    """
    Calculates the L value of the pressure calculation in accordance with
    Adrienko et al. (2016).

    Args:
        d_back (float): Maximal distance to back from where pressure can be measured.
        d_front (float): Maximal distance in front of which pressure can be measured.
        z (list of float): Float values in accordance with formulas in Adrienko et al.
            (2016).

    Returns:
        float: L value of the pressure calculation.
    """
    variable_l = d_back + (d_front - d_back) * ((z**3 + 0.3 * z) / 1.3)
    variable_l = np.maximum(
        variable_l, 0.0001
    )  # set any values less than 0.0001 to 0.0001
    return variable_l
