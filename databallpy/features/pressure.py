import math

import numpy as np
import pandas as pd

from databallpy.features.angle import get_smallest_angle
from databallpy.utils.logging import logging_wrapper
from databallpy.utils.warnings import deprecated


@logging_wrapper(__file__)
@deprecated(
    "The get_pressure_on_player function is deprecated and will removed in version 0.8.0. Please use Game.TrackingData.get_pressure_on_player() instead."
)
def get_pressure_on_player(
    td_frame: pd.Series,
    column_id: str,
    *,
    pitch_size: list[float, float],
    d_front: str | float = "variable",
    d_back: float = 3.0,
    q: float = 1.75,
):
    """
    Function to calculate the pressure in accordance with "Visual Analysis of Pressure
    in Soccer", Adrienko et al (2016). In short: pressure is determined as the sum of
    pressure of all opponents, which is a function of the angle and the distance to the
    player. This function calculates the pressure for a single player.

    :param td_frame: pandas Series with tracking data frame of all players
    :param column_id: str, column name of which player to analyse
    :param pitch_size: list, length and width of the pitch.
    :param d_front: numeric or str, distance in meters of the front of the pressure oval
                    if "variable": d_front will be variable based on the location on
                    the field from the article of Mat Herold et al (2022).
    :param d_back: float, dinstance in meters of the back of the pressure oval
    :param q: float, quotient of how fast pressure should increase/decrease as distance
              to the player changes.
    :returns: numpy array with pressure on player over the length of the df
    """

    if d_front == "variable":
        d_front = calculate_variable_dfront(
            td_frame, column_id, pitch_length=pitch_size[0]
        )

    team = column_id[:4]
    opponent_team = "away" if team == "home" else "home"
    tot_pressure = 0
    player_xy = [td_frame[column_id + "_x"], td_frame[column_id + "_y"]]

    for opponent_column_id in [
        x[:-2] for x in td_frame.index if opponent_team in x and "_x" in x
    ]:
        opponent_xy = [
            td_frame[opponent_column_id + "_x"],
            td_frame[opponent_column_id + "_y"],
        ]
        player_opponent_distance = math.dist(player_xy, opponent_xy)
        # opponent not close enough to exert pressure on the player
        if player_opponent_distance > max([d_front, d_back]):
            continue

        z = calculate_z(
            td_frame, column_id, opponent_column_id, pitch_length=pitch_size[0]
        )
        variable_l = calculate_l(d_back, d_front, z)

        current_pressure = (
            pd.to_numeric(
                (1 - player_opponent_distance / variable_l), errors="coerce"
            ).clip(0)
            ** q
            * 100
        )

        current_pressure = 0 if pd.isnull(current_pressure) else current_pressure
        tot_pressure += current_pressure

    return tot_pressure


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
