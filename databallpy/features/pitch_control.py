from typing import List

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from databallpy.utils.logging import create_logger

LOGGER = create_logger(__name__)


def get_pitch_control_period(tracking_data: pd.DataFrame, grid: list) -> np.ndarray:
    """
    Calculate the pitch control surface for a given period of time. The pitch control
    surface is the sum of the team influences of the two teams. The team influence is
    the sum of the individual player influences of the team. The player influence is
    calculated using the statistical technique presented in the article "Wide Open
    Spaces" by Fernandez & Born (2018). It incorporates the position, velocity, and
    distance to the ball of a given player to determine the influence degree at each
    location on the field. The bivariate normal distribution is utilized to model the
    player's influence, and the result is normalized to obtain values within a [0, 1]
    range.
    The returned values are, for every cell, the sum of influence on that cell of the
    home team minus the sum of influence on that cell of the away team. Note that a
    team can have a maximum influance of 11, if the means of all 11 players are on that
    cell and the radius of the covariance matrix is almost fully within the cell.
    This is never the case, so the values are very low, yet informative and
    interpretable.
    A good way to visualise the values is by using a sigmoid {1 /(1 + e^-x)}.
    This will force the values towards 0 and 1. 0 means the away team controls
    that cell and 1 means the home team controls that cell.

    Args:
        tracking_data (pd.DataFrame): tracking data.
        grid (list): Grid created with np.meshgrid(np.linspace(), np.linspace()).

    Returns:
        np.ndarray: 3d pitch control values across the grid.
            Size is (len(tracking_data), grid[0].shape[0], grid[0].shape[1]).
    """
    try:
        pitch_control = np.zeros(
            (len(tracking_data), grid[0].shape[0], grid[0].shape[1])
        )

        # precompute player ball distances
        col_ids = [
            x[:-2]
            for x in tracking_data.columns
            if ("home" in x or "away" in x) and x[-2:] == "_x"
        ]
        player_ball_distances = pd.DataFrame(columns=col_ids, index=tracking_data.index)
        for col_id in col_ids:
            player_ball_distances[col_id] = np.linalg.norm(
                tracking_data[[f"{col_id}_x", f"{col_id}_y"]].values
                - tracking_data[["ball_x", "ball_y"]].values,
                axis=1,
            )

        for i, idx in enumerate(tracking_data.index):
            pitch_control[i] = get_pitch_control_single_frame(
                tracking_data.loc[idx],
                grid,
                player_ball_distances=player_ball_distances.loc[idx],
            )
        return np.array(pitch_control)
    except Exception as e:
        LOGGER.exception(
            f"Found an unexpected exception in get_pitch_control_period()\n{e}"
        )
        raise e


def get_pitch_control_single_frame(
    frame: pd.Series, grid: list, player_ball_distances: pd.Series = None
) -> np.ndarray:
    """
    Calculate the pitch control surface at a given frame. The pitch control surface
    is the sum of the team influences of the two teams. The team influence is the
    sum of the individual player influences of the team. The player influence is
    calculated using the statistical technique presented in the article "Wide Open
    Spaces" by Fernandez & Born (2018). It incorporates the position, velocity, and
    distance to the ball of a given player to determine the influence degree at each
    location on the field. The bivariate normal distribution is utilized to model the
    player's influence, and the result is normalized to obtain values within a [0, 1]
    range.
    The returned values are, for every cell, the sum of influence on that cell of the
    home team minus the sum of influence on that cell of the away team. Note that a
    team can have a maximum influance of 11, if the means of all 11 players are on that
    cell and the radius of the covariance matrix is almost fully within the cell.
    This is never the case, so the values are very low, yet informative and
    interpretable.
    A good way to visualise the values is by using a sigmoid {1 /(1 + e^-x)}.
    This will force the values towards 0 and 1. 0 means the away team controls
    that cell and 1 means the home team controls that cell.

    Args:
        frame (pd.Series): Row of the tracking data.
        grid (list): Grid created with np.meshgrid.
        player_ball_distances (pd.Series, optional): Precomputed player ball distances.
            Defaults to None.

    Returns:
        np.ndarray: 2D pitch control values across the grid.
            Size is (grid[0].shape[0], grid[0].shape[1]).
    """
    try:
        home_col_ids = [x[:-2] for x in frame.index if "home" in x and x[-2:] == "_x"]
        away_col_ids = [x[:-2] for x in frame.index if "away" in x and x[-2:] == "_x"]

        team_influence_home = get_team_influence(
            frame,
            col_ids=home_col_ids,
            grid=grid,
            player_ball_distances=player_ball_distances,
        )
        team_influence_away = get_team_influence(
            frame,
            col_ids=away_col_ids,
            grid=grid,
            player_ball_distances=player_ball_distances,
        )
        return team_influence_home - team_influence_away
    except Exception as e:
        LOGGER.exception(f"Found an unexpected exception in get_pitch_control()\n{e}")
        raise e


def get_team_influence(
    frame: pd.Series, col_ids: list, grid: list, player_ball_distances: pd.Series = None
) -> np.ndarray:
    """
    Calculate the team influence of a given team at a given frame. The team influence
    is the sum of the individual player influences of the team.

    Args:
        frame (pd.Series): Row of the tracking data.
        col_ids (list): List of column ids of the players in the team.
        grid (list): Grid created with np.meshgrid.
        player_ball_distances (pd.Series, optional): Precomputed player ball distances.

    Returns:
        np.ndarray: Team influence values across the grid.
    """
    player_influence = []
    for col_id in col_ids:
        if pd.isnull(frame[f"{col_id}_vx"]):
            continue

        if player_ball_distances is not None:
            distance_to_ball = player_ball_distances.loc[col_id]
        else:
            distance_to_ball = np.linalg.norm(
                frame[[f"{col_id}_x", f"{col_id}_y"]].values
                - frame[["ball_x", "ball_y"]].values
            )

        player_influence.append(
            get_player_influence(
                x_val=frame[f"{col_id}_x"],
                y_val=frame[f"{col_id}_y"],
                vx_val=frame[f"{col_id}_vx"],
                vy_val=frame[f"{col_id}_vy"],
                distance_to_ball=distance_to_ball,
                grid=grid,
            )
        )
    team_influence = np.sum(player_influence, axis=0)
    return team_influence


def get_player_influence(
    x_val: float,
    y_val: float,
    vx_val: float,
    vy_val: float,
    distance_to_ball: float,
    grid: np.ndarray,
) -> np.ndarray:
    """
    Calculate player influence across the grid based on the statistical technique
    presented in the article "Wide Open Spaces" by Fernandez & Born (2018).
    It incorporates the position, velocity, and distance to the ball of a given
    player to determine the influence degree at each location on the field. The
    bivariate normal distribution is utilized to model the player's influence,
    and the result is normalized so that the sum of the players influence over
    all cells in the grid is 1. Thus, the value in a cell in the grid is the ratio
    of influence of that player in that cell.

    Args:
        x_val (float): x-coordinate of the player's current position in meters.
        y_val (float): y-coordinate of the player's current position in meters.
        vx_val (float): Velocity of the player in the x-direction in m/s.
        vy_val (float): Velocity of the player in the y-direction in m/s.
        distance_to_ball (float): distance between the ball and the (x_val, y_val) in
            meters.
        grid (np.ndarray]): Grid created with np.meshgrid.

    Returns:
        np.ndarray: Player influence values across the grid.
    """
    mean = get_mean_position_of_influence(x_val, y_val, vx_val, vy_val)
    scaling_matrix = calculate_scaling_matrix(
        np.hypot(vx_val, vy_val), distance_to_ball
    )
    covariance_matrix = calculate_covariance_matrix(vx_val, vy_val, scaling_matrix)

    grid_size = grid[0].shape
    positions = np.vstack([grid[0].ravel(), grid[1].ravel()]).T

    distribution = multivariate_normal(mean=mean, cov=covariance_matrix)
    influence_values = distribution.pdf(positions)
    influence_values = normalize_values(influence_values)
    return influence_values.reshape(grid_size[0], grid_size[1])


def normalize_values(influence_values):
    """
    Normalize influence values as ratio of the total.

    Args:
        influence_values (numpy.ndarray): Array with influence values.

    Returns:
        numpy.ndarray: Normalized influence values.
    """
    return influence_values / np.sum(influence_values)


def get_mean_position_of_influence(
    x_val: float,
    y_val: float,
    vx_val: float,
    vy_val: float,
) -> List[float]:
    """
    Calculate the mean position of player influence over time using the statistical
    technique presented in the article "Wide Open Spaces" by Fernandez & Born (2018).
    It considers the player's current position, velocity, and a specified time step.
    The mean position is obtained by translating the player's location at a given
    time by half the magnitude of the speed vector.

    This refers to u_i(t) definedin formula 21 of the appendix.

    Args:
        x_val (float): x-coordinate of the player's current position in meters.
        y_val (float): y-coordinate of the player's current position in meters.
        vx_val (float): Velocity of the player in the x-direction in m/s.
        vy_val (float): Velocity of the player in the y-direction in m/s.

    Returns:
        List[float]: Mean position [x, y] of the player's influence.
    """
    return np.array([x_val + 0.5 * vx_val, y_val + 0.5 * vy_val])


def calculate_covariance_matrix(
    vx_val: float,
    vy_val: float,
    scaling_matrix: np.ndarray,
) -> np.ndarray:
    """
    Calculate the covariance matrix using the statistical technique presented in the
    article "Wide Open Spaces" by Fernandez & Born (2018).
    It dynamically adjusts the covariance matrix to provide a player dominance
    distribution that considers both location and velocity. The method involves the
    singular value decomposition (SVD) algorithm, expressing the covariance matrix
    in terms of eigenvectors and eigenvalues. The rotation matrix and scaling
    matrix are then derived, incorporating the rotation angle of the speed vector
    and scaling factors in the x and y directions.

    The calculated value is COV_i(t) as defined in formula 20 of the appendix.

    Args:
        vx_val (float): Velocity of the player in the x-direction in m/s.
        vy_val (float): Velocity of the player in the y-direction in m/s.
        scaling_matrix (np.ndarray): 2 by 2 array based on the velocity of a
            player and its distance to the ball. Determines the spread af the
            covariance matrix.

    Returns:
        np.ndarray: Covariance matrix.
    """
    rotation_angle = np.arctan2(vy_val, vx_val)
    rotation_matrix = np.array(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )

    # Combine rotation and scaling matrices to get covariance matrix (COV)
    covariance_matrix = np.dot(
        rotation_matrix,
        np.dot(scaling_matrix, np.dot(scaling_matrix, np.linalg.inv(rotation_matrix))),
    )

    return covariance_matrix


def calculate_scaling_matrix(
    speed_magnitude: float,
    distance_to_ball: float,
    max_speed: float = 13.0,
) -> np.ndarray:
    """
    Calculate the scaling factors using the statistical technique presented in the
    article "Wide Open Spaces" by Fernandez & Born (2018).
    Determines the scaling factors based on the player's speed magnitude and the
    distance to the ball. The range of the player's pitch control surface radius is
    defined, and scaling factors are calculated. The scaling matrix is then expanded
    in the x direction and contracted in the y direction using these factors.

    This is formula 19 in the appendix, where S_i(t) is defined.

    Args:
        speed_magnitude (float): Magnitude of the player's speed in m/s.
        distance_to_ball (float): Distance from the player to the ball in meters.
        max_speed (float, optional): Max speed a player can have in m/s.
            Defaults to 13.0.

    Returns:
        np.ndarray: Scaling matrix.
    """

    # Refered to as R_i(t) in the article
    influence_radius = calculate_pitch_control_surface_radius(distance_to_ball)
    ratio_of_max_speed = np.power(speed_magnitude, 2) / np.power(max_speed, 2)

    scaling_matrix = np.array(
        [
            [(influence_radius + (influence_radius * ratio_of_max_speed)) / 2, 0],
            [0, (influence_radius - (influence_radius * ratio_of_max_speed)) / 2],
        ]
    )
    return scaling_matrix


def calculate_pitch_control_surface_radius(
    distance_to_ball: float, min_r: float = 4.0, max_r: float = 10.0
) -> float:
    """
    Calculate the pitch control surface radius based on the distance to the ball.
    Note that the article does not provide the mathematical function for this formula,
    only a figure (Figure 9 in Appendix 1.). The constants (0.00025 and 3.5) are
    obtained by visual inspection.

    Args:
        distance_to_ball (float): Distance from the player to the ball.
        min_r (float, optional): The minimal influence radius of a player.
            Defaults to 4.0.
        max_r (float, optional): The maximal influence radius of a player.
            Defaults to 10.0

    Returns:
        float: Pitch control surface radius.
    """
    val = min_r + 0.00025 * np.power(distance_to_ball, 3.5)
    return (
        min(val, max_r) * 1.8
    )  # 1.8 is a scaling factor to make the pitch control surface a bit bigger
