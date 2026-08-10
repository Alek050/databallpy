from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from databallpy.utils.utils import sigmoid
from databallpy.utils.warnings import deprecated


@lru_cache(maxsize=8)
def _get_grid(
    pitch_length: float, pitch_width: float, n_x_bins: int, n_y_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create the (constant) grid of cell centers used for the pitch control surface.

    The result is cached since the grid only depends on the pitch dimensions and the
    number of bins, and is thus identical for every frame.

    Args:
        pitch_length (float): The length (x) of the pitch in meters.
        pitch_width (float): The width (y) of the pitch in meters.
        n_x_bins (int): The number of cells in the width (x) direction.
        n_y_bins (int): The number of cells in the height (y) direction.

    Returns:
        tuple[np.ndarray, np.ndarray]: Grid created with np.meshgrid.
    """
    x_grid, y_grid = np.meshgrid(
        np.linspace(-pitch_length / 2, pitch_length / 2, n_x_bins),
        np.linspace(-pitch_width / 2, pitch_width / 2, n_y_bins),
    )
    x_grid.flags.writeable = False
    y_grid.flags.writeable = False
    return x_grid, y_grid


@deprecated(
    "The get_approximate_voronoi function is deprecated and will removed in version 0.8.0. Please use Game.TrackingData.get_approximate_voronoi() instead."
)
def get_approximate_voronoi(
    tracking_data: pd.Series | pd.DataFrame,
    pitch_dimensions: list[float, float],
    n_x_bins: int = 106,
    n_y_bins: int = 68,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the nearest player to each cell center in a grid of cells covering the
    pitch.

    Args:
        tracking_data (pd.Series | pd.DataFrame): The tracking data. If it is a
            pd.Series, it is assumed that it contains data of a single frame. If it
            is a pd.DataFrame it is assumed that it contains tracking data of multiple
            frames.
        pitch_dimensions (list[float, float]): The dimensions of the pitch.
        n_x_bins (int, optional): The number of cells in the width (x) direction.
            Defaults to 106.
        n_y_bins (int, optional): The number of cells in the height (y) direction.
            Defaults to 68.

    Returns:
        tuple[np.ndarray, np.ndarray]: The distances to the nearest player for each
            cell center and the column ids of the nearest player. If tracking_data is
            a pd.Series, the shape will be (n_y_bins x n_x_bins), otherwise
            (len(tracking_data) x n_y_bins x n_x_bins).
    """
    pitch_length, pitch_width = pitch_dimensions
    x_bins = np.linspace(-pitch_length / 2, pitch_length / 2, n_x_bins + 1)
    y_bins = np.linspace(-pitch_width / 2, pitch_width / 2, n_y_bins + 1)
    cell_centers_x, cell_centers_y = np.meshgrid(
        x_bins[:-1] + np.diff(x_bins) / 2, y_bins[:-1] + np.diff(y_bins) / 2
    )

    if isinstance(tracking_data, pd.Series):
        tracking_data = tracking_data.to_frame().T

    all_distances = np.empty((len(tracking_data), n_y_bins, n_x_bins), dtype=np.float32)
    all_assigned_players = np.empty((len(tracking_data), n_y_bins, n_x_bins), dtype="U7")
    for i, (_, frame) in enumerate(tracking_data.iterrows()):
        player_column_ids = np.array(
            [
                column[:-2]
                for column in frame.index
                if column[-2:] in ["_x", "_y"]
                and not pd.isnull(frame[column])
                and "ball" not in column
            ]
        )
        player_positions = np.array(
            [
                [frame[column + "_x"], frame[column + "_y"]]
                for column in player_column_ids
            ]
        ).astype(np.float64)

        tree = KDTree(player_positions)
        cell_centers = np.column_stack((cell_centers_x.ravel(), cell_centers_y.ravel()))
        distances, nearest_player_indices = tree.query(cell_centers)

        all_assigned_players[i] = player_column_ids[nearest_player_indices].reshape(
            n_y_bins, n_x_bins
        )
        all_distances[i] = distances.reshape(n_y_bins, n_x_bins)

    if all_distances.shape[0] == 1:
        all_distances = all_distances[0]
        all_assigned_players = all_assigned_players[0]

    return all_distances, all_assigned_players


@deprecated(
    "The get_pitch_control function is deprecated and will removed in version 0.8.0. Please use Game.TrackingData.get_pitch_control() instead."
)
def get_pitch_control(
    tracking_data: pd.DataFrame,
    pitch_dimensions: list[float, float],
    n_x_bins: int = 106,
    n_y_bins: int = 68,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> np.ndarray:
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
    The values are then passed through a sigmoid function to obtain the pitch control
    values within a [0, 1] range. Values near 1 indicate high pitch control by the home
    team, while values near 0 indicate high pitch control by the away team.

    Args:
        tracking_data (pd.DataFrame): tracking data.
        pitch_dimensions (list[float, float]): The dimensions of the pitch.
        n_x_bins (int, optional): The number of cells in the width (x) direction.
            Defaults to 106.
        n_y_bins (int, optional): The number of cells in the height (y) direction.
            Defaults to 68.
        start_idx (int, optional): The starting index of the period. Defaults to None.
        end_idx (int, optional): The ending index of the period. Defaults to None.

    Returns:
        np.ndarray: 3d pitch control values across the grid.
            Size is (len(tracking_data), grid[0].shape[0], grid[0].shape[1]).
    """

    start_idx = tracking_data.index[0] if start_idx is None else start_idx
    end_idx = tracking_data.index[-1] if end_idx is None else end_idx
    tracking_data = tracking_data.loc[start_idx:end_idx]

    pitch_control = np.zeros((len(tracking_data), n_y_bins, n_x_bins), dtype=np.float32)

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
            pitch_dimensions,
            n_x_bins,
            n_y_bins,
            player_ball_distances=player_ball_distances.loc[idx],
        )
    return np.array(pitch_control)


def get_pitch_control_surface_radius(
    distance_to_ball: float, min_r: float = 4.0, max_r: float = 10.0
) -> float:
    """
    Calculate the pitch control surface radius based on the distance to the ball.
    Note that the article does not provide the mathematical function for this formula,
    only a figure (Figure 9 in Appendix 1.). The constants (972 and 3) are
    obtained by visual inspection. The value is refered to as R_i(t) in the article.

    Args:
        distance_to_ball (float): Distance from the player to the ball.
        min_r (float, optional): The minimal influence radius of a player.
            Defaults to 4.0.
        max_r (float, optional): The maximal influence radius of a player.
            Defaults to 10.0

    Returns:
        float: Pitch control surface radius.
    """
    return min(min_r + distance_to_ball**3 / 972, max_r)


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
    influence_radius = get_pitch_control_surface_radius(distance_to_ball)
    ratio_of_max_speed = np.power(min(speed_magnitude, max_speed), 2) / np.power(
        max_speed, 2
    )
    return np.array(
        [
            [(influence_radius + (influence_radius * ratio_of_max_speed)), 0],
            [0, max(influence_radius - (influence_radius * ratio_of_max_speed), 0.001)],
        ]
    )


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

    covariance_matrix = np.dot(
        np.dot(np.dot(rotation_matrix, scaling_matrix), scaling_matrix),
        np.linalg.inv(rotation_matrix),
    )

    return covariance_matrix


def get_mean_position_of_influence(
    x_val: float,
    y_val: float,
    vx_val: float,
    vy_val: float,
) -> list[float]:
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
    scaling_matrix = calculate_scaling_matrix(np.hypot(vx_val, vy_val), distance_to_ball)
    covariance_matrix = calculate_covariance_matrix(vx_val, vy_val, scaling_matrix)

    # evaluate the gaussian inline using the closed form inverse of the 2x2 covariance
    # matrix. The normalisation constant of the gaussian cancels since the influence
    # values are normalised by their maximum.
    (var_x, cov_xy), (cov_yx, var_y) = covariance_matrix
    determinant = var_x * var_y - cov_xy * cov_yx

    delta_x = grid[0] - mean[0]
    delta_y = grid[1] - mean[1]
    exponent = (
        -0.5
        * (
            delta_x * (var_y * delta_x - cov_xy * delta_y)
            + delta_y * (var_x * delta_y - cov_yx * delta_x)
        )
        / determinant
    )
    return np.exp(exponent - exponent.max())


def get_team_influence(
    frame: pd.Series,
    col_ids: list,
    grid: list,
    player_ball_distances: pd.Series | None = None,
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


def get_pitch_control_single_frame(
    frame: pd.Series,
    pitch_dimensions: list[float, float],
    n_x_bins: int = 106,
    n_y_bins: int = 68,
    player_ball_distances: pd.Series = None,
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
    The values are then passed through a sigmoid function to obtain the pitch control
    values within a [0, 1] range. Values near 1 indicate high pitch control by the home
    team, while values near 0 indicate high pitch control by the away team.

    Args:
        frame (pd.Series): Row of the tracking data.
        pitch_dimensions (list[float, float]): The dimensions of the pitch.
        n_x_bins (int, optional): The number of cells in the width (x) direction.
            Defaults to 106.
        n_y_bins (int, optional): The number of cells in the height (y) direction.
            Defaults to 68.
        player_ball_distances (pd.Series, optional): Precomputed player ball distances.
            Defaults to None.

    Returns:
        np.ndarray: 2D pitch control values across a 2d grid.
            Size of the grid is n_y_bins x n_x_bins.
    """

    grid = _get_grid(pitch_dimensions[0], pitch_dimensions[1], n_x_bins, n_y_bins)

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
    return sigmoid(team_influence_away - team_influence_home)
