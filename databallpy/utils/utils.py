from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.spatial import Delaunay

MISSING_INT = -999


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

def ratio_player_gaussian_in_shape(player_positions:np.ndarray, shape:type(Delaunay), y_diameter:float=1.0, x_diameter:float=0.6)->float:
    """Function to calculate the ratio of the player distribution that is within the
    given shape. This function is used to calculate the obstructive players.

    Args:
        player_positions (np.ndarray): x and y positions of the player
        shape (Delaunay): shape of where the player distribution should be in
        y_diameter (float, optional): The proposed diamater of the player in x 
            direction. Defaults to 1.0.
        x_diameter (float, optional): The proposed diameter of the player in y 
            direction. Defaults to 0.6.

    Returns:
        float: The ratio of the player distributions that is within the shape.
    """
    points = shape.points
    min_x = min(points[:, 0]) - 2
    max_x = max(points[:, 0]) + 2
    min_y =  min(points[:, 1]) - 2
    max_y = max(points[:, 1]) + 2
    square = Delaunay([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]])
    x, y = np.mgrid[min_x:max_x:500j, min_y:max_y:300j]
    positions = np.dstack((x, y))
    in_triangle_mask = shape.find_simplex(positions) >= 0
    covariance = [[(x_diameter/2) ** 2, 0], [0, (y_diameter/2) ** 2]]

    total_obstructive = 0
    for player_position in player_positions:
        
        if not square.find_simplex(player_position) >= 0:
            continue

        mean = player_position
        player_distribution = multivariate_normal(mean, covariance)

        full_player = player_distribution.pdf(positions)
        part_in_triangle = full_player.copy()
        part_in_triangle[~in_triangle_mask] = 0  
        ratio = part_in_triangle.sum().sum() / full_player.sum().sum()
        if pd.isnull(ratio):
            ratio = 0.0
        total_obstructive += ratio

    return total_obstructive

