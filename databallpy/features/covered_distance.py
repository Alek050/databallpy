"""
Author: Marit Sloots
Created on: 27-5-2024
Description: Function to calculate covered distance in certain velocity and acceleration intervals
"""


#%% import libraries
import warnings

import pandas as pd
import numpy as np

from databallpy.utils.logging import create_logger
#from databallpy.features.differentiate import add_velocity
#from databallpy.features.differentiate import add_acceleration

from databallpy.features.differentiate import _differentiate
#from databallpy.features.filters import _filter_data
from scipy.signal import savgol_filter
#from databallpy.utils.constants import MISSING_INT

LOGGER = create_logger(__name__)



#%% Load data
filepath = 'C:/Users/marit/Documents/Sport_Sciences/AcA_Databallpy/data.pkl'
data = pd.read_pickle(filepath)
df = data[0]

#%% function needed for this feature that are not in main yet
MISSING_INT = -999


def _filter_data(
    array: np.ndarray,
    filter_type: str = "savitzky_golay",
    window_length: int = 7,
    polyorder: int = 2,
) -> np.ndarray:
    """Function to filter data

    Args:
        array (np.ndarray): _description_
        filter_type (str, optional): type of filter to use. Defaults to
            "savitzky_golay". Options: {"moving_average", "savitzky_golay"}
        window_length (int, optional): Window length of the filter. Defaults to 7.
        polyorder (int, optional): polyorder to use when the savitzky_golay filter
            is selected. Defaults to 2.

    Returns:
        np.ndarray: filtered data


    """
    if not isinstance(array, np.ndarray):
        raise TypeError("array should be of type np.ndarray")

    if filter_type not in ["moving_average", "savitzky_golay"]:
        raise ValueError(
            "filter_type should be one of: 'moving_average', 'savitzky_golay'"
            f", got: {filter_type}"
        )

    if not isinstance(window_length, int):
        raise TypeError(
            f"window_length should be of type int, not {type(window_length)}"
        )

    if not isinstance(polyorder, int):
        raise TypeError(f"polyorder should be of type int not {type(polyorder)}")

    if not len(array) > window_length:
        raise ValueError("length of data should be greater than the window length")

    if filter_type == "savitzky_golay":
        return savgol_filter(
            array, axis=0, window_length=window_length, polyorder=polyorder
        )

    elif filter_type == "moving_average":
        return np.convolve(array, np.ones(window_length) / window_length, mode="same")


def add_acceleration(
    tracking_data: pd.DataFrame,
    column_ids: str | list[str],
    frame_rate: float,
    filter_type: str = None,
    window_length: int = 25,
    polyorder: int = 2,
    max_acceleration: float = np.inf,
    inplace: bool = False,
) -> pd.DataFrame:
    """Function that adds acceleration columns based on the position columns

    Args:
        tracking_data (pd.DataFrame): tracking data
        column_ids (str | list[str]): columns for which acceleration should be
            calculated
        frame_rate (float): framerate of the tracking data
        filter_type (str, optional): filter type to use. Defaults to None.
            Options are `moving_average` and `savitzky_golay`.
        window_length (int, optional): window size for the filter. Defaults to 25.
        polyorder (int, optional): polynomial order for the filter. Defaults to 2.
        max_acceleration (float, optional): maximum value for the acceleration.
            Defaults to np.inf.
        inplace (bool, optional): whether to modify the DataFrame in place.
            Defaults to False.

    Returns:
        pd.DataFrame: tracking data with the added acceleration columns

    Raises:
        ValueError: if filter_type is not one of `moving_average`, `savitzky_golay`,
        or None.
        ValueError: if velocity was not found in the DataFrame for the input_columns.

    Note:
        The function will delete the columns in input_columns with the acceleration if
        they already exist.
    """
    try:
        if isinstance(column_ids, str):
            column_ids = [column_ids]

        if filter_type not in ["moving_average", "savitzky_golay", None]:
            raise ValueError(
                "filter_type should be one of: 'moving_average', "
                f"'savitzky_golay', None, got: {filter_type}"
            )
        for column_id in column_ids:
            if (
                column_id + "_vx" not in tracking_data.columns
                or column_id + "_vy" not in tracking_data.columns
            ):
                raise ValueError(
                    f"Velocity was not found for {column_id} in the DataFrame. "
                    " Please calculate velocity first using get_velocity() function."
                )

        res_df = _differentiate(
            tracking_data,
            new_name="acceleration",
            metric="v",
            frame_rate=frame_rate,
            filter_type=filter_type,
            window=window_length,
            poly_order=polyorder,
            column_ids=column_ids,
            max_val=max_acceleration,
            inplace=inplace,
        )

        return res_df
    except Exception as e:
        LOGGER.exception(f"Found unexpected exception in get_acceleration(): \n{e}")
        raise e


def add_velocity(
    tracking_data: pd.DataFrame,
    column_ids: str | list[str],
    frame_rate: float,
    filter_type: str = None,
    window_length: int = 7,
    polyorder: int = 2,
    max_velocity: float = np.inf,
    inplace: bool = False,
) -> pd.DataFrame:
    """Function that adds velocity columns based on the position columns

    Args:
        tracking_data (pd.DataFrame): tracking data
        column_ids (str | list[str]): columns for which velocity should be calculated
        frame_rate (float): framerate of the tracking data
        filter_type (str, optional): filter type to use. Defaults to None.
            Options are `moving_average` and `savitzky_golay`.
        window_length (int, optional): window size for the filter. Defaults to 7.
        polyorder (int, optional): polynomial order for the filter. Defaults to 2.
        max_velocity (float, optional): maximum value for the velocity.
            Defaults to np.inf.
        inplace (bool, optional): whether to modify the DataFrame in place. Defaults

    Returns:
        pd.DataFrame: tracking data with the added velocity columns if inplace is False
            else None.

    Raises:
        ValueError: if filter_type is not one of `moving_average`, `savitzky_golay`,
        or None.

    Note:
        The function will delete the columns in input_columns with the velocity if
        they already exist.
    """
    try:
        if isinstance(column_ids, str):
            column_ids = [column_ids]

        if filter_type not in ["moving_average", "savitzky_golay", None]:
            raise ValueError(
                "filter_type should be one of: 'moving_average', "
                f"'savitzky_golay', None, got: {filter_type}"
            )

        res_df = _differentiate(
            tracking_data,
            new_name="velocity",
            metric="",
            frame_rate=frame_rate,
            filter_type=filter_type,
            window=window_length,
            poly_order=polyorder,
            column_ids=column_ids,
            max_val=max_velocity,
            inplace=inplace,
        )

        return res_df
    except Exception as e:
        LOGGER.exception(f"Found unexpected exception in get_velocity(): \n{e}")
        raise e


def _differentiate(
    df: pd.DataFrame,
    *,
    new_name: str,
    metric: str = "",
    frame_rate: int = 25,
    filter_type: str = "savitzky_golay",
    window: int = 7,
    max_val: float = np.nan,
    poly_order: int = 2,
    column_ids: list[str] | None = None,
    inplace: bool = False,
) -> pd.DataFrame | None:
    """
    Function that adds the differentiated values to the DataFrame.

    Args:
        df (pandas DataFrame): Position data in the x and y directions of players
            and ball.
        metric (str): the metric to differentiate the value on. Note that
            f"{player}_{metric}x" and f"{player}_{metric}y" should exist.
        new_name (str): the name of the magnitude. The first letter will be used
            for the x and y directions. For example, f"{player}_vx" and
            f"{player}_velocity" if new_name = "velocity".
        frame_rate (int): the sample frequency of the data.
        filter_type (str): the type of filter to use. Options are "moving average",
            "savitzky_golay", or None.
        window (int): the window size of the filter
        max_val (float): The maximum value of the differentiated value. For
            instance, player speeds > 12 m/s are very unlikely.
        poly_order (int): the polynomial order for the Savitzky-Golay filter.
        column_ids (list[str] | None): the columns to differentiate. If None, all
            columns with the metric in the name will be used. Defaults to None.
        inplace (bool): whether to modify the DataFrame in place. Defaults to False.

    Returns:
        pd.DataFrame | None: the DataFrame with the added columns if inplace is False,
        otherwise None.
    """

    if not inplace:
        df = df.copy()

    try:
        to_skip = len(metric) + 2
        if column_ids is None:
            column_ids = [x[:-to_skip] for x in df.columns if f"_{metric}x" in x]

        dt = 1.0 / frame_rate

        cols_to_drop = np.array(
            [
                [c + f"_{new_name}", c + f"_{new_name[0]}x", c + f"_{new_name[0]}y"]
                for c in column_ids
            ]
        ).ravel()
        df.drop(cols_to_drop, axis=1, errors="ignore", inplace=True)

        res_dict = {}
        for column_id in column_ids:
            gradient_x = np.gradient(df[column_id + f"_{metric}x"].values, dt)
            gradient_y = np.gradient(df[column_id + f"_{metric}y"].values, dt)
            raw_differentiated = np.linalg.norm([gradient_x, gradient_y], axis=0)

            # Scale gradients if magnitude exceeds max_val
            if not pd.isnull(max_val):
                exceed_max = raw_differentiated > max_val
                scale_factor = max_val / raw_differentiated[exceed_max]
                gradient_x[exceed_max] *= scale_factor
                gradient_y[exceed_max] *= scale_factor

            # smoothing the signal
            if filter_type is not None:
                gradient_x = _filter_data(
                    gradient_x,
                    filter_type=filter_type,
                    window_length=window,
                    polyorder=poly_order,
                )
                gradient_y = _filter_data(
                    gradient_y,
                    filter_type=filter_type,
                    window_length=window,
                    polyorder=poly_order,
                )

            res_dict[column_id + f"_{new_name[0]}x"] = np.array(gradient_x)
            res_dict[column_id + f"_{new_name[0]}y"] = np.array(gradient_y)
            res_dict[column_id + f"_{new_name}"] = np.linalg.norm(
                [gradient_x, gradient_y], axis=0
            )

        new_columns_df = pd.DataFrame(res_dict)

        if inplace:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
                df[new_columns_df.columns] = new_columns_df
            return None
        else:
            return pd.concat([df, new_columns_df], axis=1)

    except Exception as e:
        LOGGER.exception(f"Found unexpected exception in _differentiate: \n{e}")
        raise e


#%% 3-7-2024
def _parse_intervals(intervals):
    if not all(isinstance(element, (int, float)) for element in intervals):
        raise TypeError("All elements in the tuple must be integers or floats")
    
    if len(intervals) % 2 != 0:
        raise ValueError("Intervals must contain an even number of elements.")
    
    return [(min(intervals[i], intervals[i+1]), max(intervals[i], intervals[i+1])) for i in range(0, len(intervals), 2)]

def get_covered_distance(
    tracking_data: pd.DataFrame,
    player_ids: list[str],
    framerate: int,
    velocity_intervals: tuple[float, ...] | None = None,
    acceleration_intervals: tuple[float, ...] | None = None
    ) -> dict:
    """Function that calculates the distance covered in specified velocity and acceleration intervals

    Args:
        tracking_data:      tracking data
        player_ids:         list containing players. Example = ['home_8', 'away_3', 'home_16']
        framerate:          sample frequency in Hertz
        velocity_intervals: tuple that contains the velocity interval(s). For example [0 3] & [4 8]: (0,3,8,4)
        acc_intervals:      tuple that contains the acceleration interval(s). For example [-3 0] & [3 5]: (-3,0,3,5)      

    Returns:
        dict: a dictionary with for every player the total distance covered and optionally the 
        distance covered within the given velocity and/or acceleration threshold(s)
    
    Notes: 
        The function requires the velocity for every player calculated with the add_velocity function. 
        The acceleration for every player depends on the presence of acceleration intervals in the input
    """

    try:
        # Check input types
        # tracking_data
        if not isinstance(tracking_data, pd.DataFrame):
            raise TypeError(f"tracking data must be a pandas DataFrame, not a {type(tracking_data).__name__}")
        
        # player_ids
        if not isinstance(player_ids, list):
            raise TypeError(f"player_ids must be a list, not a {type(player_ids).__name__}")

        if not all(isinstance(player, str) for player in player_ids):
            raise TypeError("All elements in player_ids must be strings")
        
        # framerate
        if not isinstance(framerate, int):
            raise TypeError(f"framerate must be a int, not a {type(framerate).__name__}")

        # check velocity
        for player_id in player_ids:
            if (
                player_id + "_vx" not in tracking_data.columns
                or player_id + "_vy" not in tracking_data.columns
                or player_id + "_velocity" not in tracking_data.columns
            ):
                raise ValueError(
                    f"Velocity was not found for {player_id} in the DataFrame. "
                    " Please calculate velocity first using add_velocity() function."
                )

        # initialize dictionary covered distance
        players = {player_id: {'total_distance': 0} for player_id in player_ids}

        # Calculate total distance
        tracking_data_velocity = pd.concat([tracking_data[player_id + '_velocity'] for player_id in player_ids], axis=1)
        total_distance = tracking_data_velocity.apply(lambda player: np.sum(player / framerate))
        
        for i, player_id in enumerate(player_ids):
            players[player_id]['total_distance'] = total_distance.iloc[i]

        # Calculate velocity distance
        if velocity_intervals is not None:
            velocity_intervals = _parse_intervals(velocity_intervals)
            for player_id in player_ids:
                # initialize dictionary for velocity intervals
                players[player_id]['total_distance_velocity'] = []
                for min_vel, max_vel in velocity_intervals: 
                    mask = (tracking_data_velocity[player_id + '_velocity'] >= min_vel) & (tracking_data_velocity[player_id + '_velocity'] <= max_vel)
                    filtered_velocities = tracking_data_velocity[player_id + '_velocity'][mask]
                    distance_covered = np.sum(filtered_velocities / framerate)
                    players[player_id]['total_distance_velocity'].append(((min_vel, max_vel), distance_covered))

        # Calculate acceleration distance
        if acceleration_intervals is not None:
            # check acceleration
            for player_id in player_ids:
                if (
                player_id + "_ax" not in tracking_data.columns
                or player_id + "_ay" not in tracking_data.columns
                or player_id + "_acceleration" not in tracking_data.columns
            ):
                    raise ValueError(
                        f"Acceleration was not found for {player_id} in the DataFrame. "
                        " Please calculate acceleration first using add_acceleration() function."
                    )
                else: tracking_data_acceleration = pd.concat([tracking_data[player_id + '_acceleration'] for player_id in player_ids], axis=1)      
            acceleration_intervals = _parse_intervals(acceleration_intervals)
            for player_id in player_ids:
                # initialize dictionary for velocity intervals
                players[player_id]['total_distance_acceleration'] = []
                for min_vel, max_vel in acceleration_intervals: 
                    mask = (tracking_data_acceleration[player_id + '_acceleration'] >= min_vel) & (tracking_data_acceleration[player_id + '_acceleration'] <= max_vel)
                    filtered_acceleration = tracking_data_velocity[player_id + '_velocity'][mask]
                    distance_covered = np.sum(filtered_acceleration / framerate)
                    players[player_id]['total_distance_acceleration'].append(((min_vel, max_vel), distance_covered))

    except Exception as e:
        LOGGER.exception(f"Found unexpected exception in get_covered_distance(): \n{e}")
        raise e
    
    return players

#%%
input = pd.DataFrame(
    {
        "home_1_x": [10, 20, -30, 40, np.nan, 60],
        "home_1_y": [5, 12, -20, 30, np.nan, 60],
        "home_1_velocity": [12.2, 23.6, 13.5, np.nan, 18.0, np.nan],
        "home_1_acceleration": [35.8, 1, np.nan, 3, np.nan, np.nan],

        "away_2_x": [0, 2, 6, 4, 8, 10],
        "away_2_y": [1, 2, 3, 4, 5, 6],
        "away_2_velocity": [2.2, 3.2, 1.4, 1.4, 3.2, 2.2],
        "away_2_acceleration": [1, 0.5, 1, 1, 0.5, 1]
    }
)
framerate = 1
vel_intervals = (-1, 13, 30, 17)
acc_intervals = (35, 10, 0, 0.75)

test2 = add_velocity(input,['home_1','away_2'],1)

test3 = add_acceleration(test2,['home_1','away_2'],1)


dist_vel_add = get_covered_distance(test3, ["home_1","away_2"], 1, velocity_intervals=vel_intervals, acceleration_intervals=acc_intervals)