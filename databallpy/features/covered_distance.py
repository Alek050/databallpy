"""
Author: Marit Sloots
Created on: 27-5-2024
Description: Function to calculate covered distance in certain velocity and acceleration intervals
"""


#%% import libraries
import pandas as pd
import numpy as np

from databallpy import get_match
from databallpy.utils.logging import create_logger
from databallpy.features.differentiate import get_velocity

# Add following or remove function if they are in main
from databallpy.features.differentiate import _differentiate

#from databallpy.features.differentiate import get_acceleration
#from databallpy.utils.constants import MISSING_INT

from math import sqrt

LOGGER = create_logger(__name__)

#%% Load data
filepath = 'C:/Users/marit/Documents/Sport_Sciences/AcA_Databallpy/data.pkl'
data = pd.read_pickle(filepath)


#%% function needed for this feature that are not in main yet
MISSING_INT = -999

def get_acceleration(
    df: pd.DataFrame,
    input_columns: list,
    framerate: float,
    filter_type: str = None,
    window: int = 25,
    poly_order: int = 2,
    max_val: float = 7.0,
) -> pd.DataFrame:
    """Function that adds acceleration columns based on the position columns

    Args:
        df (pd.DataFrame): tracking data
        input_columns (list): columns for which acceleration should be calculated
        framerate (float): framerate of the tracking data
        filter_type (str, optional): filter type to use. Defaults to None.
            Options are `moving_average` and `savitzky_golay`.
        window (int, optional): window size for the filter. Defaults to 25.
        poly_order (int, optional): polynomial order for the filter. Defaults to 2.
        max_val (float, optional): maximum value for the acceleration. Defaults to
            7.0.

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
        if filter_type not in ["moving_average", "savitzky_golay", None]:
            raise ValueError(
                "filter_type should be one of: 'moving_average', "
                "'savitzky_golay', None, got: {filter_type}"
            )
        for input_column in input_columns:
            if (
                input_column + "_vx" not in df.columns
                or input_column + "_vy" not in df.columns
            ):
                raise ValueError(
                    f"Velocity was not found for {input_column} in the DataFrame. "
                    " Please calculate velocity first using get_velocity() function."
                )

        res_df = _differentiate(
            df,
            new_name="acceleration",
            metric="v",
            frame_rate=framerate,
            filter_type=filter_type,
            window=window,
            poly_order=poly_order,
            column_ids=input_columns,
            max_val=max_val,
        )

        return res_df
    except Exception as e:
        LOGGER.exception(f"Found unexpected exception in get_acceleration(): \n{e}")
        raise e



#%% pre 2-7-2024
def get_covered_distance(
    tracking_data: pd.DataFrame,
    player_ids: list[str],
    framerate: int,
    vel_intervals: list[tuple[float, float]] | None = None,
    acc_intervals: list[tuple[float, float]] | None = None,
    start_frame: int = None,
    end_frame: int = None
    ) -> dict:
    """Function that calculates the distance covered in specified velocity and acceleration intervals

    Args:
        tracking_data:   tracking data
        player_ids:      list containing players. Example = ['home_8', 'away_3', 'home_16']
        framerate:       sample frequency in hertz
        vel_intervals:   list of tuples that contains the velocity interval(s). For example: [(0,3), (4,8)]
        acc_intervals:   list of tuples that contains the acceleration interval(s). For example: [(-3,0), (3,5)]
        start_frame:     startframe to calculate covered distances
        end_frame:       endframe to calculate covered distances      
    
    Raises:
        TypeError: ...

    Returns:
        dict: a dictionary with for every player the total distance covered and optionally the distance covered within the given velocity and/or acceleration thresholds
    """
    try:
        if start_frame is not None:
            tracking_data = tracking_data.iloc[start_frame:]
        
        if end_frame is not None:
            tracking_data = tracking_data.iloc[:end_frame]
        
        players = {
            player_id: {'total_distance': 0, 'total_distance_velocity': [], 'total_distance_acceleration': []} for player_id in player_ids
        }

        # Calculate total distance
        tracking_data = get_velocity(tracking_data, player_ids, framerate)
        tracking_data_velocity = pd.concat([tracking_data[player + '_velocity'] for player in player_ids], axis=1)
        total_distance = tracking_data_velocity.apply(lambda column: np.mean(column * len(tracking_data) / framerate))
        
        for i, player in enumerate(player_ids):
            players[player]['total_distance'] = total_distance.iloc[i]
        
        # Calculate distances for velocity intervals
        if vel_intervals is not None:
            for vel_interval in vel_intervals:
                min_vel, max_vel = vel_interval
                interval_distance = tracking_data_velocity.apply(lambda x: np.sum(x[(x >= min_vel) & (x <= max_vel)] / framerate))
                
                for i, player in enumerate(player_ids):
                    players[player]['total_distance_velocity'].append((vel_interval, interval_distance.iloc[i]))

        # Calculate distances for acceleration intervals
        if acc_intervals is not None:
            tracking_data = get_acceleration(tracking_data, player_ids, framerate)
            tracking_data_acceleration = pd.concat([tracking_data[player + '_acceleration'] for player in player_ids], axis=1)
            
            for acc_interval in acc_intervals:
                min_acc, max_acc = acc_interval
                interval_distance = tracking_data_acceleration.apply(lambda x: np.sum(x[(x >= min_acc) & (x <= max_acc)] / (framerate ** 2)))
                
                for i, player in enumerate(player_ids):
                    players[player]['total_distance_acceleration'].append((acc_interval, interval_distance.iloc[i]))

    except Exception as e:
        LOGGER.exception(f"Found unexpected exception in get_velocity(): \n{e}")
        raise e
    
    return players

#%% 3-7-2024
def _check_intervals(intervals):
    if not all(isinstance(element, int) for element in intervals):
        raise TypeError("All elements in the tuple must be integers")
    
    elif len(intervals) % 2 != 0:
        raise ValueError("Intervals must contain an even number of elements.")
    
    else:
        result = [(min(intervals[i], intervals[i+1]), max(intervals[i], intervals[i+1])) for i in range(0, len(intervals), 2)]
    
    return result



def get_covered_distance(
    tracking_data: pd.DataFrame,
    player_ids: list[str],
    framerate: int,
    velocity_intervals: tuple[float, ...] | None = None,
    acceleration_intervals: tuple[float, ...] | None = None,
    start_frame: int = None,
    end_frame: int = None
    ) -> dict:
    """Function that calculates the distance covered in specified velocity and acceleration intervals

    Args:
        tracking_data:      tracking data
        player_ids:         list containing players. Example = ['home_8', 'away_3', 'home_16']
        framerate:          sample frequency in hertz
        velocity_intervals: tuple that contains the velocity intervals. For example [0 3] & [4 8]: (0,3,8,4)
        acc_intervals:      tuple that contains the acceleration intervals. For example [-3 0] & [3 5]: (-3,0,3,5)
        start_frame:        start frame to calculate covered distances
        end_frame:          end frame to calculate covered distances      
    
    Raises:
        ...

    Returns:
        dict: a dictionary with for every player the total distance covered and optionally the 
        distance covered within the given velocity and/or acceleration thresholds
    """

    try:
        # Check input types
        # tracking_data
        if not isinstance(tracking_data, pd.DataFrame):
            raise TypeError(f"tracking data must be a pandas DataFrame, not a{type(tracking_data)}")
        
        # player_ids
        if not isinstance(player_ids, list):
            raise TypeError(f"player_ids must be a list, not a{type(player_ids).__name__}")

        if not all(isinstance(player, str) for player in player_ids):
            raise TypeError("All elements in player_ids must be strings")
        
        # framerate
        if not isinstance(framerate, int):
            raise TypeError(f"framerate must be a integer, not a {type(framerate).__name__}")

        # start_frame and end_frame
        if start_frame is not None and not isinstance(start_frame, int):
            raise TypeError(f"start_frame must be an integer, not a {type(start_frame).__name__}")

        if end_frame is not None and not isinstance(end_frame, int):
            raise TypeError(f"end_frame must be an integer, not a {type(end_frame).__name__}")

        if start_frame is not None and start_frame < 0:
            raise ValueError("start_frame must be greater than or equal to 0")

        if end_frame is not None and end_frame <= 0:
            raise ValueError("end_frame must be greater than 0")

        if start_frame is not None and start_frame > len(tracking_data):
            raise ValueError(f"start_frame must be less than or equal to {len(tracking_data)}")

        if end_frame is not None and end_frame > len(tracking_data):
            raise ValueError(f"end_frame must be less than or equal to {len(tracking_data)}")

        if start_frame is not None and end_frame is not None and end_frame <= start_frame:
            raise ValueError("end_frame must be greater than start_frame")

        # intervals
        if velocity_intervals is not None:
            velocity_intervals = _check_intervals(velocity_intervals)

        if acceleration_intervals is not None:
            acceleration_intervals = _check_intervals(acceleration_intervals)

        # Slice the tracking_data DataFrame
        if start_frame is not None:
            tracking_data = tracking_data.iloc[start_frame:]
        
        if end_frame is not None:
            tracking_data = tracking_data.iloc[:end_frame]
        
        players = {
            player_id: {'total_distance': 0, 'total_distance_velocity': [], 'total_distance_acceleration': []} for player_id in player_ids
        }

        # Calculate total distance
        tracking_data = get_velocity(tracking_data, player_ids, framerate)
        tracking_data_velocity = pd.concat([tracking_data[player + '_velocity'] for player in player_ids], axis=1)
        total_distance = tracking_data_velocity.apply(lambda column: np.mean(column * len(tracking_data) / framerate))
        
        for i, player in enumerate(player_ids):
            players[player]['total_distance'] = total_distance.iloc[i]
        
        # Calculate distances for velocity intervals
        if velocity_intervals is not None:
            for min_vel, max_vel in velocity_intervals:
                interval_distance = tracking_data_velocity.apply(lambda x: np.sum(x[(x >= min_vel) & (x <= max_vel)] / framerate))
                
                for i, player in enumerate(player_ids):
                    players[player]['total_distance_velocity'].append(((min_vel, max_vel), interval_distance.iloc[i]))

        # Calculate distances for acceleration intervals
        if acceleration_intervals is not None:
            tracking_data = get_acceleration(tracking_data, player_ids, framerate)
            tracking_data_acceleration = pd.concat([tracking_data[player + '_acceleration'] for player in player_ids], axis=1)
            
            for min_acc, max_acc in acceleration_intervals:
                interval_distance = tracking_data_acceleration.apply(lambda x: np.sum(x[(x >= min_acc) & (x <= max_acc)] / (framerate ** 2)))
                
                for i, player in enumerate(player_ids):
                    players[player]['total_distance_acceleration'].append(((min_acc, max_acc), interval_distance.iloc[i]))

    except Exception as e:
        LOGGER.exception(f"Found unexpected exception in get_covered_distance(): \n{e}")
        raise e
    
    return players

test = get_covered_distance(data[0],['away_17','home_10'],25,(2,6,10,16))
#%%
df = pd.DataFrame(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "away_2_x": [0, 2, 4, 6, 8, 10],
                "away_2_y": [1, 2, 3, 4, 5, 6],
            }
        )

vel_intervals = (0, 'b', 90, 50, 3, -1)
acc_intervals = (-2, 30, 0, 10, -2, -50)

test = get_covered_distance(df,['home_1','away_2'],1,vel_intervals = vel_intervals)


#%%
# Velocity intervals: [(min_vel1, max_vel1), (min_vel2, max_vel2)]
vel_intervals = [(0, 5), (5, 10)]

# Acceleration intervals: [(min_acc1, max_acc1), (min_acc2, max_acc2)]
acc_intervals = [(-3, 0), (0, 3)]

# Define the start and end frames
start_frame = 10
end_frame = 90

# Call the function
covered_distances = get_covered_distance(
    data[1],
    ['away_17','home_5'],
    25,
    vel_intervals=vel_intervals,
    acc_intervals=acc_intervals,
    start_frame=start_frame,
    end_frame=end_frame
)

# Print the results
print(covered_distances)

#%%
df = pd.DataFrame(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "away_2_x": [0, 2, 4, 6, 8, 10],
                "away_2_y": [1, 2, 3, 4, 5, 6],
            }
        )
expected = {
            'home_1': {'total_distance': 193.65, 'total_distance_velocity': [], 'total_distance_acceleration': []},
            'away_2': {'total_distance': 11.2, 'total_distance_velocity': [], 'total_distance_acceleration': []},
        }



def get_covered_distance(
    tracking_data: pd.DataFrame,
    player_ids: list[str],
    framerate: int,
    vel_intervals: list[tuple[float, float]] | None = None,
    acc_intervals: list[tuple[float, float]] | None = None,
    start_frame: int = None,
    end_frame: int = None
    ) -> dict:
    """Function that calculates the distance covered in specified velocity and acceleration intervals

    Args:
        tracking_data:   tracking data
        player_ids:      list containing players. Example = ['home_8', 'away_3', 'home_16']
        framerate:       sample frequency in hertz
        vel_intervals:   list of tuples that contains the velocity interval(s). For example: [(0,3), (4,8)]
        acc_intervals:   list of tuples that contains the acceleration interval(s). For example: [(-3,0), (3,5)]
        start_frame:     startframe to calculate covered distances
        end_frame:       endframe to calculate covered distances      
    
    Raises:
        TypeError: ...

    Returns:
        dict: a dictionary with for every player the total distance covered and optionally the distance covered within the given velocity and/or acceleration thresholds
    """

    if start_frame is not None:
        tracking_data = tracking_data.iloc[start_frame:]
    
    if end_frame is not None:
        tracking_data = tracking_data.iloc[:end_frame]
    
    players = {
        player_id: {'total_distance': 0, 'total_distance_velocity': [], 'total_distance_acceleration': []} for player_id in player_ids
    }

    # Calculate total distance
    tracking_data = get_velocity(tracking_data, player_ids, framerate)
    tracking_data_velocity = pd.concat([tracking_data[player + '_velocity'] for player in player_ids], axis=1)
    total_distance = tracking_data_velocity.apply(lambda column: np.mean(column * len(tracking_data) / framerate))
    
    for i, player in enumerate(player_ids):
        players[player]['total_distance'] = total_distance.iloc[i]
    
    
    return players


test = get_covered_distance(df,['home_1','away_2'],1,vel_intervals = [(0, 15), (50, 90)])


#%%
player_ids = ["home_1"]
tracking_data = get_velocity(df,player_ids,1)
tracking_data_velocity = pd.concat([tracking_data[player + '_velocity'] for player in player_ids], axis=1)
total_distance = tracking_data_velocity.apply(lambda column: np.mean(column * (len(tracking_data) - 1)/ 1))