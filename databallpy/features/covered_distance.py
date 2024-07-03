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



#%%
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

