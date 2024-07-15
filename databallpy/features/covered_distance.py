import warnings

import pandas as pd
import numpy as np

from databallpy.utils.logging import create_logger

LOGGER = create_logger(__name__)

def _parse_intervals(intervals):
    if all(isinstance(element, (int, float)) for element in intervals):
        pairs = [(min(intervals[i], intervals[i+1]), max(intervals[i], intervals[i+1]))
                 for i in range(0, len(intervals) - 1)]   
    elif all(isinstance(element, (tuple, list)) for element in intervals):
        pairs = [(min(interval[0], interval[1]), max(interval[0], interval[1])) for interval in intervals]
    else:
        raise TypeError("Intervals must contain either all floats/integers or all tuples/lists.")
    
    return pairs

def _validate_inputs(tracking_data, player_ids, framerate):
    if not isinstance(tracking_data, pd.DataFrame):
        raise TypeError(f"tracking data must be a pandas DataFrame, not a {type(tracking_data).__name__}")

    if not isinstance(player_ids, list):
        raise TypeError(f"player_ids must be a list, not a {type(player_ids).__name__}")

    if not all(isinstance(player, str) for player in player_ids):
        raise TypeError("All elements in player_ids must be strings")

    if not isinstance(framerate, int):
        raise TypeError(f"framerate must be a int, not a {type(framerate).__name__}")
    return tracking_data, player_ids, framerate

def get_covered_distance(
    tracking_data: pd.DataFrame,
    player_ids: list[str],
    framerate: int,
    velocity_intervals: tuple[float, ...] | tuple[tuple[float, ...], ...] | None = None,
    acceleration_intervals: tuple[float, ...] | tuple[tuple[float, ...], ...] | None = None
    ) -> dict:
    """Calculates the distance covered based on the velocity magnitude at each frame.
        This function requires the `add_velocity` function to be called. Optionally,
        it can also calculate the distance covered within specified velocity and/or
        acceleration intervals.

    Args:
        tracking_data (pd.DataFrame): Tracking data with player positions.
        player_ids (list[str]): columns for which covered distance should be
            calculated
        frame_rate (int): framerate of the tracking data
        velocity_intervals: tuple that contains the velocity interval(s).
            Defaults to None
        acceleration_intervals: tuple that contains the acceleration interval(s).
            Defaults to None     

    Returns:
        dict: a dictionary with for every player the total distance covered and optionally the 
        distance covered within the given velocity and/or acceleration threshold(s)
    
    Notes: 
        The function requires the velocity for every player calculated with the add_velocity function. 
        The acceleration for every player depends on the presence of acceleration intervals in the input
    """

    try:
        tracking_data, player_ids, framerate = _validate_inputs(tracking_data, player_ids, framerate)

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

        players = {player_id: {'total_distance': 0} for player_id in player_ids}

        # total distance
        tracking_data_velocity = pd.concat([tracking_data[player_id + '_velocity'] for player_id in player_ids], axis=1)
        distance_per_frame = tracking_data_velocity.fillna(0) / framerate

        for i, player_id in enumerate(player_ids):
            players[player_id]['total_distance'] = distance_per_frame.iloc[:,i].sum()

        # total distance velocity
        if velocity_intervals is not None:
            velocity_intervals = _parse_intervals(velocity_intervals)
            for player_id in player_ids:
                players[player_id]['total_distance_velocity'] = []
                for min_vel, max_vel in velocity_intervals: 
                    mask = tracking_data_velocity[player_id + '_velocity'].between(min_vel,max_vel)
                    total_distance_velocity = distance_per_frame[player_id + '_velocity'][mask].sum()
                    players[player_id]['total_distance_velocity'].append(((min_vel, max_vel), total_distance_velocity))

        # total distance acceleration
        if acceleration_intervals is not None:
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
                players[player_id]['total_distance_acceleration'] = []
                for min_vel, max_vel in acceleration_intervals: 
                    mask = tracking_data_acceleration[player_id + '_acceleration'].between(min_vel,max_vel)
                    total_distance_acceleration = distance_per_frame[player_id + '_velocity'][mask].sum()
                    players[player_id]['total_distance_acceleration'].append(((min_vel, max_vel), total_distance_acceleration))

    except Exception as e:
        LOGGER.exception(f"Found unexpected exception in get_covered_distance(): \n{e}")
        raise e
    
    return players