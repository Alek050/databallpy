import numpy as np
import pandas as pd


def _add_covered_distance_interval(
    result_dict: dict,
    interval_type: str,
    tracking_data: pd.DataFrame,
    distance_per_frame: pd.DataFrame,
    intervals: list[tuple[float, float]],
    player_ids: list[str],
) -> dict:
    for player_id in player_ids:
        for min_val, max_val in intervals:
            mask = tracking_data[player_id + "_" + interval_type].between(
                min_val, max_val
            )
            total_distance = distance_per_frame[player_id][mask].sum()
            result_dict[f"total_distance_{interval_type}_{min_val}_{max_val}"].append(
                total_distance
            )
    return result_dict


def _parse_intervals(intervals):
    if all(isinstance(element, (int, float)) for element in intervals):
        pairs = [
            (min(intervals[i], intervals[i + 1]), max(intervals[i], intervals[i + 1]))
            for i in range(0, len(intervals) - 1)
        ]
    elif all(isinstance(element, (tuple, list)) for element in intervals):
        pairs = [
            (min(interval[0], interval[1]), max(interval[0], interval[1]))
            for interval in intervals
        ]
    else:
        raise TypeError(
            "Intervals must contain either all floats/integers"
            f"or all tuples/lists, not {intervals}"
        )
    return pairs


def _validate_inputs(
    tracking_data, player_ids, framerate, acceleration_intervals, start_idx, end_idx
):
    if not isinstance(tracking_data, pd.DataFrame):
        raise TypeError(
            "tracking data must be a pandas DataFrame, "
            f"not a {type(tracking_data).__name__}"
        )

    if not isinstance(player_ids, list):
        raise TypeError(f"player_ids must be a list, not a {type(player_ids).__name__}")

    if not all(isinstance(player, str) for player in player_ids):
        raise TypeError("All elements in player_ids must be strings")

    if not isinstance(framerate, (int, np.integer, float, np.floating)):
        raise TypeError(
            f"framerate must be a int or float, not a {type(framerate).__name__}"
        )

    for player_id in player_ids:
        if player_id + "_velocity" not in tracking_data.columns:
            raise ValueError(
                f"Velocity was not found for {player_id} in the DataFrame. "
                "Please calculate velocity first using add_velocity() function."
            )
        elif acceleration_intervals is not None and len(acceleration_intervals) > 0:
            if (
                player_id + "_ax" not in tracking_data.columns
                or player_id + "_ay" not in tracking_data.columns
                or player_id + "_acceleration" not in tracking_data.columns
            ):
                raise ValueError(
                    f"Acceleration was not found for {player_id} in the DataFrame. "
                    "Please calculate acceleration first using add_acceleration() "
                    "function."
                )

    for idx in [idx for idx in [start_idx, end_idx] if idx is not None]:
        if not isinstance(idx, int):
            raise TypeError(
                f"start_idx and end_idx must be integers, not {type(idx).__name__}"
            )

        if idx not in tracking_data.index:
            raise ValueError(f"Index {idx} is not in the tracking data")

        if start_idx is not None and end_idx is not None and start_idx >= end_idx:
            raise ValueError("start_idx must be smaller than end_idx")
