import pandas as pd

from databallpy.utils.logging import logging_wrapper
from databallpy.utils.warnings import deprecated


@logging_wrapper(__file__)
@deprecated(
    "The get_covered_distance function is deprecated and will removed in version 0.8.0. Please use Game.TrackingData.get_covered_distance() instead."
)
def get_covered_distance(
    tracking_data: pd.DataFrame,
    column_ids: list[str],
    frame_rate: int,
    velocity_intervals: tuple[float, ...] | tuple[tuple[float, ...], ...] = (),
    acceleration_intervals: tuple[float, ...] | tuple[tuple[float, ...], ...] = (),
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> pd.DataFrame:
    """Calculates the distance covered based on the velocity magnitude at each frame.
        This function requires the `add_velocity` function to be called. Optionally,
        it can also calculate the distance covered within specified velocity and/or
        acceleration intervals.

    Args:
        tracking_data (pd.DataFrame): Tracking data with player positions.
        column_ids (list[str]): columns for which covered distance should be
            calculated
        frame_rate (int): framerate of the tracking data
        velocity_intervals (optional): tuple that contains the velocity interval(s).
            Defaults to ()
        acceleration_intervals (optional): tuple that contains the acceleration
            interval(s). Defaults to ()
        start_idx (int, optional): start index of the tracking data. Defaults to None.
        end_idx (int, optional): end index of the tracking data. Defaults to None

    Returns:
        pd.DataFrame: DataFrame with the covered distance for each player. The
        columns are the player_ids and the rows are the covered distance for each
        player. If velocity_intervals or acceleration_intervals are provided, the
        columns will be the player_ids and the intervals. The rows will be the
        covered distance for each player within the specified intervals.

    Notes:
        The function requires the velocity for every player calculated with the
        add_velocity function. The acceleration for every player depends on the
        presence of acceleration intervals in the input
    """

    _validate_inputs(
        tracking_data,
        column_ids,
        frame_rate,
        acceleration_intervals,
        start_idx,
        end_idx,
    )

    column_ids = sorted(column_ids)
    velocity_intervals = (
        _parse_intervals(velocity_intervals) if len(velocity_intervals) > 0 else []
    )
    acceleration_intervals = (
        _parse_intervals(acceleration_intervals)
        if len(acceleration_intervals) > 0
        else []
    )
    result_dict = (
        {"total_distance": []}
        | {
            f"total_distance_velocity_{interval[0]}_{interval[1]}": []
            for interval in velocity_intervals
        }
        | {
            f"total_distance_acceleration_{interval[0]}_{interval[1]}": []
            for interval in acceleration_intervals
        }
    )

    tracking_data_velocity = pd.concat(
        [tracking_data[player_id + "_velocity"] for player_id in column_ids], axis=1
    ).fillna(0)
    tracking_data_velocity.columns = tracking_data_velocity.columns.str.replace(
        "_velocity", ""
    )
    distance_per_frame = tracking_data_velocity / frame_rate

    start_idx = start_idx if start_idx is not None else tracking_data.index[0]
    end_idx = end_idx if end_idx is not None else tracking_data.index[-1]
    distance_per_frame = distance_per_frame.loc[start_idx:end_idx]
    tracking_data = tracking_data.loc[start_idx:end_idx]

    result_dict["total_distance"] = distance_per_frame.sum().values

    for intervals, interval_name in zip(
        [velocity_intervals, acceleration_intervals], ["velocity", "acceleration"]
    ):
        if len(intervals) > 0:
            result_dict = _add_covered_distance_interval(
                result_dict,
                interval_name,
                tracking_data,
                distance_per_frame,
                intervals,
                column_ids,
            )

    return pd.DataFrame(result_dict, index=column_ids)


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

    if not isinstance(framerate, int):
        raise TypeError(f"framerate must be a int, not a {type(framerate).__name__}")

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
                "start_idx and end_idx must be integers, " f"not {type(idx).__name__}"
            )

        if idx not in tracking_data.index:
            raise ValueError(f"Index {idx} is not in the tracking data")

        if start_idx is not None and end_idx is not None and start_idx >= end_idx:
            raise ValueError("start_idx must be smaller than end_idx")
