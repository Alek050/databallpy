import pandas as pd

from databallpy.utils.logging import create_logger

LOGGER = create_logger(__name__)


def get_covered_distance(
    tracking_data: pd.DataFrame,
    player_ids: list[str],
    framerate: int,
    velocity_intervals: tuple[float, ...] | tuple[tuple[float, ...], ...] | None = None,
    acceleration_intervals: tuple[float, ...] | tuple[tuple[float, ...], ...] | None = None,
    start_idx: int | None = None,
    end_idx: int | None = None,
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
        velocity_intervals (optional): tuple that contains the velocity interval(s).
            Defaults to None
        acceleration_intervals (optional): tuple that contains the acceleration
            interval(s). Defaults to None
        start_idx (int, optional): start index of the tracking data. Defaults to None.
        end_idx (int, optional): end index of the tracking data. Defaults to None

    Returns:
        dict: a dictionary with for every player the total distance covered and
        optionally the distance covered within the given velocity and/or acceleration
        threshold(s)

    Notes:
        The function requires the velocity for every player calculated with the
        add_velocity function. The acceleration for every player depends on the
        presence of acceleration intervals in the input
    """

    try:
        _validate_inputs(
            tracking_data,
            player_ids,
            framerate,
            acceleration_intervals,
            start_idx,
            end_idx)

        start_idx = start_idx if start_idx is not None else tracking_data.index[0]
        end_idx = end_idx if end_idx is not None else tracking_data.index[-1]
        tracking_data = tracking_data.loc[start_idx:end_idx]

        tracking_data_velocity = pd.concat([tracking_data[player_id + '_velocity'] for player_id in player_ids], axis=1).fillna(0)
        tracking_data_velocity.columns = tracking_data_velocity.columns.str.replace('_velocity', '')
        distance_per_frame = tracking_data_velocity / framerate

        total_distances = distance_per_frame.sum()
        players = {player_id: {'total_distance': total_distances[player_id]} for player_id in player_ids}

        if velocity_intervals:
            velocity_intervals = _parse_intervals(velocity_intervals)
            players = _add_covered_distance_interval(players,
                                                     "velocity",
                                                     tracking_data,
                                                     distance_per_frame,
                                                     velocity_intervals,
                                                     player_ids)
        if acceleration_intervals:
            acceleration_intervals = _parse_intervals(acceleration_intervals)
            players = _add_covered_distance_interval(players,
                                                     "acceleration",
                                                     tracking_data,
                                                     distance_per_frame,
                                                     acceleration_intervals,
                                                     player_ids)

        return players
    except Exception as e:
        LOGGER.exception(f"Found unexpected exception in get_covered_distance(): \n{e}")
        raise e


def _add_covered_distance_interval(
    players: dict,
    interval_type: str,
    tracking_data: pd.DataFrame,
    distance_per_frame: pd.DataFrame,
    intervals: list[tuple[float, float]],
    player_ids: list[str]
    ):
    for player_id in player_ids:
        players[player_id]['total_distance_' + interval_type] = []
        for min_val, max_val in intervals:
            mask = tracking_data[player_id + '_' + interval_type].between(min_val, max_val)
            total_distance = distance_per_frame[player_id][mask].sum()
            players[player_id]['total_distance_' + interval_type].append(
                {f"{min_val}_{max_val}": total_distance})
    return players


def _parse_intervals(
        intervals
    ):
    if all(isinstance(element, (int, float)) for element in intervals):
        pairs = [(min(intervals[i], intervals[i + 1]), max(intervals[i], intervals[i + 1])) for i in range(0, len(intervals) - 1)]
    elif all(isinstance(element, (tuple, list)) for element in intervals):
        pairs = [(min(interval[0], interval[1]), max(interval[0], interval[1])) for interval in intervals]
    else:
        raise TypeError("Intervals must contain either all floats/integers \
            or all tuples/lists.")
    return pairs


def _validate_inputs(
        tracking_data, player_ids, framerate, acceleration_intervals, start_idx, end_idx
    ):
    if not isinstance(tracking_data, pd.DataFrame):
        raise TypeError(f"tracking data must be a pandas DataFrame, \
            not a {type(tracking_data).__name__}")

    if not isinstance(player_ids, list):
        raise TypeError(f"player_ids must be a list, not a {type(player_ids).__name__}")

    if not all(isinstance(player, str) for player in player_ids):
        raise TypeError("All elements in player_ids must be strings")

    if not isinstance(framerate, int):
        raise TypeError(f"framerate must be a int, not a {type(framerate).__name__}")

    for player_id in player_ids:
        if (player_id + "_velocity" not in tracking_data.columns):
            raise ValueError(
                f"Velocity was not found for {player_id} in the DataFrame. "
                "Please calculate velocity first using add_velocity() function."
            )
        elif acceleration_intervals is not None:
            if (
                player_id + "_ax" not in tracking_data.columns
                or player_id + "_ay" not in tracking_data.columns
                or player_id + "_acceleration" not in tracking_data.columns
            ):
                raise ValueError(
                    f"Acceleration was not found for {player_id} in the DataFrame. \
                    Please calculate acceleration first using add_acceleration() \
                    function."
                )

    for idx in [idx for idx in [start_idx, end_idx] if idx is not None]:
        if not isinstance(idx, int):
            raise TypeError(f"start_idx and end_idx must be integers, \
                            not {type(idx).__name__}")

        if idx not in tracking_data.index:
            raise ValueError(f"Index {idx} is not in the tracking data")
