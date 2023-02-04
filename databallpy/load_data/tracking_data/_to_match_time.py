import numpy as np

from databallpy.load_data.metadata import Metadata


def _to_match_time(i: int, metadata: Metadata) -> str:
    """_summary_

    Args:
        tracking_data (pd.DataFrame): _description_
        metadata (Metadata): _description_

    Returns:
        pd.DataFrame: _description_
    """
    frame_rate = metadata.frame_rate
    periods_frames = metadata.periods_frames
    period = periods_frames[
        (i >= periods_frames["start_frame"]) & (i <= periods_frames["end_frame"])
    ]["period"]
    if len(period) == 0:
        return "Break"

    period = period.iloc[0]
    if period == 5:
        return "Penalty Shootout"
    rel_timestamp = i - periods_frames["start_frame"][period - 1]
    rel_time = np.timedelta64(rel_timestamp // frame_rate, "s")

    start_periods = {
        1: np.timedelta64(0, "s"),
        2: np.timedelta64(45 * 60, "s"),
        3: np.timedelta64(90 * 60, "s"),
        4: np.timedelta64(105 * 60, "s"),
    }

    time = start_periods[period] + rel_time
    time_seconds = int(time / np.timedelta64(1, "s"))

    max_time_dict = {
        1: 45,
        2: 90,
        3: 105,
        4: 120,
    }

    seconds = str(time_seconds % 60)
    if len(seconds) == 1:
        seconds = "0" + str(seconds)
    minutes = str(time_seconds // 60)

    if int(minutes) < max_time_dict[period]:
        time_string = minutes + ":" + seconds
    else:
        max_time = str(max_time_dict[period]) + ":00"
        minutes_extra = str(int(minutes) - max_time_dict[period])
        time_string = max_time + "+" + minutes_extra + ":" + seconds

    return time_string
