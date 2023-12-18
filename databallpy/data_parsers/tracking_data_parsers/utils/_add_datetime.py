import datetime

import pandas as pd


def _add_datetime(
    frames: pd.Series, frame_rate: int, dt_start_match: datetime.datetime
) -> pd.Series:
    """Function to add datetime to tracking data. Checks if the frame is a timestamp by
    comparing it to the add dt_start_match. If it is a timestamp, it use only the date
    of the dt_start_match and adds the time of the timestamp. If it is not a timestamp,
    it adds the time of the frame to the dt_start_match.

    Args:
        frames (pd.Series): series of frames
        frame_rate (int): frame rate of the tracking data
        dt_start_match (datetime.datetime): datetime of the start of the match.

    Returns:
        pd.Series: series of datetime
    """

    frames_time = pd.to_datetime(frames.iloc[0] / frame_rate, unit="s").time()
    frames_minutes = frames_time.hour * 60 + frames_time.minute
    dt_start_match_minutes = dt_start_match.hour * 60 + dt_start_match.minute

    # if diff in minutes < 10, assume it is a timestamp
    if abs(dt_start_match_minutes - frames_minutes) <= 10:
        date = dt_start_match.date()
        date = datetime.datetime.combine(
            date, datetime.time(0, 0, 0), tzinfo=dt_start_match.tzinfo
        )
        return frames.apply(
            lambda timestamp: date
            + pd.to_timedelta(timestamp / frame_rate, unit="seconds")
        )

    else:
        return frames.apply(
            lambda frame: dt_start_match
            + pd.to_timedelta((frame - frames.iloc[0]) / frame_rate, unit="seconds")
        )
