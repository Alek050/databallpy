import numpy as np
import pandas as pd

from databallpy.load_data.metadata import Metadata


def _to_matchtime(secs: int, max_m: int, start_m: int) -> str:
    """Transforms the number of seconds into matchtime format

    Args:
        s (int): number of seconds since period started
        max_m (int): max number of minutes the period can last
        start_m (int): start of the period in minutes

    Returns:
        str: the time in matchtime format
    """
    seconds = str(secs % 60)
    if len(seconds) == 1:
        seconds = "0" + str(seconds)

    minutes = str(secs // 60 + start_m)
    if len(minutes) == 1:
        minutes = "0" + str(minutes)

    if int(minutes) < max_m:
        time_string = minutes + ":" + seconds
    else:
        max_time = str(max_m) + ":00"
        minutes_extra = str(int(minutes) - max_m)
        time_string = max_time + "+" + minutes_extra + ":" + seconds

    return time_string


def _get_matchtime(timestamp_column: pd.Series, metadata: Metadata) -> pd.Series:
    """Gives a series with time in the matchtime format based
    on the original timestamps and framerate

    Args:
        timestamp_column (pd.Series): containing the timestamps
        from tracking data dataframe
        metadata (Metadata): metadata including framerate and
        information on start and end of periods

    Returns:
        pd.Series: contains the time of the match in matchtime format
    """
    frame_rate = metadata.frame_rate
    periods_frames = metadata.periods_frames

    bins = list(periods_frames["start_frame"].values.flatten())
    bins = [b for b in bins if b != 0]
    periods = np.digitize(timestamp_column, bins, right=True)
    periods[0] = periods[1]

    period_start_dict = dict(
        zip(periods_frames["period"], periods_frames["start_frame"])
    )

    max_seconds_period = (
        periods_frames["end_frame"] - periods_frames["start_frame"]
    ) // frame_rate
    rel_timestamp = np.array(
        [x - period_start_dict[p] for x, p in zip(timestamp_column.values, periods)]
    )
    seconds = rel_timestamp // frame_rate
    df = pd.DataFrame(
        {
            "seconds": seconds,
            "period": periods,
        }
    )

    start_m_dict = {1: 0, 2: 45, 3: 90, 4: 105}

    max_m_dict = {1: 45, 2: 90, 3: 105, 4: 120}

    matchtime_list = []
    for p in [1, 2, 3, 4]:
        for seconds in df[df["period"] == p]["seconds"].unique():
            if seconds <= (max_seconds_period[p - 1]):
                matchtime_list.extend(
                    [_to_matchtime(int(seconds), max_m_dict[p], start_m_dict[p])]
                    * frame_rate
                )
            else:
                matchtime_list.extend([f"Break ({p})"] * frame_rate)

        matchtime_list = matchtime_list[: len(df[df["period"] <= p])]

    for _ in df[df["period"] == 5]["seconds"].unique():
        matchtime_list.extend(["Penalty Shootout"] * frame_rate)

    matchtime_list = matchtime_list[: len(df)]
    df["matchtime"] = matchtime_list

    return df["matchtime"]
