import numpy as np
import pandas as pd

from databallpy.load_data.metadata import Metadata
from databallpy.utils.utils import MISSING_INT


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


def _get_matchtime(
    timestamp_column: pd.Series, period_column: pd.Series, metadata: Metadata
) -> list:
    """Gives a list with time in the matchtime format based
    on the original timestamps and framerate

    Args:
        timestamp_column (pd.Series): containing the timestamps from tracking data
        dataframe
        period_column (pd.Series): containing the period for every frame
        metadata (Metadata): metadata including framerate and
        information on start and end of periods

    Returns:
        list: for every frame the match time.
    """
    frame_rate = metadata.frame_rate
    periods_frames = metadata.periods_frames

    period_start_dict = dict(
        zip(periods_frames["period"], periods_frames["start_frame"])
    )

    n_frames_period = dict(
        zip(
            periods_frames["period"],
            periods_frames["end_frame"] - periods_frames["start_frame"],
        )
    )

    rel_timestamp = np.array(
        [
            x - period_start_dict[p] if p > 0 else MISSING_INT * frame_rate
            for x, p in zip(timestamp_column.values, period_column.values)
        ]
    )
    seconds = rel_timestamp // frame_rate
    df = pd.DataFrame(
        {
            "seconds": seconds,
            "period": period_column.values,
        }
    )
    start_m_dict = {1: 0, 2: 45, 3: 90, 4: 105}
    max_m_dict = {1: 45, 2: 90, 3: 105, 4: 120}

    matchtime_list = []
    for p in [1, 2, 3, 4]:
        frame_end_current_p = periods_frames.loc[
            periods_frames["period"] == p, "end_frame"
        ].iloc[0]
        frame_start_next_p = periods_frames.loc[
            periods_frames["period"] == p + 1, "start_frame"
        ].iloc[0]
        if frame_start_next_p > 0 and frame_end_current_p > 0:
            n_frames_break = frame_start_next_p - frame_end_current_p - 1
        else:
            n_frames_break = 0
        matchtime_list_period = []
        for seconds in df[df["period"] == p]["seconds"].unique():
            matchtime_list_period.extend(
                [_to_matchtime(int(seconds), max_m_dict[p], start_m_dict[p])]
                * frame_rate
            )
        matchtime_list_period = matchtime_list_period[: n_frames_period[p] + 1]
        matchtime_list_period.extend(["Break"] * n_frames_break)
        matchtime_list.extend(matchtime_list_period)

    for _ in df[df["period"] == 5]["seconds"].unique():
        matchtime_list.extend(["Penalty Shootout"] * frame_rate)

    matchtime_list = matchtime_list[: len(df)]

    return matchtime_list
