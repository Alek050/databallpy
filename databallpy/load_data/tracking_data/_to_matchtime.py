import pandas as pd
import numpy as np

from databallpy.load_data.metadata import Metadata
def _to_match_time(s:int, max_m:int, start_m:int) -> str:
    """Transforms the number of seconds into matchtime format

    Args:
        s (int): number of seconds since periods start
        max_m (int): max number of minutes the period can last
        start_m (int): start of the period in minutes

    Returns:
        str: the time in matchtime format
    """
    seconds = str(s%60)
    if len(seconds) == 1:
        seconds =  "0" + str(seconds)
    
    minutes = str(s//60 + start_m)
    if len(minutes) == 1:
        minutes =  "0" + str(minutes)

    if int(minutes) < max_m:
        time_string = minutes + ":" + seconds
    else:
        max_time = str(max_m) + ":00"
        minutes_extra = str(int(minutes)-max_m)
        time_string = max_time + "+" + minutes_extra + ":" + seconds
    
    return time_string


def _get_match_time(timestamp_column:pd.Series, metadata:Metadata) -> pd.Series:
    """Gives a series with time in the matchtime format based on the original timestamps and framerate

    Args:
        timestamp_column (pd.Series): containing the timestamps from tracking data dataframe
        metadata (Metadata): metadata including framerate and information on start and end of periods

    Returns:
        pd.Series: contains the time of the match in matchtime format
    """
    frame_rate = metadata.frame_rate
    periods_frames = metadata.periods_frames
    bins = list(periods_frames["start_frame"].values.flatten())
    bins = [b for b in bins if b != 0]
    periods = np.digitize(timestamp_column, bins, right=True)
    periods[0] = 1

    period_start_dict = {}
    for _, row in periods_frames.iterrows():
        period_start_dict[row["period"]] = row["start_frame"]

    rel_timestamp = np.array([x - period_start_dict[p] for x, p in zip(timestamp_column.values, periods)])
    seconds = rel_timestamp//frame_rate
    df = pd.DataFrame(
        {
            "rel_timestamp": rel_timestamp,
            "seconds": seconds,
            "period": periods,
            "matchtime": [""]*len(timestamp_column)
        }
    )

    matchtime_list = []
    for seconds in df[df["period"] == 1]["seconds"].unique():
        if seconds < (45*60):
            matchtime_list.extend([_to_match_time(seconds, 45, 0)]*frame_rate)
        else:
            matchtime_list.extend(["Break"]*frame_rate)
    
    matchtime_list = matchtime_list[:len(df[df["period"] == 1])]

    for seconds in df[df["period"] == 2]["seconds"].unique():
        if seconds < (90*60):
            matchtime_list.extend([_to_match_time(seconds, 90, 45)]*frame_rate)
        else:
            matchtime_list.extend(["Break"]*frame_rate)
    
    matchtime_list = matchtime_list[:len(df[df["period"] <= 2])]

    for seconds in df[df["period"] == 3]["seconds"].unique():
        if seconds < (105*60):
            matchtime_list.extend([_to_match_time(seconds, 105, 90)]*frame_rate)
        else:
            matchtime_list.extend(["Break"]*frame_rate)
    
    matchtime_list = matchtime_list[:len(df[df["period"] <= 3])]
    
    for seconds in df[df["period"] == 4]["seconds"].unique():
        if seconds < (120*60):
            matchtime_list.extend([_to_match_time(seconds, 120, 105)]*frame_rate)
        else:
            matchtime_list.extend(["Break"]*frame_rate)
    
    matchtime_list = matchtime_list[:len(df[df["period"] <= 4])]
    
    for _ in df[df["period"] == 5]["seconds"].unique():
        matchtime_list.extend(["Break"]*frame_rate)
    
    matchtime_list = matchtime_list[:len(df)]
    df["matchtime"] = matchtime_list

    return df["matchtime"]