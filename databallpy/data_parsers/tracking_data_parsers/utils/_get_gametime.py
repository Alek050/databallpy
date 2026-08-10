import pandas as pd

from databallpy.data_parsers.metadata import Metadata


def _to_gametime(secs: int, max_m: int, start_m: int) -> str:
    """Transforms the number of seconds into gametime format

    Args:
        s (int): number of seconds since period started
        max_m (int): max number of minutes the period can last
        start_m (int): start of the period in minutes

    Returns:
        str: the time in gametime format
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


def _get_gametime(
    frame_num_column: pd.Series, period_column: pd.Series, metadata: Metadata
) -> list:
    """Gives a list with time in the gametime format based
    on the original timestamps and framerate

    Args:
        frame_num_column (pd.Series): containing the frame number from tracking data
        dataframe
        period_column (pd.Series): containing the period for every frame
        metadata (Metadata): metadata including framerate and
        information on start and end of periods

    Returns:
        list: for every frame the game time.
    """
    frame_rate = metadata.frame_rate
    period_start_dict = dict(
        zip(
            metadata.periods_frames["period_id"],
            metadata.periods_frames["start_frame"],
        )
    )

    start_m_dict = {1: 0, 2: 45, 3: 90, 4: 105}
    max_m_dict = {1: 45, 2: 90, 3: 105, 4: 120}

    gametime_list = []
    game_started = False
    for frame, period_id in zip(frame_num_column.values, period_column.values):
        if period_id in start_m_dict:
            game_started = True
            seconds = int((frame - period_start_dict[period_id]) // frame_rate)
            gametime_list.append(
                _to_gametime(seconds, max_m_dict[period_id], start_m_dict[period_id])
            )
        elif period_id == 5:
            game_started = True
            gametime_list.append("Penalty Shootout")
        else:
            gametime_list.append("Break" if game_started else None)

    return gametime_list
