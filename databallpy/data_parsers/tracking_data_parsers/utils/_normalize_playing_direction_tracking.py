import numpy as np
import pandas as pd


def _normalize_playing_direction_tracking(
    td: pd.DataFrame, periods: pd.DataFrame
) -> tuple[pd.DataFrame, list[int]]:
    """Function to represent the home team as playing from left to right for the
    full game, and the away team from right to left.

    Args:
        td (pd.DataFrame): tracking data of the game
        periods (pd.DataFrame): description of the start and end frames of the periods

    Returns:
        Tuple[pd.DataFrame, list]: tracking data of the game, normalized in x and y
        direction in such a way that the home team is represented of playing from left
        to right for the full game. The period ids of the periods in which the
        playing direction changed are returned as well.
    """

    home_x = [x for x in td.columns if "_x" in x and "home" in x]
    all_x_y = [x for x in td.columns if "_x" in x or "_y" in x]
    changed_periods = []
    for _, period_row in periods.iterrows():
        if len(td[td["frame"] == period_row["start_frame"]].index) > 0:
            idx_start = td[td["frame"] >= period_row["start_frame"]].index[0]
            idx_end = td[td["frame"] <= period_row["end_frame"]].index[-1]

            frame = td.loc[idx_start, home_x]

            if np.mean(frame) > 0:
                changed_periods.append(period_row["period_id"])
                td.loc[idx_start:idx_end, all_x_y] *= -1
    return td, changed_periods
