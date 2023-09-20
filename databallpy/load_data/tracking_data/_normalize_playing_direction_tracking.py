import pandas as pd


def _normalize_playing_direction_tracking(
    td: pd.DataFrame, periods: pd.DataFrame
) -> pd.DataFrame:
    """Function to represent the home team as playing from left to right for the
    full match, and the away team from right to left.

    Args:
        td (pd.DataFrame): tracking data of the match
        periods (pd.DataFrame): description of the start and end frames of the periods

    Returns:
        pd.DataFrame: tracking data of the match, normalized in x and y direction in
        such a way that the home team is represented of playing from left to right
        for the full match.
    """

    home_x = [x for x in td.columns if "_x" in x and "home" in x]
    all_x_y = [x for x in td.columns if "_x" in x or "_y" in x]
    for _, period_row in periods.iterrows():
        if len(td[td["frame"] == period_row["start_frame"]].index) > 0:
            idx_start = td[td["frame"] == period_row["start_frame"]].index[0]
            idx_end = td[td["frame"] == period_row["end_frame"]].index[0]

            if td.loc[idx_start, home_x].mean() > 0:
                td.loc[idx_start:idx_end, all_x_y] *= -1

    return td
