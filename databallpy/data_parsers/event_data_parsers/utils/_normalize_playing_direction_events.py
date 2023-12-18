import pandas as pd


def _normalize_playing_direction_events(
    event_data: pd.DataFrame, home_team_id: int, away_team_id: int
) -> pd.DataFrame:
    """Function to normalize the playing direction based on databallpy shots so
    that the home team is always represented as playing from left to right and the
    away team from right to left.

    Args:
        event_data (pd.DataFrame): the event data which to normalize
        home_team_id (int): the home team id
        away_team_id (int): the away team id

    Returns:
        pd.DataFrame: normalized event data.
    """

    to_changes_cols = [xy for xy in event_data.columns if "_x" in xy or "_y" in xy]
    for period in event_data["period_id"].unique():
        temp_ed = event_data[event_data["period_id"] == period]

        # home team
        home = temp_ed.loc[
            (temp_ed["team_id"] == home_team_id)
            & (temp_ed["databallpy_event"] == "shot")
        ]
        if home["start_x"].mean() < 0:  # home players shoot on the left goal
            event_data.loc[
                temp_ed[temp_ed["team_id"] == home_team_id].index, to_changes_cols
            ] *= -1

        # away team
        away = temp_ed.loc[
            (temp_ed["team_id"] == away_team_id)
            & (temp_ed["databallpy_event"] == "shot")
        ]
        if away["start_x"].mean() > 0:  # away players shoot on the right goal
            event_data.loc[
                temp_ed[temp_ed["team_id"] == away_team_id].index, to_changes_cols
            ] *= -1

    return event_data
