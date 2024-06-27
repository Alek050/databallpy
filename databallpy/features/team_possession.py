import pandas as pd

from databallpy.utils.constants import MISSING_INT
from databallpy.utils.logging import create_logger

LOGGER = create_logger(__name__)


def add_team_possession(
    tracking_data: pd.DataFrame,
    event_data: pd.DataFrame,
    home_team_id: int,
    inplace: bool = False,
) -> None | pd.DataFrame:
    """Function to add a column 'ball_possession' to the tracking data, indicating
    which team has possession of the ball at each frame, either 'home' or 'away'.

    Raises:
        ValueError: If the tracking and event data are not synchronised.
        ValueError: If the home_team_id is not in the event data.


    Args:
        tracking_data (pd.DataFrame): Tracking data for a match
        event_data (pd.DataFrame): Event data for a match
        home_team_id (int): The ID of the home team.
        inplace (bool, optional): Whether to modify the DataFrame in place.
            Defaults to False.

    Returns:
        None | pd.DataFrame: The tracking data with the 'ball_possession' column added.
    """
    try:
        if "event_id" not in tracking_data.columns:
            raise ValueError(
                "Tracking and event data are not synchronised, please synchronise the"
                " data first"
            )
        if home_team_id not in event_data.team_id.unique():
            raise ValueError(
                "The home team ID is not in the event data, please check"
                " the home team ID"
            )

        if not inplace:
            tracking_data = tracking_data.copy()

        on_ball_events = ["pass", "dribble", "shot"]
        current_team_id = event_data.loc[
            ~pd.isnull(event_data["databallpy_event"]), "team_id"
        ].iloc[0]
        start_idx = 0
        tracking_data["ball_possession"] = None
        for event_id in [x for x in tracking_data.event_id if x != MISSING_INT]:
            event = event_data[event_data.event_id == event_id].iloc[0]

            if (
                event["databallpy_event"] in on_ball_events
                and event.team_id != current_team_id
                and event.outcome == 1
            ):
                end_idx = tracking_data[tracking_data.event_id == event_id].index[0]
                team = "home" if current_team_id == home_team_id else "away"
                tracking_data.loc[start_idx:end_idx, "ball_possession"] = team

                current_team_id = event.team_id
                start_idx = end_idx

        last_team = "home" if current_team_id == home_team_id else "away"
        tracking_data.loc[start_idx:, "ball_possession"] = last_team

        if not inplace:
            return tracking_data

    except Exception as e:
        LOGGER.exception(e)
        raise e
