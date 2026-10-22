import pandas as pd

from databallpy.events.spadl import MIN_DRIBBLE_LENGTH


def _add_recovery_dribbles(
    event_data: pd.DataFrame, is_recovery: pd.Series
) -> pd.DataFrame:
    """Function to map ball recoveries to SPADL dribbles. A ball recovery becomes a
    dribble to the start of the next action in the same period, or no action when the
    ball did not move at least MIN_DRIBBLE_LENGTH or there is no next action in the
    same period.

    Args:
        event_data (pd.DataFrame): The event data with the SPADL columns, the
            period_id and location columns
        is_recovery (pd.Series): Whether the event is a ball recovery that is mapped
            to a dribble

    Returns:
        pd.DataFrame: The event data with the updated ball recoveries
    """
    spadl_columns = ["action_type", "action_result", "action_bodypart"]
    next_action = (
        event_data.loc[
            event_data["action_type"].notna() & ~is_recovery,
            ["period_id", "start_x", "start_y"],
        ]
        .reindex(event_data.index)
        .shift(-1)
        .bfill()
    )
    dx = event_data["start_x"] - next_action["start_x"]
    dy = event_data["start_y"] - next_action["start_y"]
    moved = (dx**2 + dy**2 >= MIN_DRIBBLE_LENGTH**2) & (
        event_data["period_id"] == next_action["period_id"]
    )
    event_data.loc[is_recovery & ~moved, spadl_columns] = None
    event_data.loc[is_recovery & moved, ["end_x", "end_y"]] = next_action.loc[
        is_recovery & moved, ["start_x", "start_y"]
    ].values
    return event_data
