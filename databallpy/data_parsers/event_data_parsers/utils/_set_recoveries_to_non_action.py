import pandas as pd


def _set_recoveries_to_non_action(
    event_data: pd.DataFrame, is_recovery: pd.Series
) -> pd.DataFrame:
    """Function to map ball recoveries to SPADL non actions, as socceraction does.
    SPADL has no ball recovery action, the dribble (carry) after the recovery is added
    between the surrounding actions when converting to SPADL.

    Args:
        event_data (pd.DataFrame): The event data with the SPADL columns
        is_recovery (pd.Series): Whether the event is a ball recovery

    Returns:
        pd.DataFrame: The event data with the updated ball recoveries
    """
    event_data.loc[is_recovery, "action_type"] = "non_action"
    event_data.loc[is_recovery, ["action_result", "action_bodypart"]] = None
    return event_data
