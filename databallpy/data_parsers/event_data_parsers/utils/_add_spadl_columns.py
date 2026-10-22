import pandas as pd

from databallpy.utils.constants import SPADL_COLUMNS


def _add_spadl_columns(event_data: pd.DataFrame) -> pd.DataFrame:
    """Function to add the SPADL vocabulary columns (action_type, action_result and
    action_bodypart) to the event data. Columns that are already present are left
    untouched, missing columns are added with null values.

    Args:
        event_data (pd.DataFrame): the event data to add the SPADL columns to

    Returns:
        pd.DataFrame: event data with the SPADL columns.
    """
    for col in SPADL_COLUMNS:
        if col not in event_data.columns:
            event_data[col] = pd.Series(None, index=event_data.index, dtype=object)
    return event_data
