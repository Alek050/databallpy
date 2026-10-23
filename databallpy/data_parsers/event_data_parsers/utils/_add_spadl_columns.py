import pandas as pd

from databallpy.utils.constants import SPADL_COLUMNS


def _add_spadl_columns(event_data: pd.DataFrame) -> pd.DataFrame:
    """Function to add the SPADL vocabulary columns (action_type, action_result and
    action_bodypart) to the event data. Columns that are already present are left
    untouched, missing columns are added with null values. The SPADL columns are
    placed directly after the databallpy_event column, if present.

    Args:
        event_data (pd.DataFrame): the event data to add the SPADL columns to

    Returns:
        pd.DataFrame: event data with the SPADL columns.
    """
    for col in SPADL_COLUMNS:
        if col not in event_data.columns:
            event_data[col] = pd.Series(
                [None] * len(event_data), index=event_data.index, dtype=object
            )

    if "databallpy_event" in event_data.columns:
        other_columns = [col for col in event_data.columns if col not in SPADL_COLUMNS]
        loc = other_columns.index("databallpy_event") + 1
        event_data = event_data[
            other_columns[:loc] + SPADL_COLUMNS + other_columns[loc:]
        ]
    return event_data
