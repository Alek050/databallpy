import pandas as pd


def _check_column_ids(df: pd.DataFrame, column_ids: list[str]):
    """Function to check if column ids is well formatted. Raises errors if not.

    1. Check if column_ids is a list
    2. Check if column_ids is not empty
    3. Check if all elements in column_ids are strings
    4. Check if all elements in column_ids are in the columns of the dataframe

    Args:
        df (pd.DataFrame): The tracking data dataframe
        column_ids (list[str]): The list of column ids to check
    """

    if not isinstance(column_ids, list):
        raise TypeError(f"column_ids should be of type list, not {type(column_ids)}")

    if len(column_ids) == 0:
        raise ValueError("column_ids should not be empty")

    if not all(isinstance(column_id, str) for column_id in column_ids):
        raise TypeError("All elements in column_ids should be of type str")

    not_in_df = [x for x in column_ids if x + "_x" not in df.columns]
    if len(not_in_df) > 0:
        raise ValueError(f"Columns {not_in_df} are not in the dataframe")
