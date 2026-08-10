import pandas as pd


def _downcast_tracking_data(td: pd.DataFrame) -> pd.DataFrame:
    """Function to downcast the float columns of the tracking data to float32.

    Tracking systems report positions to the centimetre at best, which is far
    above the precision of float32. Downcasting halves the memory footprint of
    the tracking data.

    Args:
        td (pd.DataFrame): tracking data of the game

    Returns:
        pd.DataFrame: tracking data with all float64 columns as float32
    """
    float_columns = td.select_dtypes(include="float64").columns
    td[float_columns] = td[float_columns].astype("float32")
    return td
