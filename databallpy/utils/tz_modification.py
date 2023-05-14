from datetime import timezone

import pandas as pd

from databallpy.errors import DataBallPyError

CHARACTERISTIC_TIMEZONE = {
    "Netherlands": "Europe/Amsterdam",
    "Keuken Kampioen Divisie": "Europe/Amsterdam",
    "Eredivisie": "Europe/Amsterdam",
}


def utc_to_local_datetime(dt_series: pd.Series, characteristic: str) -> pd.Series:
    """Function to convert UTC time to local time

    Args:
        dt_series (pd.Series): series with datetime object
        characteristic (str): country or competition to convert to local time

    Returns:
        pd.Series: series with converted time
    """
    if pd.isnull(dt_series).all():
        return dt_series

    if characteristic not in CHARACTERISTIC_TIMEZONE:
        raise DataBallPyError(
            f"Country or competition {characteristic} is not implemented. Please open\
 an issue with this error on our github page to get it added in our next version."
        )

    assert dt_series.dt.tz == timezone.utc, "dt_series should be in UTC timezone"

    converted = dt_series.copy()

    return converted.dt.tz_convert(CHARACTERISTIC_TIMEZONE[characteristic])


def localize_datetime(
    dt_series: pd.Series,
    characteristic: str,
) -> pd.Series:
    """Function to localize the timezone of a datetime series

    Args:
        dt_series (pd.Series): series with datetime objects
        characteristic (str): country or competition to localize to

    Returns:
        pd.Series: series with localized time
    """

    if pd.isnull(dt_series).all():
        return dt_series

    if characteristic not in CHARACTERISTIC_TIMEZONE:
        raise DataBallPyError(
            f"Country or competition {characteristic} is not implemented. Please open\
 an issue with this error on our github page to get it added in our next version."
        )

    assert dt_series.dt.tz is None, "dt_series should not have a timezone"

    localized = dt_series.copy()
    return localized.dt.tz_localize(CHARACTERISTIC_TIMEZONE[characteristic])
