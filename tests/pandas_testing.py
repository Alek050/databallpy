"""Wrappers around pandas.testing that compare the same values the same way on
pandas 2 and pandas 3. Since pandas 3.0, text columns default to the ``str`` dtype
instead of ``object`` and datetimes are no longer always stored in nanoseconds, so
an expected frame built in a test can differ in dtype only from the parsed frame.
On pandas 2 these wrappers behave exactly like pandas.testing."""

import pandas as pd

from databallpy.utils.utils import _align_dtypes_for_comparison


def assert_series_equal(left: pd.Series, right: pd.Series, **kwargs) -> None:
    left, right = _align_dtypes_for_comparison(left, right)
    pd.testing.assert_series_equal(left, right, **kwargs)


def assert_frame_equal(left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> None:
    left = left.copy()
    right = right.copy()
    for col in left.columns.intersection(right.columns):
        left[col], right[col] = _align_dtypes_for_comparison(left[col], right[col])
    pd.testing.assert_frame_equal(left, right, **kwargs)


def to_list(series: pd.Series) -> list:
    """Function to get the values of a series as a list with None for missing
    values, since pandas 3 stores missing values in text columns as NaN."""
    return [None if pd.isna(value) else value for value in series]
