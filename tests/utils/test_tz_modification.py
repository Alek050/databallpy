import unittest

import numpy as np
import pandas as pd

from databallpy import DataBallPyError
from databallpy.utils.tz_modification import localize_datetime, utc_to_local_datetime


class TestTzModification(unittest.TestCase):
    def test_utc_to_local_datetime(self):
        dates = pd.Series(
            pd.date_range("2023-05-01 20:00:00", periods=3, freq="d", tz="UTC")
        )
        expected_dates = pd.Series(
            pd.date_range(
                "2023-05-01 22:00:00", periods=3, freq="d", tz="Europe/Amsterdam"
            )
        )
        result = utc_to_local_datetime(dates, "Netherlands")
        pd.testing.assert_series_equal(result, expected_dates)

        # unkown characeristic
        with self.assertRaises(DataBallPyError):
            utc_to_local_datetime(dates, "Unknown_key")

        # should not changes, all are NaT
        dates = pd.Series(np.array(["NaT", "NaT", "NaT"], dtype="datetime64[ns]"))
        result = utc_to_local_datetime(dates, "Netherlands")
        pd.testing.assert_series_equal(result, dates)

    def test_localize_date(self):
        dates = pd.Series(pd.date_range("2023-05-01 20:00:00", periods=3, freq="d"))
        expected_dates = pd.Series(
            pd.date_range(
                "2023-05-01 20:00:00", periods=3, freq="d", tz="Europe/Amsterdam"
            )
        )
        result = localize_datetime(dates, "Netherlands")
        pd.testing.assert_series_equal(result, expected_dates)

        # unkown characeristic
        with self.assertRaises(DataBallPyError):
            localize_datetime(dates, "Unknown_key")

        # should not changes, all are NaT
        dates = pd.Series(np.array(["NaT", "NaT", "NaT"], dtype="datetime64[ns]"))
        result = localize_datetime(dates, "Netherlands")
        pd.testing.assert_series_equal(result, dates)
