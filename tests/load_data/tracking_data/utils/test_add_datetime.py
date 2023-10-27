import unittest

import pandas as pd

from databallpy.load_data.tracking_data.utils._add_datetime import _add_datetime


class TestAddDatetime(unittest.TestCase):
    def setUp(self):
        self.frames_no_ts = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.frames_ts = pd.Series(
            [90000, 90001, 90002, 90003, 90004, 90005, 90006, 90007, 90008, 90009]
        )
        self.frame_rate = 10
        self.date_time = pd.to_datetime("2020-01-01 02:31:00").tz_localize(
            "Europe/Amsterdam"
        )

    def test_add_datetime_no_timestamp(self):
        res = _add_datetime(self.frames_no_ts, self.frame_rate, self.date_time)
        expected = pd.to_datetime("2020-01-01 02:31:00").tz_localize(
            "Europe/Amsterdam"
        ) + pd.to_timedelta(self.frames_no_ts / self.frame_rate, unit="seconds")
        pd.testing.assert_series_equal(res, expected)

    def test_add_datetime_timestamp(self):
        res = _add_datetime(self.frames_ts, self.frame_rate, self.date_time)
        expected = pd.to_datetime("2020-01-01 00:00:00").tz_localize(
            "Europe/Amsterdam"
        ) + pd.to_timedelta(self.frames_ts / self.frame_rate, unit="seconds")
        pd.testing.assert_series_equal(res, expected)
