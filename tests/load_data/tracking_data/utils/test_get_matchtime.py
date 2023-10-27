import unittest

import numpy as np
import pandas as pd

from databallpy.load_data.metadata import Metadata
from databallpy.load_data.tracking_data.utils._get_matchtime import (
    _get_matchtime,
    _to_matchtime,
)
from databallpy.utils.utils import MISSING_INT


class TestGetMatchtime(unittest.TestCase):
    def setUp(self):
        self.metadata = Metadata(
            match_id=123,
            pitch_dimensions=[100.0, 50.0],
            periods_frames=pd.DataFrame(
                {
                    "period_id": [1, 2, 3, 4, 5],
                    "start_frame": [1509993, 1509997, 1510001, 1510005, 1510009],
                    "end_frame": [1509995, 1509999, 1510003, 1510007, 1510011],
                    "start_time": [np.nan, np.nan, np.nan, np.nan, np.nan],
                    "end_time": [np.nan, np.nan, np.nan, np.nan, np.nan],
                }
            ),
            frame_rate=25,
            home_team_id=1,
            home_team_name="",
            home_formation="",
            home_score=np.nan,
            home_players=pd.DataFrame({"id": [], "full_name": [], "shirt_num": []}),
            away_team_id=2,
            away_team_name="",
            away_formation="",
            away_score=np.nan,
            away_players=pd.DataFrame({"id": [], "full_name": [], "shirt_num": []}),
            country="",
        )

    def test_get_matchtime(self):
        input = []
        input.extend([1509993, 1509994, 1509995])
        input.extend([1509996])
        input.extend([1509997, 1509998, 1509999])
        input.extend([1501000])
        input.extend([1510001, 1510002, 1510003])
        input.extend([1510004])
        input.extend([1510005, 1510006, 1510007])
        input.extend([1510008])
        input.extend([1510009, 1510010, 1510011])
        input = pd.Series(input)

        period_column = []
        period_column.extend([1] * 3)
        period_column.extend([MISSING_INT])
        period_column.extend([2] * 3)
        period_column.extend([MISSING_INT])
        period_column.extend([3] * 3)
        period_column.extend([MISSING_INT])
        period_column.extend([4] * 3)
        period_column.extend([MISSING_INT])
        period_column.extend([5] * 3)
        period_column = pd.Series(period_column)

        expected_result = []
        expected_result.extend(["00:00"] * 3)
        expected_result.extend(["Break"])
        expected_result.extend(["45:00"] * 3)
        expected_result.extend(["Break"])
        expected_result.extend(["90:00"] * 3)
        expected_result.extend(["Break"])
        expected_result.extend(["105:00"] * 3)
        expected_result.extend(["Break"])
        expected_result.extend(["Penalty Shootout"] * 3)
        expected_result = pd.Series(expected_result)
        expected_result = expected_result.rename("matchtime")

        result_list = _get_matchtime(input, period_column, self.metadata)
        result = pd.Series(result_list)
        result = result.rename("matchtime")
        pd.testing.assert_series_equal(expected_result, result)

    def test_to_matchtime(self):
        test_frames = [12030, 67100, 67600, 52377, 73566, 19500, 22800, 19500, 22800]
        seconds = [int(x) // self.metadata.frame_rate for x in test_frames]
        periods = [1, 1, 1, 2, 2, 3, 3, 4, 4]
        start_m = {1: 0, 2: 45, 3: 90, 4: 105}

        max_m = {1: 45, 2: 90, 3: 105, 4: 120}

        res = []

        for s, p in zip(seconds, periods):
            res.append(_to_matchtime(s, max_m[p], start_m[p]))

        expected_res = [
            "08:01",
            "44:44",
            "45:00+0:04",
            "79:55",
            "90:00+4:02",
            "103:00",
            "105:00+0:12",
            "118:00",
            "120:00+0:12",
        ]

        assert res == expected_res
