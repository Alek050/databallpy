import unittest

import numpy as np
import pandas as pd

from databallpy.load_data.metadata import Metadata
from databallpy.load_data.tracking_data._get_matchtime import (
    _get_matchtime,
    _to_matchtime,
)


class TestGetMatchtime(unittest.TestCase):
    def setUp(self):
        self.metadata = Metadata(
            match_id=np.nan,
            pitch_dimensions=None,
            match_start_datetime=np.nan,
            periods_frames=pd.DataFrame(
                {
                    "period": [1, 2, 3, 4, 5],
                    "start_frame": [1509993, 1602134, 1684000, 1709800, 1737000],
                    "end_frame": [1577667, 1675726, 1709500, 1735000, 1738000],
                    "start_time": [np.nan, np.nan, np.nan, np.nan, np.nan],
                    "end_time": [np.nan, np.nan, np.nan, np.nan, np.nan],
                }
            ),
            frame_rate=25,
            home_team_id=np.nan,
            home_team_name=None,
            home_formation=None,
            home_score=np.nan,
            home_players=np.nan,
            away_team_id=np.nan,
            away_team_name=None,
            away_formation=None,
            away_score=np.nan,
            away_players=np.nan,
        )

    def test_get_matchtime(self):
        input = []
        input.extend([1522023] * 25)
        input.extend([1577093] * 25)
        input.extend([1577593] * 20)
        input.extend([1654511] * 25)
        input.extend([1675700] * 2)
        input.extend([1703500] * 25)
        input.extend([1706800] * 18)
        input.extend([1729300] * 25)
        input.extend([1732600] * 1)
        input.extend([1737500] * 25)
        input = pd.Series(input)

        expected_result = []
        expected_result.extend(["08:01"] * 25)
        expected_result.extend(["44:44"] * 25)
        expected_result.extend(["45:00+0:04"] * 20)
        expected_result.extend(["79:55"] * 25)
        expected_result.extend(["90:00+4:02"] * 2)
        expected_result.extend(["103:00"] * 25)
        expected_result.extend(["105:00+0:12"] * 18)
        expected_result.extend(["118:00"] * 25)
        expected_result.extend(["120:00+0:12"])
        expected_result.extend(["Penalty Shootout"] * 25)
        expected_result = pd.Series(expected_result)
        expected_result = expected_result.rename("matchtime")

        pd.testing.assert_series_equal(
            expected_result, _get_matchtime(input, self.metadata)
        )

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
