import unittest

import pandas as pd

from databallpy.utils.utils import (
    MISSING_INT,
    _to_float,
    _to_int,
    get_next_possession_frame,
)


class TestUtils(unittest.TestCase):
    def test_to_float(self):
        assert pd.isnull(_to_float("s2"))
        assert pd.isnull(_to_float(None))
        assert 2.0 == _to_float("2")
        assert 3.3 == _to_float("3.3")

    def test_to_int(self):
        assert MISSING_INT == _to_int("s2.3")
        assert MISSING_INT == _to_int(None)
        assert 2 == _to_int("2")
        assert 3 == _to_int("3.3")

    def test_missing_int(self):
        assert MISSING_INT == -999

    def test_get_next_possession_frame(self):
        tracking_data = pd.DataFrame(
            {
                "ball_status": ["dead", "alive", "alive", "dead", "alive", "alive"],
                "player_possession": [None, None, "home_1", None, None, None],
            },
            index=[0, 1, 2, 3, 4, 5],
        )
        frame = tracking_data.iloc[0]
        res1 = get_next_possession_frame(tracking_data, frame, "home_2")

        pd.testing.assert_series_equal(res1, tracking_data.iloc[2])

        res2 = get_next_possession_frame(tracking_data, frame, "home_1")
        pd.testing.assert_series_equal(res2, tracking_data.iloc[3])

        frame = tracking_data.iloc[4]
        tracking_data["ball_status"] = [
            "dead",
            "alive",
            "alive",
            "dead",
            "dead",
            "dead",
        ]
        res3 = get_next_possession_frame(tracking_data, frame, "home_2")
        pd.testing.assert_series_equal(res3, tracking_data.iloc[5])

        tracking_data["ball_status"] = [
            "dead",
            "alive",
            "alive",
            "alive",
            "alive",
            "alive",
        ]
        res4 = get_next_possession_frame(tracking_data, frame, "home_2")
        pd.testing.assert_series_equal(res4, tracking_data.iloc[5])
