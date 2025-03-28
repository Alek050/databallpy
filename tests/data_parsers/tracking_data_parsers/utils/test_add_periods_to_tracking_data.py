import unittest

import pandas as pd

from databallpy.data_parsers.tracking_data_parsers.utils import (
    _add_periods_to_tracking_data,
)
from tests.expected_outcomes import MD_METRICA_TD, TD_METRICA


class TestAddPeriodsToTrackingData(unittest.TestCase):
    def test_add_periods_to_tracking_data(self):
        input = TD_METRICA.drop(["period_id"], axis=1).copy()
        res = input
        res["period_id"] = _add_periods_to_tracking_data(
            input["frame"], MD_METRICA_TD.periods_frames
        )

        res = res.reindex(
            columns=[
                "frame",
                "ball_x",
                "ball_y",
                "ball_z",
                "ball_status",
                "team_possession",
                "home_11_x",
                "home_11_y",
                "home_1_x",
                "home_1_y",
                "away_34_x",
                "away_34_y",
                "away_35_x",
                "away_35_y",
                "datetime",
                "period_id",
                "gametime_td",
            ]
        )
        pd.testing.assert_frame_equal(res, TD_METRICA)
