import unittest

import pandas as pd

from databallpy.load_data.tracking_data.tracab import (
    _get_metadata,
    _get_players_metadata,
    _get_tracking_data,
    load_tracab_tracking_data,
)
from tests.expected_outcomes import MD_TRACAB, TD_TRACAB


class TestTracab(unittest.TestCase):
    def setUp(self):
        self.tracking_data_loc = "tests/test_data/tracab_td_test.dat"
        self.metadata_loc = "tests/test_data/tracab_metadata_test.xml"

    def test_get_metadata(self):
        metadata = _get_metadata(self.metadata_loc)
        assert metadata == MD_TRACAB

    def test_get_tracking_data(self):
        tracking_data = _get_tracking_data(self.tracking_data_loc, verbose=False)
        expected_td = TD_TRACAB.drop(["matchtime_td", "period_id", "datetime"], axis=1)
        pd.testing.assert_frame_equal(tracking_data, expected_td)

    def test_load_tracking_data(self):
        tracking_data, metadata = load_tracab_tracking_data(
            self.tracking_data_loc, self.metadata_loc, verbose=False
        )
        assert metadata == MD_TRACAB
        pd.testing.assert_frame_equal(tracking_data, TD_TRACAB)

    def test_get_players_metadata(self):
        input_players_info = [
            {
                "PlayerId": "1234",
                "FirstName": "Bart",
                "LastName": "Bakker",
                "JerseyNo": "4",
                "StartFrameCount": "1212",
                "EndFrameCount": "2323",
            },
            {
                "PlayerId": "1235",
                "FirstName": "Bram",
                "LastName": "Slager",
                "JerseyNo": "3",
                "StartFrameCount": "1218",
                "EndFrameCount": "2327",
            },
        ]

        expected_df_players = pd.DataFrame(
            {
                "id": [1234, 1235],
                "full_name": ["Bart Bakker", "Bram Slager"],
                "shirt_num": [4, 3],
                "start_frame": [1212, 1218],
                "end_frame": [2323, 2327],
            }
        )
        df_players = _get_players_metadata(input_players_info)
        pd.testing.assert_frame_equal(df_players, expected_df_players)
