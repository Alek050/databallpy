import unittest

import pandas as pd

from databallpy.data_parsers.tracking_data_parsers.tracab_parser import (
    _get_metadata,
    _get_players_metadata_v1,
    _get_tracking_data_txt,
    load_tracab_tracking_data,
)
from tests.expected_outcomes import MD_TRACAB, TD_TRACAB


class TestTracabParser(unittest.TestCase):
    def setUp(self):
        self.tracking_data_dat_loc = "tests/test_data/tracab_td_test.dat"
        self.metadata_dat_loc = "tests/test_data/tracab_metadata_test.xml"
        self.metadata_sportec_loc = "tests/test_data/sportec_metadata.xml"
        self.wrong_format_loc = "tests/test_data/tracab_td_wrong_format.json"

    def test_load_tracab_tracking_data(self):
        tracking_data, metadata = load_tracab_tracking_data(
            self.tracking_data_dat_loc, self.metadata_dat_loc, verbose=False
        )
        assert metadata == MD_TRACAB
        pd.testing.assert_frame_equal(tracking_data, TD_TRACAB)

    def test_load_tracab_tracking_data_errors(self):
        with self.assertRaises(FileNotFoundError):
            load_tracab_tracking_data(
                self.tracking_data_dat_loc[:-3], self.metadata_dat_loc, verbose=False
            )
        with self.assertRaises(FileNotFoundError):
            load_tracab_tracking_data(
                self.tracking_data_dat_loc, self.metadata_dat_loc + ".sml", verbose=False
            )
        
        with self.assertRaises(ValueError):
            load_tracab_tracking_data(
                self.wrong_format_loc, self.metadata_dat_loc
            )

    def test_get_metadata(self):
        metadata = _get_metadata(self.metadata_dat_loc)
        expected_metadata = MD_TRACAB.copy()
        expected_metadata.periods_changed_playing_direction = None
        assert metadata == expected_metadata

    def test_get_tracking_data(self):
        tracking_data = _get_tracking_data_txt(self.tracking_data_dat_loc, verbose=False)
        expected_td = TD_TRACAB.drop(["matchtime_td", "period_id", "datetime"], axis=1)
        pd.testing.assert_frame_equal(tracking_data, expected_td)

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
        df_players = _get_players_metadata_v1(input_players_info)
        pd.testing.assert_frame_equal(df_players, expected_df_players)
