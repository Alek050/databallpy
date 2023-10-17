import unittest

import pandas as pd
from bs4 import BeautifulSoup

from databallpy.load_data.tracking_data._normalize_playing_direction_tracking import (
    _normalize_playing_direction_tracking,
)
from databallpy.load_data.tracking_data.inmotio import (
    _get_metadata,
    _get_player_data,
    _get_td_channels,
    _get_tracking_data,
    load_inmotio_tracking_data,
)
from tests.expected_outcomes import MD_INMOTIO, TD_INMOTIO


class TestInmotio(unittest.TestCase):
    def setUp(self):
        self.tracking_data_loc = r"tests/test_data/inmotio_td_test.txt"
        self.metadata_loc = r"tests/test_data/inmotio_metadata_test.xml"
        self.expected_channels = ["home_1", "home_2", "away_1", "away_2"]

    def test_get_metadata(self):
        metadata = _get_metadata(self.metadata_loc)
        assert metadata == MD_INMOTIO

    def test_get_td_channels(self):
        channels = _get_td_channels(self.metadata_loc, MD_INMOTIO)
        assert channels == self.expected_channels

    def test_get_tracking_data(self):
        tracking_data = _get_tracking_data(
            self.tracking_data_loc, self.expected_channels, [100.0, 50.0]
        )
        expected_td = TD_INMOTIO.drop(["matchtime_td", "period_id"], axis=1)
        pd.testing.assert_frame_equal(tracking_data, expected_td)

    def test_load_tracking_data(self):
        tracking_data, metadata = load_inmotio_tracking_data(
            self.tracking_data_loc, self.metadata_loc, verbose=False
        )
        expected_tracking_data = TD_INMOTIO.iloc[1:, :].reset_index(drop=True)
        expected_tracking_data = _normalize_playing_direction_tracking(
            expected_tracking_data, MD_INMOTIO.periods_frames
        )
        assert metadata == MD_INMOTIO
        pd.testing.assert_frame_equal(tracking_data, expected_tracking_data, atol=1e-7)

    def test_get_player_data(self):
        file = open(self.metadata_loc, "r", encoding="UTF-8")
        lines = file.read()
        input = BeautifulSoup(lines, "xml").find_all("Player", {"teamId": "T-0001"})
        file.close()
        df_players = _get_player_data(input)
        pd.testing.assert_frame_equal(df_players, MD_INMOTIO.home_players)

    def test_not_string_error(self):
        with self.assertRaises(TypeError):
            load_inmotio_tracking_data(
                tracking_data_loc=1, metadata_loc=self.metadata_loc
            )
