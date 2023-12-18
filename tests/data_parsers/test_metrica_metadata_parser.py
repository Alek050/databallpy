import unittest

import numpy as np
import pandas as pd

from databallpy.data_parsers.metrica_metadata_parser import (
    _get_metadata,
    _get_td_channels,
    _update_metadata,
)
from tests.expected_outcomes import MD_METRICA_TD, TD_CHANNELS_METRICA


class TestMetricaMetadataParser(unittest.TestCase):
    def setUp(self):
        self.md_loc = "tests/test_data/metrica_metadata_test.xml"
        self.expected_metadata = MD_METRICA_TD.copy()
        self.expected_metadata.periods_changed_playing_direction = None
        self.expected_td_channels = TD_CHANNELS_METRICA

    def test_get_metadata(self):
        expected_metadata = self.expected_metadata.copy()
        expected_metadata.home_formation = ""
        expected_metadata.away_formation = ""
        expected_metadata.home_players["starter"] = np.nan
        expected_metadata.away_players["starter"] = np.nan
        assert _get_metadata(self.md_loc) == expected_metadata

    def test_get_td_channels(self):
        input_metadata = self.expected_metadata.copy()
        input_metadata.home_formation = ""
        input_metadata.away_formation = ""
        input_metadata.home_players["starter"] = np.nan
        input_metadata.away_players["starter"] = np.nan
        res = _get_td_channels(self.md_loc, input_metadata)
        pd.testing.assert_frame_equal(res, self.expected_td_channels)

    def test_update_metadata(self):
        input_metadata = self.expected_metadata.copy()
        input_metadata.home_formation = ""
        input_metadata.away_formation = ""
        input_metadata.home_players["starter"] = np.nan
        input_metadata.away_players["starter"] = np.nan

        res = _update_metadata(self.expected_td_channels, input_metadata)

        assert res == self.expected_metadata
