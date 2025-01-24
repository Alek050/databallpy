import unittest

import pandas as pd

from databallpy.data_parsers.sportec_metadata_parser import (
    SPORTEC_BASE_URL,
    SPORTEC_METADATA_ID_MAP,
    SPORTEC_PRIVATE_LINK,
    _get_sportec_metadata,
    _get_sportec_open_data_url,
)
from databallpy.utils.constants import MISSING_INT
from tests.expected_outcomes import SPORTEC_METADATA_ED, SPORTEC_METADATA_TD


class TestMetricaMetadataParser(unittest.TestCase):
    def setUp(self):
        self.md_loc = "tests/test_data/sportec_metadata.xml"

        self.expected_metadata_td = SPORTEC_METADATA_TD.copy()
        self.expected_metadata_td.periods_frames["start_frame"] = MISSING_INT
        self.expected_metadata_td.periods_frames["end_frame"] = MISSING_INT
        self.expected_metadata_td.periods_frames["start_datetime_td"] = pd.to_datetime(
            "NaT"
        )
        self.expected_metadata_td.periods_frames["end_datetime_td"] = pd.to_datetime(
            "NaT"
        )
        self.expected_metadata_td.frame_rate = MISSING_INT
        self.expected_metadata_td.periods_changed_playing_direction = None

        self.expected_metadata_ed = SPORTEC_METADATA_ED.copy()
        self.expected_metadata_ed.periods_frames["start_datetime_ed"] = pd.to_datetime(
            "NaT"
        )
        self.expected_metadata_ed.periods_frames["end_datetime_ed"] = pd.to_datetime(
            "NaT"
        )
        self.expected_metadata_ed.periods_changed_playing_direction = None

    def test_get_metadata(self):
        self.assertEqual(_get_sportec_metadata(self.md_loc), self.expected_metadata_td)
        self.assertEqual(
            _get_sportec_metadata(self.md_loc, only_event_data=True),
            self.expected_metadata_ed,
        )

    def test_get_sportec_open_data_url(self):
        with self.assertRaises(ValueError):
            _get_sportec_open_data_url(game_id="unknown", data_type="metadata")
        with self.assertRaises(ValueError):
            _get_sportec_open_data_url(game_id="J03WMX", data_type="wrong")

        self.assertEqual(
            _get_sportec_open_data_url(game_id="J03WMX", data_type="metadata"),
            SPORTEC_BASE_URL
            + "/"
            + SPORTEC_METADATA_ID_MAP["J03WMX"]
            + "?private_link="
            + SPORTEC_PRIVATE_LINK,
        )
