import unittest

from databallpy.data_parsers.sportec_metadata_parser import (
    SPORTEC_BASE_URL,
    SPORTEC_METADATA_ID_MAP,
    SPORTEC_PRIVATE_LINK,
    _get_sportec_metadata,
    _get_sportec_open_data_url,
)
from tests.expected_outcomes import SPORTEC_METADATA_ED, SPORTEC_METADATA_TD


class TestMetricaMetadataParser(unittest.TestCase):
    def setUp(self):
        self.md_loc = "tests/test_data/sportec_metadata.xml"
        self.expected_metadata_td = SPORTEC_METADATA_TD.copy()
        self.expected_metadata_ed = SPORTEC_METADATA_ED.copy()

    def test_get_metadata(self):
        assert _get_sportec_metadata(self.md_loc) == self.expected_metadata_td
        assert (
            _get_sportec_metadata(self.md_loc, only_event_data=True)
            == self.expected_metadata_ed
        )

    def test_get_sportec_open_data_url(self):
        with self.assertRaises(ValueError):
            _get_sportec_open_data_url(match_id="unknown", data_type="metadata")
        with self.assertRaises(ValueError):
            _get_sportec_open_data_url(match_id="J03WMX", data_type="wrong")

        self.assertEqual(
            _get_sportec_open_data_url(match_id="J03WMX", data_type="metadata"),
            SPORTEC_BASE_URL
            + "/"
            + SPORTEC_METADATA_ID_MAP["J03WMX"]
            + "?private_link="
            + SPORTEC_PRIVATE_LINK,
        )
