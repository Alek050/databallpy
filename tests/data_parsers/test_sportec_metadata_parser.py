import unittest

from databallpy.data_parsers.sportec_metadata_parser import _get_sportec_metadata
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
