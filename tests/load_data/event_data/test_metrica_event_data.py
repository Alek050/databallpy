import unittest
from unittest.mock import Mock, patch

import pandas as pd

from databallpy.load_data.event_data.metrica_event_data import (
    _get_event_data,
    load_metrica_event_data,
    load_metrica_open_event_data,
)
from tests.expected_outcomes import ED_METRICA, MD_METRICA
from tests.mocks import ED_METRICA_RAW, MD_METRICA_RAW


class TestMetricaEventData(unittest.TestCase):
    def setUp(self):
        self.md_loc = "tests/test_data/metrica_metadata_test.xml"
        self.ed_loc = "tests/test_data/metrica_event_data_test.json"

    def test_get_event_data(self):
        expected_event_data = ED_METRICA.copy()
        for col in ["end_x", "start_x"]:
            expected_event_data[col] = (expected_event_data[col] + 50) / 100.0
        for col in ["end_y", "start_y"]:
            expected_event_data[col] = (expected_event_data[col] + 25) / 50.0

        expected_event_data.drop(["datetime"], axis=1, inplace=True)
        ed = _get_event_data(self.ed_loc)
        pd.testing.assert_frame_equal(ed, expected_event_data)

    def test_load_metrica_event_data(self):
        ed, md = load_metrica_event_data(self.ed_loc, self.md_loc)
        assert md == MD_METRICA
        pd.testing.assert_frame_equal(ed, ED_METRICA)

        with self.assertRaises(TypeError):
            ed, md = load_metrica_event_data(22, self.md_loc)

    @patch(
        "requests.get",
        side_effect=[Mock(text=ED_METRICA_RAW), Mock(text=MD_METRICA_RAW)],
    )
    def test_load_open_metrica_event_data(self, _):
        ed, md = load_metrica_open_event_data()
        assert md == MD_METRICA
        pd.testing.assert_frame_equal(ed, ED_METRICA)
