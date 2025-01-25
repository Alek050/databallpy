import unittest
from unittest.mock import Mock, patch

import pandas as pd

from databallpy.data_parsers.tracking_data_parsers.metrica_tracking_data_parser import (
    _get_tracking_data,
    load_metrica_open_tracking_data,
    load_metrica_tracking_data,
)
from tests.expected_outcomes import MD_METRICA_TD, TD_CHANNELS_METRICA, TD_METRICA
from tests.mocks import MD_METRICA_RAW, TD_METRICA_RAW


class TestMetricaTrackingDataParser(unittest.TestCase):
    def setUp(self):
        self.md_loc = "tests/test_data/metrica_metadata_test.xml"
        self.td_loc = "tests/test_data/metrica_tracking_data_test.txt"

    def test_get_tracking_data(self):
        expected = TD_METRICA.copy()
        expected.drop(["gametime_td", "period_id", "datetime"], axis=1, inplace=True)
        res = _get_tracking_data(
            self.td_loc, TD_CHANNELS_METRICA, [100.0, 50.0], verbose=False
        )
        pd.testing.assert_frame_equal(res, expected)

    def test_load_metrica_tracking_data(self):
        res_td, res_md = load_metrica_tracking_data(
            self.td_loc, self.md_loc, verbose=False
        )

        pd.testing.assert_frame_equal(res_td, TD_METRICA)
        pd.testing.assert_frame_equal(
            res_md.periods_frames, MD_METRICA_TD.periods_frames
        )
        assert res_md == MD_METRICA_TD

        with self.assertRaises(TypeError):
            load_metrica_tracking_data(22, self.md_loc, verbose=False)
        with self.assertRaises(FileNotFoundError):
            load_metrica_tracking_data(
                "some_wrong_string.txt", self.md_loc, verbose=False
            )
        with self.assertRaises(FileNotFoundError):
            load_metrica_tracking_data(self.td_loc, self.md_loc + ".xml", verbose=False)

    @patch(
        "requests.get",
        side_effect=[Mock(text=TD_METRICA_RAW), Mock(text=MD_METRICA_RAW)],
    )
    def test_load_metrica_open_tracking_data(self, _):
        td, md = load_metrica_open_tracking_data()
        assert md == MD_METRICA_TD
        pd.testing.assert_frame_equal(td, TD_METRICA)
