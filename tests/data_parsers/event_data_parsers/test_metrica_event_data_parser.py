import unittest
from unittest.mock import Mock, patch

import pandas as pd

from databallpy.data_parsers.event_data_parsers.metrica_event_data_parser import (
    _get_databallpy_events,
    _get_event_data,
    load_metrica_event_data,
    load_metrica_open_event_data,
)
from tests.expected_outcomes import (
    DRIBBLE_EVENTS_METRICA,
    ED_METRICA,
    MD_METRICA_ED,
    PASS_EVENTS_METRICA,
    SHOT_EVENTS_METRICA,
)
from tests.mocks import ED_METRICA_RAW, MD_METRICA_RAW


class TestMetricaEventDataParser(unittest.TestCase):
    def setUp(self):
        self.md_loc = "tests/test_data/metrica_metadata_test.xml"
        self.ed_loc = "tests/test_data/metrica_event_data_test.json"

    def test_load_metrica_event_data(self):
        ed, md, dbpe = load_metrica_event_data(self.ed_loc, self.md_loc)
        assert md == MD_METRICA_ED
        pd.testing.assert_frame_equal(ed, ED_METRICA)
        assert dbpe["shot_events"] == SHOT_EVENTS_METRICA
        assert dbpe["pass_events"] == PASS_EVENTS_METRICA
        assert dbpe["dribble_events"] == DRIBBLE_EVENTS_METRICA

        with self.assertRaises(TypeError):
            load_metrica_event_data(22, self.md_loc)

    def test_load_metrica_event_data_errors(self):
        with self.assertRaises(FileNotFoundError):
            load_metrica_event_data(self.ed_loc[3:], self.md_loc)
        with self.assertRaises(FileNotFoundError):
            load_metrica_event_data(self.ed_loc, self.md_loc[1:])

    def test_get_event_data(self):
        expected_event_data = ED_METRICA.copy()
        for col in ["end_x", "start_x"]:
            expected_event_data[col] = (expected_event_data[col] + 50) / 100.0
        for col in ["end_y", "start_y"]:
            expected_event_data[col] = (expected_event_data[col] + 25) / 50.0

        expected_event_data.drop(["datetime"], axis=1, inplace=True)
        ed = _get_event_data(self.ed_loc)
        pd.testing.assert_frame_equal(ed, expected_event_data)

    @patch(
        "requests.get",
        side_effect=[Mock(text=ED_METRICA_RAW), Mock(text=MD_METRICA_RAW)],
    )
    def test_load_open_metrica_event_data(self, _):
        ed, md, dbpe = load_metrica_open_event_data()
        assert md == MD_METRICA_ED
        pd.testing.assert_frame_equal(ed, ED_METRICA)
        assert dbpe["shot_events"] == SHOT_EVENTS_METRICA
        assert dbpe["pass_events"] == PASS_EVENTS_METRICA
        assert dbpe["dribble_events"] == DRIBBLE_EVENTS_METRICA

    def test_get_databallpy_events(self):
        res_dbpe = _get_databallpy_events(
            ED_METRICA, pitch_dimensions=(106, 68), home_team_id=1
        )
        shot_events = res_dbpe["shot_events"]
        expected_shot_events = {}
        for event_id, event in shot_events.items():
            expected_shot_events[event_id] = event.copy()
            expected_shot_events[event_id].pitch_size = (106, 68)

        pass_events = res_dbpe["pass_events"]
        expected_pass_events = {}
        for event_id, event in pass_events.items():
            expected_pass_events[event_id] = event.copy()
            expected_pass_events[event_id].pitch_size = (106, 68)
        dribble_events = res_dbpe["dribble_events"]
        expected_dribble_events = {}
        for event_id, event in dribble_events.items():
            expected_dribble_events[event_id] = event.copy()
            expected_dribble_events[event_id].pitch_size = (106, 68)

        assert res_dbpe["shot_events"] == expected_shot_events
        assert res_dbpe["pass_events"] == expected_pass_events
        assert res_dbpe["dribble_events"] == expected_dribble_events
