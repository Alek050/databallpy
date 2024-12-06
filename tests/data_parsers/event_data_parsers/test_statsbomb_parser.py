import json
import unittest

import pandas as pd

from databallpy.data_parsers.event_data_parsers.statsbomb_parser import (
    _get_close_to_ball_event_info,
    _get_dribble_event,
    _get_pass_event,
    _get_player_info,
    _get_shot_event,
    _load_event_data,
    _load_metadata,
    load_statsbomb_event_data,
)
from databallpy.utils.utils import MISSING_INT
from tests.expected_outcomes import (
    DRIBBLE_EVENT_STATSBOMB,
    ED_STATSBOMB,
    MD_STATSBOMB,
    PASS_EVENT_STATSBOMB,
    SHOT_EVENT_STATSBOMB,
)


class TestStatsbombParser(unittest.TestCase):
    def setUp(self):
        self.event_loc = "tests/test_data/statsbomb_event_test.json"
        self.match_loc = "tests/test_data/statsbomb_match_test.json"
        self.lineup_loc = "tests/test_data/statsbomb_lineup_test.json"
        self.pitch_dimensions = (105.0, 68.0)

    def test_load_statsbomb_event_data_errors(self):
        with self.assertRaises(ValueError):
            load_statsbomb_event_data(
                events_loc=self.event_loc,
                match_loc=self.match_loc,
                lineup_loc="statsbomb_lineup_test.xml",
                pitch_dimensions=(120.0, 80.0),
            )
        with self.assertRaises(TypeError):
            load_statsbomb_event_data(
                events_loc=self.event_loc,
                match_loc=3,
                lineup_loc=self.lineup_loc,
                pitch_dimensions=(120.0, 80.0),
            )
        with self.assertRaises(FileNotFoundError):
            load_statsbomb_event_data(
                events_loc="file_does_not_exist.json",
                match_loc=self.match_loc,
                lineup_loc=self.lineup_loc,
                pitch_dimensions=(120.0, 80.0),
            )
        with self.assertRaises(ValueError):
            load_statsbomb_event_data(
                events_loc=self.event_loc,
                match_loc=self.match_loc,
                lineup_loc=self.lineup_loc,
                pitch_dimensions=(120.0),
            )

    def test_load_metadata(self):
        expected_outcome = MD_STATSBOMB.copy()
        expected_outcome.home_formation = ""
        expected_outcome.away_formation = ""
        result = _load_metadata(
            match_loc=self.match_loc,
            lineup_loc=self.lineup_loc,
            pitch_dimensions=(105.0, 68.0),
        )

        assert expected_outcome == result

    def test_get_player_info(self):
        expected_outcome = MD_STATSBOMB.home_players
        with open(self.lineup_loc, "r", encoding="utf-8") as f:
            input = json.load(f)[0]["lineup"]
        result = _get_player_info(input)
        pd.testing.assert_frame_equal(expected_outcome, result)

    def test_load_event_data(self):
        expected_outcome_event_data = ED_STATSBOMB.copy()

        expected_outcome_metadata = MD_STATSBOMB

        metadata_input = MD_STATSBOMB.copy()
        metadata_input.home_formation = ""
        metadata_input.away_formation = ""

        outcome_event_data, outcome_dbp_events, outcome_metadata = _load_event_data(
            events_loc=self.event_loc,
            metadata=metadata_input,
            pitch_dimensions=(105.0, 68.0),
        )

        assert "shot_events" in outcome_dbp_events.keys()
        for key, event in outcome_dbp_events["shot_events"].items():
            assert key in SHOT_EVENT_STATSBOMB.keys()
            assert event == SHOT_EVENT_STATSBOMB[key]

        assert "pass_events" in outcome_dbp_events.keys()
        for key, event in outcome_dbp_events["pass_events"].items():
            assert key in PASS_EVENT_STATSBOMB.keys()
            assert event == PASS_EVENT_STATSBOMB[key]

        assert "dribble_events" in outcome_dbp_events.keys()
        for key, event in outcome_dbp_events["dribble_events"].items():
            assert key in DRIBBLE_EVENT_STATSBOMB.keys()
            assert event == DRIBBLE_EVENT_STATSBOMB[key]

        pd.testing.assert_frame_equal(
            outcome_event_data, expected_outcome_event_data, rtol=1e-3
        )
        assert outcome_metadata == expected_outcome_metadata

    def test_load_statsbomb_event_data(self):
        expected_outcome_event_data = ED_STATSBOMB
        expected_outcome_metadata = MD_STATSBOMB

        outcome_event_data, outcome_metadata, outcome_dbp_events = (
            load_statsbomb_event_data(
                events_loc=self.event_loc,
                match_loc=self.match_loc,
                lineup_loc=self.lineup_loc,
                pitch_dimensions=(105.0, 68.0),
            )
        )

        assert "shot_events" in outcome_dbp_events.keys()
        for key, event in outcome_dbp_events["shot_events"].items():
            assert key in SHOT_EVENT_STATSBOMB.keys()
            assert event == SHOT_EVENT_STATSBOMB[key]

        assert "pass_events" in outcome_dbp_events.keys()
        for key, event in outcome_dbp_events["pass_events"].items():
            assert key in PASS_EVENT_STATSBOMB.keys()
            assert event == PASS_EVENT_STATSBOMB[key]

        assert "dribble_events" in outcome_dbp_events.keys()
        for key, event in outcome_dbp_events["dribble_events"].items():
            assert key in DRIBBLE_EVENT_STATSBOMB.keys()
            assert event == DRIBBLE_EVENT_STATSBOMB[key]

        pd.testing.assert_frame_equal(
            expected_outcome_event_data, outcome_event_data, rtol=1e-3
        )
        assert expected_outcome_metadata == outcome_metadata

    def test_get_close_to_ball_event_info(self):
        expected_outcome = {
            "event_id": 0,
            "period_id": 1,
            "minutes": 0,
            "seconds": 11,
            "datetime": pd.to_datetime("2018-08-18 22:15:11+0000", utc=True),
            "start_x": -30.0125,
            "start_y": -32.64,
            "team_id": 217,
            "team_side": "home",
            "pitch_size": (105.0, 68.0),
            "player_id": 5211,
            "jersey": MISSING_INT,
        }

        with open(self.event_loc, "r", encoding="utf-8") as f:
            input = json.load(f)[2]

        outcome = _get_close_to_ball_event_info(
            event=input,
            id=0,
            pitch_dimensions=MD_STATSBOMB.pitch_dimensions,
            away_team_id=MD_STATSBOMB.away_team_id,
            periods=MD_STATSBOMB.periods_frames,
            x_multiplier=105 / 120,
            y_multiplier=68 / 80,
        )
        assert expected_outcome == outcome

    def test_get_shot_event(self):
        expected_outcome = SHOT_EVENT_STATSBOMB[2].copy()
        expected_outcome.start_y *= -1
        expected_outcome.jersey = MISSING_INT
        with open(self.event_loc, "r", encoding="utf-8") as f:
            input = json.load(f)[4]

        res = _get_shot_event(
            event=input,
            id=2,
            pitch_dimensions=MD_STATSBOMB.pitch_dimensions,
            periods=MD_STATSBOMB.periods_frames,
            away_team_id=MD_STATSBOMB.away_team_id,
            x_multiplier=105 / 120,
            y_multiplier=68 / 80,
        )
        assert expected_outcome == res

    def test_get_pass_event(self):
        expected_outcome = PASS_EVENT_STATSBOMB[0].copy()
        expected_outcome.start_y *= -1
        expected_outcome.end_y *= -1
        expected_outcome.jersey = MISSING_INT
        with open(self.event_loc, "r", encoding="utf-8") as f:
            input = json.load(f)[2]

        res = _get_pass_event(
            event=input,
            id=0,
            pitch_dimensions=MD_STATSBOMB.pitch_dimensions,
            periods=MD_STATSBOMB.periods_frames,
            away_team_id=MD_STATSBOMB.away_team_id,
            x_multiplier=105 / 120,
            y_multiplier=68 / 80,
        )
        assert expected_outcome == res

    def test_get_dribble_event(self):
        expected_outcome = DRIBBLE_EVENT_STATSBOMB[3].copy()
        expected_outcome.start_x *= -1
        expected_outcome.jersey = MISSING_INT
        with open(self.event_loc, "r", encoding="utf-8") as f:
            input = json.load(f)[5]

        res = _get_dribble_event(
            event=input,
            id=3,
            pitch_dimensions=MD_STATSBOMB.pitch_dimensions,
            periods=MD_STATSBOMB.periods_frames,
            away_team_id=MD_STATSBOMB.away_team_id,
            x_multiplier=105 / 120,
            y_multiplier=68 / 80,
        )
        assert expected_outcome == res
