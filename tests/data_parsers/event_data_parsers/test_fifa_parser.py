"""
Unit tests for fifa_parser.py.

Run with:  pytest test_fifa_parser.py -v
"""

import os
import unittest

import numpy as np
import pandas as pd

from databallpy.data_parsers import Metadata
from databallpy.data_parsers.event_data_parsers.fifa_parser import (
    _determine_period_flips,
    _get_game_score,
    _get_player_info,
    _get_transformed_coordinates,
    _load_event_data,
    _load_metadata,
    _make_pass_instance,
    _make_shot_event_instance,
    load_fifa_event_data,
)
from databallpy.events import PassEvent, ShotEvent
from databallpy.utils.constants import MISSING_INT

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "test_data")
METADATA_LOC = os.path.join(TEST_DATA_DIR, "fifa_metadata_test.json")
EVENTS_LOC = os.path.join(TEST_DATA_DIR, "fifa_events_test.json")

_UTC = "UTC"
_CET = "Europe/Amsterdam"
_KO_UTC = pd.Timestamp("2023-01-01T14:00:00", tz=_UTC)


def _evt_dt(ms: int) -> pd.Timestamp:
    """Event datetime in CET given match_time_in_ms."""
    return (_KO_UTC + pd.to_timedelta(ms / 1000.0, unit="s")).tz_convert(_CET)


_PERIODS_FRAMES = pd.DataFrame(
    {
        "period_id": [1, 2, 3, 4, 5],
        "start_datetime_ed": pd.to_datetime(
            ["2023-01-01T14:00:00Z", "2023-01-01T15:00:00Z", pd.NaT, pd.NaT, pd.NaT],
            utc=True,
        ).tz_convert(_CET),
        "end_datetime_ed": pd.to_datetime(
            ["2023-01-01T14:45:00Z", "2023-01-01T15:45:00Z", pd.NaT, pd.NaT, pd.NaT],
            utc=True,
        ).tz_convert(_CET),
    }
)

HOME_PLAYERS_FIFA = pd.DataFrame(
    {
        "id": [1001, 1002],
        "full_name": ["Home Keeper", "Home Striker"],
        "formation_place": [MISSING_INT, MISSING_INT],
        "position": ["unspecified", "unspecified"],
        "starter": [False, False],
        "shirt_num": [1, 9],
    }
)

AWAY_PLAYERS_FIFA = pd.DataFrame(
    {
        "id": [2001, 2002],
        "full_name": ["Away Keeper", "Away Striker"],
        "formation_place": [MISSING_INT, MISSING_INT],
        "position": ["unspecified", "unspecified"],
        "starter": [False, False],
        "shirt_num": [1, 9],
    }
)

MD_FIFA = Metadata(
    game_id=99999,
    pitch_dimensions=[105.0, 68.0],
    periods_frames=_PERIODS_FRAMES,
    frame_rate=MISSING_INT,
    home_team_id=100,
    home_team_name="HOME TEAM",
    home_players=HOME_PLAYERS_FIFA,
    home_score=1,
    home_formation="4231",
    away_team_id=200,
    away_team_name="AWAY TEAM",
    away_players=AWAY_PLAYERS_FIFA,
    away_score=2,
    away_formation="442",
    country="Netherlands",
)

ED_FIFA = pd.DataFrame(
    {
        "event_id": [1, 2, 3, 4, 5, 6, 7],
        "databallpy_event": [None, "pass", "pass", "shot", "shot", "shot", "own_goal"],
        "period_id": [1, 1, 1, 1, 2, 2, 2],
        "minutes": [0, 0, 0, 1, 61, 83, 90],
        "seconds": [0.0, 5.0, 30.0, 0.0, 40.0, 20.0, 0.0],
        "player_id": [1002, 1002, 2002, 2002, 1002, 2002, 1001],
        "player_name": [
            "Home Striker",
            "Home Striker",
            "Away Striker",
            "Away Striker",
            "Home Striker",
            "Away Striker",
            "Home Keeper",
        ],
        "team_id": [100, 100, 200, 200, 100, 200, 100],
        "is_successful": pd.array(
            [pd.NA, True, pd.NA, False, True, True, True], dtype="boolean"
        ),
        "start_x": [0.0, -26.25, 26.25, 26.25, -26.25, 26.25, 26.25],
        "start_y": [0.0, 0.0, -17.0, 0.0, 0.0, 0.0, -17.0],
        "datetime": [
            _evt_dt(0),
            _evt_dt(5_000),
            _evt_dt(30_000),
            _evt_dt(60_000),
            _evt_dt(3_700_000),
            _evt_dt(5_000_000),
            _evt_dt(5_400_000),
        ],
        "phase": ["in_possession"] * 7,
        "original_event_id": [101, 102, 103, 104, 105, 106, 107],
        "original_event": [
            "kickoff",
            "pass",
            "cross",
            "attempt_at_goal",
            "goal",
            "goal",
            "own_goal",
        ],
        "original_event_type": [
            "start_restart",
            "pass_like",
            "pass_like",
            "shot",
            "shot",
            "shot",
            "shot",
        ],
    }
)

PASS_INSTANCES_FIFA = {
    2: PassEvent(
        event_id=2,
        period_id=1,
        minutes=0,
        seconds=5,
        datetime=_evt_dt(5_000),
        start_x=-26.25,
        start_y=0.0,
        pitch_size=[105.0, 68.0],
        team_id=100,
        team_side="home",
        player_id=1002,
        jersey=9,
        related_event_id=MISSING_INT,
        set_piece="no_set_piece",
        possession_type="open_play",
        body_part="right_foot",
        outcome=True,
        outcome_str="successful",
        end_x=26.25,
        end_y=0.0,
        pass_type="unspecified",
        _xt=-1.0,
    ),
    3: PassEvent(
        event_id=3,
        period_id=1,
        minutes=0,
        seconds=30,
        datetime=_evt_dt(30_000),
        start_x=26.25,
        start_y=-17.0,
        pitch_size=[105.0, 68.0],
        team_id=200,
        team_side="away",
        player_id=2002,
        jersey=9,
        related_event_id=MISSING_INT,
        set_piece="no_set_piece",
        possession_type="open_play",
        body_part="left_foot",
        outcome=True,
        outcome_str="successful",
        end_x=0.0,
        end_y=17.0,
        pass_type="cross",
        _xt=-1.0,
    ),
}

SHOT_INSTANCES_FIFA = {
    4: ShotEvent(
        event_id=4,
        period_id=1,
        minutes=1,
        seconds=0,
        datetime=_evt_dt(60_000),
        start_x=26.25,
        start_y=0.0,
        pitch_size=[105.0, 68.0],
        team_id=200,
        team_side="away",
        player_id=2002,
        jersey=9,
        related_event_id=MISSING_INT,
        set_piece="no_set_piece",
        possession_type="open_play",
        body_part="head",
        outcome=False,
        outcome_str="miss_on_target",
        _xt=-1.0,
    ),
    5: ShotEvent(
        event_id=5,
        period_id=2,
        minutes=61,
        seconds=40,
        datetime=_evt_dt(3_700_000),
        start_x=-26.25,
        start_y=0.0,
        pitch_size=[105.0, 68.0],
        team_id=100,
        team_side="home",
        player_id=1002,
        jersey=9,
        related_event_id=MISSING_INT,
        set_piece="no_set_piece",
        possession_type="open_play",
        body_part="unspecified",
        outcome=True,
        outcome_str="goal",
        _xt=-1.0,
    ),
    6: ShotEvent(
        event_id=6,
        period_id=2,
        minutes=83,
        seconds=20,
        datetime=_evt_dt(5_000_000),
        start_x=26.25,
        start_y=0.0,
        pitch_size=[105.0, 68.0],
        team_id=200,
        team_side="away",
        player_id=2002,
        jersey=9,
        related_event_id=MISSING_INT,
        set_piece="no_set_piece",
        possession_type="open_play",
        body_part="unspecified",
        outcome=True,
        outcome_str="goal",
        _xt=-1.0,
    ),
    7: ShotEvent(
        event_id=7,
        period_id=2,
        minutes=90,
        seconds=0,
        datetime=_evt_dt(5_400_000),
        start_x=26.25,
        start_y=-17.0,
        pitch_size=[105.0, 68.0],
        team_id=100,
        team_side="home",
        player_id=1001,
        jersey=1,
        related_event_id=MISSING_INT,
        set_piece="no_set_piece",
        possession_type="open_play",
        body_part="unspecified",
        outcome=False,
        outcome_str="own_goal",
        _xt=-1.0,
    ),
}


class TestFifaParser(unittest.TestCase):
    def setUp(self):
        self.metadata_loc = METADATA_LOC
        self.events_loc = EVENTS_LOC
        self._all_players = pd.concat(
            [HOME_PLAYERS_FIFA, AWAY_PLAYERS_FIFA], ignore_index=True
        )

    def test_load_fifa_event_data(self):
        event_data, metadata, dbp_events = load_fifa_event_data(
            self.metadata_loc, self.events_loc
        )
        pd.testing.assert_frame_equal(event_data, ED_FIFA)
        assert metadata == MD_FIFA

        assert "shot_events" in dbp_events
        for key, event in dbp_events["shot_events"].items():
            assert key in SHOT_INSTANCES_FIFA, f"Unexpected shot key {key}"
            assert event == SHOT_INSTANCES_FIFA[key], f"Shot event {key} mismatch"

        assert "pass_events" in dbp_events
        for key, event in dbp_events["pass_events"].items():
            assert key in PASS_INSTANCES_FIFA, f"Unexpected pass key {key}"
            assert event == PASS_INSTANCES_FIFA[key], f"Pass event {key} mismatch"

    def test_load_fifa_event_data_errors(self):
        with self.assertRaises(TypeError):
            load_fifa_event_data(123, self.events_loc)
        with self.assertRaises(TypeError):
            load_fifa_event_data(self.metadata_loc, ["not_a_string"])
        with self.assertRaises(ValueError):
            load_fifa_event_data(self.metadata_loc + ".xml", self.events_loc)
        with self.assertRaises(ValueError):
            load_fifa_event_data(self.metadata_loc, self.events_loc[:-5])

    def test_load_metadata(self):
        metadata = _load_metadata(self.metadata_loc, [105.0, 68.0])
        expected = MD_FIFA.copy()
        expected.home_score = np.nan
        expected.away_score = np.nan
        assert metadata == expected

    def test_get_player_info(self):
        players_data = [
            {"player_id": 1001, "player_name": "HOME KEEPER", "player_shirt_number": 1},
            {"player_id": 1002, "player_name": "HOME STRIKER", "player_shirt_number": 9},
        ]
        result = _get_player_info(players_data)
        pd.testing.assert_frame_equal(result, HOME_PLAYERS_FIFA)

    def test_load_event_data(self):
        event_data, dbp_events = _load_event_data(
            self.events_loc,
            home_team_id=100,
            away_team_id=200,
            players=self._all_players,
        )
        self.assertEqual(len(event_data), 7)
        self.assertNotIn("player_name", event_data.columns)
        self.assertIn("shot_events", dbp_events)
        self.assertIn("pass_events", dbp_events)
        self.assertEqual(set(dbp_events["pass_events"].keys()), {2, 3})
        self.assertEqual(set(dbp_events["shot_events"].keys()), {4, 5, 6, 7})

    def test_make_pass_instance(self):
        """Regular pass: right_foot, possession_complete, no line_break."""
        event = {
            "team_id": 100,
            "from_player_id": 1002,
            "event_id": 102,
            "event_type": "pass_like",
            "category": "in_possession",
            "event": "pass",
            "half_time": 1,
            "match_time_in_ms": 5000,
            "side": None,
            "x": 0.25,
            "y": 0.5,
            "outcome": "possession_complete",
            "body_type": "right_foot",
            "origin": None,
            "line_break_direction": None,
            "x_location_start": 0.25,
            "y_location_start": 0.5,
            "x_location_end": 0.75,
            "y_location_end": 0.5,
        }
        pass_event = _make_pass_instance(
            event,
            home_team_id=100,
            away_team_id=200,
            players=self._all_players,
            id=2,
            period_id=1,
            flip_first_half=False,
            flip_second_half=True,
            kickoff_time=_KO_UTC,
        )
        expected = PassEvent(
            event_id=2,
            period_id=1,
            minutes=0,
            seconds=5,
            datetime=_KO_UTC + pd.to_timedelta(5, unit="s"),
            start_x=-26.25,
            start_y=0.0,
            pitch_size=[105.0, 68.0],
            team_id=100,
            team_side="home",
            player_id=1002,
            jersey=9,
            related_event_id=MISSING_INT,
            set_piece="no_set_piece",
            possession_type="open_play",
            body_part="right_foot",
            outcome=True,
            outcome_str="successful",
            end_x=26.25,
            end_y=0.0,
            pass_type="unspecified",
            _xt=-1.0,
        )
        assert pass_event == expected

    def test_make_cross_instance(self):
        """Cross: pass_type='cross', left_foot, away team."""
        event = {
            "team_id": 200,
            "from_player_id": 2002,
            "event_id": 103,
            "event_type": "pass_like",
            "category": "in_possession",
            "event": "cross",
            "half_time": 1,
            "match_time_in_ms": 30000,
            "side": None,
            "x": 0.75,
            "y": 0.25,
            "outcome": "possession_complete",
            "body_type": "left_foot",
            "origin": None,
            "line_break_direction": None,
            "x_location_start": 0.75,
            "y_location_start": 0.25,
            "x_location_end": 0.5,
            "y_location_end": 0.75,
        }
        pass_event = _make_pass_instance(
            event,
            home_team_id=100,
            away_team_id=200,
            players=self._all_players,
            id=3,
            period_id=1,
            flip_first_half=False,
            flip_second_half=True,
            kickoff_time=_KO_UTC,
        )
        expected = PassEvent(
            event_id=3,
            period_id=1,
            minutes=0,
            seconds=30,
            datetime=_KO_UTC + pd.to_timedelta(30, unit="s"),
            start_x=26.25,
            start_y=-17.0,
            pitch_size=[105.0, 68.0],
            team_id=200,
            team_side="away",
            player_id=2002,
            jersey=9,
            related_event_id=MISSING_INT,
            set_piece="no_set_piece",
            possession_type="open_play",
            body_part="left_foot",
            outcome=True,
            outcome_str="successful",
            end_x=0.0,
            end_y=17.0,
            pass_type="cross",
            _xt=-1.0,
        )
        assert pass_event == expected

    def test_make_shot_event_instance_on_target(self):
        """attempt_at_goal with outcome='on_target' → miss_on_target."""
        event = {
            "team_id": 200,
            "from_player_id": 2002,
            "event_id": 104,
            "event_type": "shot",
            "category": "in_possession",
            "event": "attempt_at_goal",
            "half_time": 1,
            "match_time_in_ms": 60000,
            "side": None,
            "x": 0.75,
            "y": 0.5,
            "outcome": "on_target",
            "body_type": "head",
            "origin": None,
            "line_break_direction": None,
            "x_location_start": None,
            "y_location_start": None,
            "x_location_end": None,
            "y_location_end": None,
        }
        shot_event = _make_shot_event_instance(
            event,
            home_team_id=100,
            away_team_id=200,
            players=self._all_players,
            id=4,
            period_id=1,
            flip_first_half=False,
            flip_second_half=True,
            kickoff_time=_KO_UTC,
        )
        expected = ShotEvent(
            event_id=4,
            period_id=1,
            minutes=1,
            seconds=0,
            datetime=_KO_UTC + pd.to_timedelta(60, unit="s"),
            start_x=26.25,
            start_y=0.0,
            pitch_size=[105.0, 68.0],
            team_id=200,
            team_side="away",
            player_id=2002,
            jersey=9,
            related_event_id=MISSING_INT,
            set_piece="no_set_piece",
            possession_type="open_play",
            body_part="head",
            outcome=False,
            outcome_str="miss_on_target",
            _xt=-1.0,
        )
        assert shot_event == expected

    def test_make_shot_event_instance_goal(self):
        """Standalone 'goal' event (FIFA records scored goals separately)."""
        event = {
            "team_id": 100,
            "from_player_id": 1002,
            "event_id": 105,
            "event_type": "shot",
            "category": "in_possession",
            "event": "goal",
            "half_time": 2,
            "match_time_in_ms": 3_700_000,
            "side": None,
            "x": 0.75,
            "y": 0.5,
            "outcome": None,
            "body_type": None,
            "origin": None,
            "line_break_direction": None,
            "x_location_start": None,
            "y_location_start": None,
            "x_location_end": None,
            "y_location_end": None,
        }
        shot_event = _make_shot_event_instance(
            event,
            home_team_id=100,
            away_team_id=200,
            players=self._all_players,
            id=5,
            period_id=2,
            flip_first_half=False,
            flip_second_half=True,
            kickoff_time=_KO_UTC,
        )
        expected = ShotEvent(
            event_id=5,
            period_id=2,
            minutes=61,
            seconds=40,
            datetime=_KO_UTC + pd.to_timedelta(3700, unit="s"),
            start_x=-26.25,
            start_y=0.0,
            pitch_size=[105.0, 68.0],
            team_id=100,
            team_side="home",
            player_id=1002,
            jersey=9,
            related_event_id=MISSING_INT,
            set_piece="no_set_piece",
            possession_type="open_play",
            body_part="unspecified",
            outcome=True,
            outcome_str="goal",
            _xt=-1.0,
        )
        assert shot_event == expected

    def test_make_shot_event_instance_own_goal(self):
        """own_goal: outcome=False, outcome_str='own_goal', coords flipped in P2."""
        event = {
            "team_id": 100,
            "from_player_id": 1001,
            "event_id": 107,
            "event_type": "shot",
            "category": "in_possession",
            "event": "own_goal",
            "half_time": 2,
            "match_time_in_ms": 5_400_000,
            "side": None,
            "x": 0.25,
            "y": 0.75,
            "outcome": None,
            "body_type": None,
            "origin": None,
            "line_break_direction": None,
            "x_location_start": None,
            "y_location_start": None,
            "x_location_end": None,
            "y_location_end": None,
        }
        shot_event = _make_shot_event_instance(
            event,
            home_team_id=100,
            away_team_id=200,
            players=self._all_players,
            id=7,
            period_id=2,
            flip_first_half=False,
            flip_second_half=True,
            kickoff_time=_KO_UTC,
        )
        expected = ShotEvent(
            event_id=7,
            period_id=2,
            minutes=90,
            seconds=0,
            datetime=_KO_UTC + pd.to_timedelta(5400, unit="s"),
            start_x=26.25,
            start_y=-17.0,
            pitch_size=[105.0, 68.0],
            team_id=100,
            team_side="home",
            player_id=1001,
            jersey=1,
            related_event_id=MISSING_INT,
            set_piece="no_set_piece",
            possession_type="open_play",
            body_part="unspecified",
            outcome=False,
            outcome_str="own_goal",
            _xt=-1.0,
        )
        assert shot_event == expected

    def test_get_transformed_coordinates_no_flip(self):
        """Period 1, flip_first_half=False: linear rescale only."""
        cases = [
            (0.5, 0.5, 0.0, 0.0),  # centre
            (0.25, 0.5, -26.25, 0.0),  # pass start
            (0.75, 0.5, 26.25, 0.0),  # pass end
            (0.75, 0.25, 26.25, -17.0),  # cross start
            (0.5, 0.75, 0.0, 17.0),  # cross end
        ]
        for x_in, y_in, x_exp, y_exp in cases:
            with self.subTest(x=x_in, y=y_in):
                x, y = _get_transformed_coordinates(
                    x_in, y_in, [105.0, 68.0], 1, False, True
                )
                self.assertAlmostEqual(x, x_exp)
                self.assertAlmostEqual(y, y_exp)

    def test_get_transformed_coordinates_with_flip(self):
        """Period 2, flip_second_half=True: mirror then rescale."""
        cases = [
            (0.75, 0.5, -26.25, 0.0),  # home goal: 1-0.75=0.25 → -26.25
            (0.25, 0.5, 26.25, 0.0),  # away goal: 1-0.25=0.75 → 26.25
            (0.25, 0.75, 26.25, -17.0),  # own_goal:  1-0.25=0.75, 1-0.75=0.25
        ]
        for period_id in [1, 2]:
            for x_in, y_in, x_exp, y_exp in cases:
                with self.subTest(x=x_in, y=y_in):
                    x, y = _get_transformed_coordinates(
                        x_in, y_in, [105.0, 68.0], period_id, True, True
                    )
                    self.assertAlmostEqual(x, x_exp)
                    self.assertAlmostEqual(y, y_exp)

    def test_get_transformed_coordinates_null(self):
        """None or NaN coordinates return (NaN, NaN)."""
        import numpy as np

        for x_in, y_in in [(None, 0.5), (0.5, None), (None, None)]:
            with self.subTest(x=x_in, y=y_in):
                x, y = _get_transformed_coordinates(
                    x_in, y_in, [105.0, 68.0], 1, False, True
                )
                self.assertTrue(np.isnan(x))
                self.assertTrue(np.isnan(y))

    def test_determine_period_flips_home_side_l(self):
        """Home team kicks off from left (side='l'): no first-half flip."""
        events = [
            {
                "half_time": 1,
                "event": "kickoff",
                "from_player_id": 1002,
                "team_id": 100,
                "side": "l",
            },
        ]
        flip_first, flip_second = _determine_period_flips(events, 100, 200)
        self.assertFalse(flip_first)
        self.assertTrue(flip_second)

    def test_determine_period_flips_home_side_r(self):
        """Home team kicks off from right (side='r'): flip first half."""
        events = [
            {
                "half_time": 1,
                "event": "kickoff",
                "from_player_id": 1002,
                "team_id": 100,
                "side": "r",
            },
        ]
        flip_first, flip_second = _determine_period_flips(events, 100, 200)
        self.assertTrue(flip_first)
        self.assertFalse(flip_second)

    def test_determine_period_flips_away_kickoff(self):
        """Away team kicks off from right (side='r'): home plays left → no flip."""
        events = [
            {
                "half_time": 1,
                "event": "kickoff",
                "from_player_id": 2002,
                "team_id": 200,
                "side": "r",
            },
        ]
        flip_first, flip_second = _determine_period_flips(events, 100, 200)
        self.assertFalse(flip_first)
        self.assertTrue(flip_second)

    def test_determine_period_flips_no_kickoff(self):
        """No kickoff event found: both flips default to False."""
        events = [
            {
                "half_time": 1,
                "event": "pass",
                "from_player_id": 1002,
                "team_id": 100,
                "side": None,
            },
        ]
        flip_first, flip_second = _determine_period_flips(events, 100, 200)
        self.assertFalse(flip_first)
        self.assertFalse(flip_second)

    def test_get_game_score(self):
        """home=1 (one goal), away=2 (one goal + one own-goal by home)."""
        event_data, _ = _load_event_data(
            self.events_loc,
            home_team_id=100,
            away_team_id=200,
            players=self._all_players,
        )
        home_score, away_score = _get_game_score(event_data, 100, 200)
        self.assertEqual(home_score, 1)
        self.assertEqual(away_score, 2)

        event_data = pd.concat(
            [
                event_data,
                pd.DataFrame({"team_id": 200, "original_event": "own_goal"}, index=[7]),
            ]
        )
        event_data = pd.concat(
            [
                event_data,
                pd.DataFrame(
                    {
                        "team_id": 200,
                        "original_event": "attempt_at_goal",
                        "is_successful": True,
                    },
                    index=[8],
                ),
            ]
        )

        home_score, away_score = _get_game_score(event_data, 100, 200)
        self.assertEqual(home_score, 2)
        self.assertEqual(away_score, 3)


if __name__ == "__main__":
    unittest.main()
