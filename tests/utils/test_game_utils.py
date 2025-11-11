import unittest
from dataclasses import dataclass

import pandas as pd

from databallpy.utils.game_utils import (
    _remove_offside_players,
    create_event_attributes_dataframe,
    player_column_id_to_full_name,
    player_id_to_column_id,
)


class TestMatchUtils(unittest.TestCase):
    def setUp(self):
        self.home_players = pd.DataFrame(
            {
                "shirt_num": [1, 2],
                "full_name": ["Home Player 1", "Home Player 2"],
                "id": [101, 102],
            }
        )
        self.away_players = pd.DataFrame(
            {
                "shirt_num": [1, 2],
                "full_name": ["Away Player 1", "Away Player 2"],
                "id": [201, 202],
            }
        )

    def test_player_column_id_to_full_name(self):
        self.assertEqual(
            player_column_id_to_full_name(
                self.home_players, self.away_players, "home_1"
            ),
            "Home Player 1",
        )
        self.assertEqual(
            player_column_id_to_full_name(
                self.home_players, self.away_players, "away_2"
            ),
            "Away Player 2",
        )

    def test_player_id_to_column_id(self):
        self.assertEqual(
            player_id_to_column_id(self.home_players, self.away_players, 101), "home_1"
        )
        self.assertEqual(
            player_id_to_column_id(self.home_players, self.away_players, 202), "away_2"
        )

        with self.assertRaises(ValueError):
            player_id_to_column_id(self.home_players, self.away_players, 999)

    def test_create_event_attributes_dataframe(self):
        @dataclass
        class Event:
            event_id: int
            event_type: str
            event_team: str

            @property
            def df_attributes(self):
                return ["event_id", "event_type", "event_team"]

        events = {
            1: Event(1, "pass", "home"),
            2: Event(2, "shot", "away"),
        }
        df = create_event_attributes_dataframe(events)
        expected_df = pd.DataFrame(
            {
                "event_id": [1, 2],
                "event_type": ["pass", "shot"],
                "event_team": ["home", "away"],
            }
        )
        pd.testing.assert_frame_equal(df, expected_df)

        df = create_event_attributes_dataframe({})
        pd.testing.assert_frame_equal(df, pd.DataFrame())


class TestRemoveOffsidePlayers(unittest.TestCase):
    def test_ball_not_alive(self):
        frame = pd.Series({"ball_status": "dead", "team_possession": "home"})
        col_ids = ["home_1", "away_2"]
        self.assertEqual(_remove_offside_players(col_ids, frame), col_ids)

    def test_home_possession_no_offside(self):
        frame = pd.Series(
            {
                "ball_status": "alive",
                "team_possession": "home",
                "ball_x": 30,
                "home_1_x": 25,
                "home_2_x": 28,
                "away_1_x": 20,
                "away_2_x": 22,
            }
        )
        col_ids = ["home_1", "home_2", "away_1", "away_2"]
        result = _remove_offside_players(col_ids, frame)
        self.assertEqual(set(result), set(col_ids))

    def test_home_possession_with_offside(self):
        frame = pd.Series(
            {
                "ball_status": "alive",
                "team_possession": "home",
                "ball_x": 30,
                "home_1_x": 35,
                "home_2_x": 28,
                "away_1_x": 32,
                "away_2_x": 22,
            }
        )
        col_ids = ["home_1", "home_2", "away_1", "away_2"]
        result = _remove_offside_players(col_ids, frame)
        self.assertEqual(set(result), {"home_2", "away_1", "away_2"})

    def test_away_possession_with_offside_within_tolerance(self):
        frame = pd.Series(
            {
                "ball_status": "alive",
                "team_possession": "away",
                "ball_x": -10,
                "home_1_x": -21.6,
                "home_2_x": -22,
                "away_1_x": -35,
                "away_2_x": -22,
            }
        )
        col_ids = ["home_1", "home_2", "away_1", "away_2"]
        result = _remove_offside_players(col_ids, frame)
        self.assertNotIn("away_1", result)
        self.assertIn("away_2", result)
