import unittest
from dataclasses import dataclass

import pandas as pd

from databallpy.data_parsers.metadata import Metadata
from databallpy.utils.constants import MISSING_INT
from databallpy.utils.game_utils import (
    _add_starter_information,
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


class TestAddStarterInformation(unittest.TestCase):
    def setUp(self):
        self.home_players = pd.DataFrame(
            {
                "id": [101, 102, 103],
                "full_name": ["Player 1", "Player 2", "Player 3"],
                "shirt_num": [1, 2, 3],
                "position": ["goalkeeper", "defender", "midfielder"],
                "start_frame": [0, 0, 100],
                "end_frame": [1000, 1000, 1000],
                "starter": [False, False, False],
            }
        )
        self.away_players = pd.DataFrame(
            {
                "id": [201, 202, 203],
                "full_name": ["Player 4", "Player 5", "Player 6"],
                "shirt_num": [1, 2, 3],
                "position": ["goalkeeper", "defender", "midfielder"],
                "start_frame": [0, 0, 100],
                "end_frame": [1000, 1000, 1000],
                "starter": [False, False, False],
            }
        )
        self.periods_frames = pd.DataFrame(
            {
                "period_id": [1, 2, 3, 4, 5],
                "start_frame": [0, 500, MISSING_INT, MISSING_INT, MISSING_INT],
                "end_frame": [499, 999, MISSING_INT, MISSING_INT, MISSING_INT],
            }
        )
        self.metadata = Metadata(
            game_id=1,
            pitch_dimensions=[105.0, 68.0],
            periods_frames=self.periods_frames,
            frame_rate=10,
            home_team_id=1,
            home_team_name="Home Team",
            home_players=self.home_players.copy(),
            home_score=0,
            home_formation="4-3-3",
            away_team_id=2,
            away_team_name="Away Team",
            away_players=self.away_players.copy(),
            away_score=0,
            away_formation="4-4-2",
            country="Test Country",
        )

    def test_with_tracking_data(self):
        # Create tracking data where players 1 and 2 from each team have data at frame 0
        tracking_data = pd.DataFrame(
            {
                "frame": [0, 1, 2],
                "home_1_x": [10.0, 11.0, 12.0],
                "home_1_y": [20.0, 21.0, 22.0],
                "home_2_x": [15.0, 16.0, 17.0],
                "home_2_y": [25.0, 26.0, 27.0],
                "home_3_x": [None, None, None],
                "home_3_y": [None, None, None],
                "away_1_x": [-10.0, -11.0, -12.0],
                "away_1_y": [20.0, 21.0, 22.0],
                "away_2_x": [-15.0, -16.0, -17.0],
                "away_2_y": [25.0, 26.0, 27.0],
                "away_3_x": [None, None, None],
                "away_3_y": [None, None, None],
            }
        )

        result = _add_starter_information(self.metadata, tracking_data=tracking_data)

        # Players 1 and 2 should be marked as starters
        self.assertTrue(
            result.home_players.loc[
                result.home_players["shirt_num"] == 1, "starter"
            ].iloc[0]
        )
        self.assertTrue(
            result.home_players.loc[
                result.home_players["shirt_num"] == 2, "starter"
            ].iloc[0]
        )
        self.assertFalse(
            result.home_players.loc[
                result.home_players["shirt_num"] == 3, "starter"
            ].iloc[0]
        )

        self.assertTrue(
            result.away_players.loc[
                result.away_players["shirt_num"] == 1, "starter"
            ].iloc[0]
        )
        self.assertTrue(
            result.away_players.loc[
                result.away_players["shirt_num"] == 2, "starter"
            ].iloc[0]
        )
        self.assertFalse(
            result.away_players.loc[
                result.away_players["shirt_num"] == 3, "starter"
            ].iloc[0]
        )

    def test_with_event_data(self):
        # Create event data where players 1 and 2 participated before first substitute
        event_data = pd.DataFrame(
            {
                "event_id": [1, 2, 3, 4, 5, 6],
                "player_id": [101, 102, 201, 202, 203, 103],
                "original_event": [
                    "pass",
                    "pass",
                    "pass",
                    "pass",
                    "substitution",
                    "pass",
                ],
                "databallpy_event": ["pass", "pass", "pass", "pass", None, "pass"],
            }
        )

        result = _add_starter_information(self.metadata, event_data=event_data)

        # Players 1 and 2 from home, and player 1 and 2 from away should be starters
        # Player 3 from away is involved in the substitution, so should not be a starter
        self.assertTrue(
            result.home_players.loc[result.home_players["id"] == 101, "starter"].iloc[0]
        )
        self.assertTrue(
            result.home_players.loc[result.home_players["id"] == 102, "starter"].iloc[0]
        )
        self.assertFalse(
            result.home_players.loc[result.home_players["id"] == 103, "starter"].iloc[0]
        )

        self.assertTrue(
            result.away_players.loc[result.away_players["id"] == 201, "starter"].iloc[0]
        )
        self.assertTrue(
            result.away_players.loc[result.away_players["id"] == 202, "starter"].iloc[0]
        )
        self.assertFalse(
            result.away_players.loc[result.away_players["id"] == 203, "starter"].iloc[0]
        )

    def test_with_existing_starter_info(self):
        # Add starter info to metadata
        self.metadata.home_players["starter"] = [True, True, False]
        self.metadata.away_players["starter"] = [True, True, False]

        # Create tracking data that would suggest different starters
        tracking_data = pd.DataFrame(
            {
                "frame": [0],
                "home_1_x": [None],
                "home_2_x": [None],
                "home_3_x": [10.0],
                "away_1_x": [None],
                "away_2_x": [None],
                "away_3_x": [-10.0],
            }
        )

        result = _add_starter_information(self.metadata, tracking_data=tracking_data)

        # Should keep existing starter info, not overwrite
        self.assertTrue(
            result.home_players.loc[
                result.home_players["shirt_num"] == 1, "starter"
            ].iloc[0]
        )
        self.assertTrue(
            result.home_players.loc[
                result.home_players["shirt_num"] == 2, "starter"
            ].iloc[0]
        )
        self.assertFalse(
            result.home_players.loc[
                result.home_players["shirt_num"] == 3, "starter"
            ].iloc[0]
        )

    def test_with_no_data(self):
        # Should initialize starter column to False when no data provided
        result = _add_starter_information(self.metadata)

        self.assertIn("starter", result.home_players.columns)
        self.assertIn("starter", result.away_players.columns)
        self.assertFalse(result.home_players["starter"].any())
        self.assertFalse(result.away_players["starter"].any())
