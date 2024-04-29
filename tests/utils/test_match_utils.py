import unittest
from dataclasses import dataclass

import pandas as pd

from databallpy.utils.match_utils import (
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
