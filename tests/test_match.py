import unittest

import numpy as np
import pandas as pd

from databallpy.load_data.event_data.opta import load_opta_event_data
from databallpy.load_data.tracking_data.tracab import load_tracab_tracking_data
from databallpy.match import Match, get_match


class TestMatch(unittest.TestCase):
    def setUp(self):
        self.td_loc = "tests/test_data/tracab_td_test.dat"
        self.td_md_loc = "tests/test_data/tracab_metadata_test.xml"
        self.td_provider = "tracab"
        self.ed_loc = "tests/test_data/f24_test.xml"
        self.ed_md_loc = "tests/test_data/f7_test.xml"
        self.ed_provider = "opta"

        self.td, self.td_md = load_tracab_tracking_data(self.td_loc, self.td_md_loc)
        self.ed, self.ed_md = load_opta_event_data(
            f7_loc=self.ed_md_loc, f24_loc=self.ed_loc
        )

        corrected_ed = self.ed.copy()
        corrected_ed["start_x"] *= (
            100.0 / 106.0
        )  # pitch dimensions of td and ed metadata
        corrected_ed["start_y"] *= 50.0 / 68.0

        expected_periods = pd.DataFrame(
            {
                "period": [1, 2, 3, 4, 5],
                "start_frame": [100, 200, 300, 400, 0],
                "end_frame": [400, 600, 900, 1200, 0],
                "start_time_td": [
                    np.datetime64("2023-01-14 00:00:04"),
                    np.datetime64("2023-01-14 00:00:08"),
                    np.datetime64("2023-01-14 00:00:12"),
                    np.datetime64("2023-01-14 00:00:16"),
                    np.nan,
                ],
                "end_time_td": [
                    np.datetime64("2023-01-14 00:00:16"),
                    np.datetime64("2023-01-14 00:00:24"),
                    np.datetime64("2023-01-14 00:00:36"),
                    np.datetime64("2023-01-14 00:00:48"),
                    np.nan,
                ],
                "start_datetime_opta": [
                    pd.to_datetime("20230122T121832+0000"),
                    pd.to_datetime("20230122T132113+0000"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                ],
                "end_datetime_opta": [
                    pd.to_datetime("20230122T130432+0000"),
                    pd.to_datetime("20230122T140958+0000"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                ],
            }
        )

        expected_home_players = pd.DataFrame(
            {
                "id": [19367, 45849],
                "full_name": ["Piet Schrijvers", "Jan Boskamp"],
                "shirt_num": [1, 2],
                "start_frame": [100, 100],
                "end_frame": [1200, 400],
                "formation_place": [4, 0],
                "position": ["midfielder", "midfielder"],
                "starter": [True, False],
            }
        )

        expected_away_players = pd.DataFrame(
            {
                "id": [184934, 450445],
                "full_name": ["Pepijn Blok", "TestSpeler"],
                "shirt_num": [1, 2],
                "start_frame": [100, 100],
                "end_frame": [1200, 400],
                "formation_place": [8, 0],
                "position": ["midfielder", "midfielder"],
                "starter": [True, False],
            }
        )

        self.expected_match = Match(
            tracking_data=self.td,
            tracking_data_provider=self.td_provider,
            event_data=corrected_ed,
            event_data_provider=self.ed_provider,
            pitch_dimensions=self.td_md.pitch_dimensions,
            periods=expected_periods,
            frame_rate=self.td_md.frame_rate,
            home_team_id=self.ed_md.home_team_id,
            home_formation=self.ed_md.home_formation,
            home_score=self.ed_md.home_score,
            home_team_name=self.ed_md.home_team_name,
            home_players=expected_home_players,
            away_team_id=self.ed_md.away_team_id,
            away_formation=self.ed_md.away_formation,
            away_score=self.ed_md.away_score,
            away_team_name=self.ed_md.away_team_name,
            away_players=expected_away_players,
        )

    def test_get_match_wrong_provider(self):
        self.assertRaises(
            AssertionError,
            get_match,
            tracking_data_loc=self.td_loc,
            tracking_metadata_loc=self.td_md_loc,
            event_data_loc=self.ed_loc,
            event_metadata_loc=self.ed_md_loc,
            tracking_data_provider=self.td_provider,
            event_data_provider="wrong",
        )

        self.assertRaises(
            AssertionError,
            get_match,
            tracking_data_loc=self.td_loc,
            tracking_metadata_loc=self.td_md_loc,
            event_data_loc=self.ed_loc,
            event_metadata_loc=self.ed_md_loc,
            tracking_data_provider="also wrong",
            event_data_provider=self.ed_provider,
        )

    def test_get_match(self):
        match = get_match(
            tracking_data_loc=self.td_loc,
            tracking_metadata_loc=self.td_md_loc,
            event_data_loc=self.ed_loc,
            event_metadata_loc=self.ed_md_loc,
            tracking_data_provider=self.td_provider,
            event_data_provider=self.ed_provider,
        )

        assert match == self.expected_match

    def test_match__eq__(self):
        assert not self.expected_match == pd.DataFrame()

    def test_match_name(self):
        assert self.expected_match.name == "TeamOne 3 - 1 TeamTwo"

    def test_match_home_players_column_ids(self):
        assert self.expected_match.home_players_column_ids == ["home_34_x", "home_34_y"]

    def test_match_away_players_column_ids(self):
        assert self.expected_match.away_players_column_ids == ["away_17_x", "away_17_y"]

    def test_match_player_column_id_to_full_name(self):
        res_name_home = self.expected_match.player_column_id_to_full_name("home_1")
        assert res_name_home == "Piet Schrijvers"

        res_name_away = self.expected_match.player_column_id_to_full_name("away_2")
        assert res_name_away == "TestSpeler"
