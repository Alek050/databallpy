import unittest

import numpy as np
import pandas as pd

from databallpy.load_data.event_data.opta import load_opta_event_data
from databallpy.load_data.tracking_data.tracab import load_tracab_tracking_data
from databallpy.match import Match, _create_sim_mat, _needleman_wunsch, get_match


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
                "start_time": [
                    np.datetime64("2023-01-14 00:00:04"),
                    np.datetime64("2023-01-14 00:00:08"),
                    np.datetime64("2023-01-14 00:00:12"),
                    np.datetime64("2023-01-14 00:00:16"),
                    np.nan,
                ],
                "end_time": [
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

        period_conditions = [
            (self.td["timestamp"] <= expected_periods.loc[0, "end_frame"]),
            (self.td["timestamp"] > expected_periods.loc[0, "end_frame"])
            & (self.td["timestamp"] < expected_periods.loc[1, "start_frame"]),
            (self.td["timestamp"] >= expected_periods.loc[1, "start_frame"])
            & (self.td["timestamp"] <= expected_periods.loc[1, "end_frame"]),
            (self.td["timestamp"] > expected_periods.loc[1, "end_frame"])
            & (self.td["timestamp"] < expected_periods.loc[2, "start_frame"]),
            (self.td["timestamp"] >= expected_periods.loc[2, "start_frame"])
            & (self.td["timestamp"] < expected_periods.loc[2, "end_frame"]),
            (self.td["timestamp"] > expected_periods.loc[2, "end_frame"])
            & (self.td["timestamp"] < expected_periods.loc[3, "start_frame"]),
            (self.td["timestamp"] >= expected_periods.loc[3, "start_frame"])
            & (self.td["timestamp"] < expected_periods.loc[3, "end_frame"]),
            (self.td["timestamp"] > expected_periods.loc[3, "end_frame"])
            & (self.td["timestamp"] < expected_periods.loc[4, "start_frame"]),
            (self.td["timestamp"] > expected_periods.loc[4, "start_frame"]),
        ]
        period_values = [
            "1",
            "Break after 1",
            "2",
            "Break after 2",
            "3",
            "Break after 3",
            "4",
            "Break after 4",
            "5",
        ]
        self.td["period"] = np.select(period_conditions, period_values)

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

        self.match_to_sync = get_match(
            tracking_data_loc="tests/test_data/sync/tracab_td_sync_test.dat",
            tracking_metadata_loc="tests/test_data/sync/tracab_metadata_sync_test.xml",
            tracking_data_provider="tracab",
            event_data_loc="tests/test_data/sync/opta_events_sync_test.xml",
            event_metadata_loc="tests/test_data/sync/opta_metadata_sync_test.xml",
            event_data_provider="opta",
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

    def test_get_match_rigth_to_left(self):
        flipped_match = get_match(
            tracking_data_loc="tests/test_data/tracab_td_test_flipped.dat",
            tracking_metadata_loc=self.td_md_loc,
            event_data_loc=self.ed_loc,
            event_metadata_loc=self.ed_md_loc,
            tracking_data_provider=self.td_provider,
            event_data_provider=self.ed_provider,
        )
        assert flipped_match == self.expected_match

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

    def test_match_player_id_to_column_id(self):
        res_column_id_home = self.expected_match.player_id_to_column_id(19367)
        assert res_column_id_home == "home_1"

        res_column_id_away = self.expected_match.player_id_to_column_id(450445)
        assert res_column_id_away == "away_2"

        with self.assertRaises(ValueError):
            self.expected_match.player_id_to_column_id(4)

    def test_synchronise_tracking_and_event_data(self):
        expected_event_data = self.match_to_sync.event_data
        expected_tracking_data = self.match_to_sync.tracking_data
        expected_tracking_data["period"] = ["1"] * 13
        expected_tracking_data["event"] = [
            np.nan,
            "pass",
            np.nan,
            np.nan,
            np.nan,
            "pass",
            np.nan,
            np.nan,
            np.nan,
            "take on",
            np.nan,
            "tackle",
            np.nan,
        ]
        expected_tracking_data["event_id"] = [
            np.nan,
            2499594225,
            np.nan,
            np.nan,
            np.nan,
            2499594243,
            np.nan,
            np.nan,
            np.nan,
            2499594285,
            np.nan,
            2499594291,
            np.nan,
        ]
        
        expected_event_data.loc[:, "tracking_frame"] = [
                np.nan,
                np.nan, 
                np.nan,
                1.0, 
                5.0, 
                np.nan,
                np.nan,
                9.0, 
                11.0
            ]
        expected_event_data = expected_event_data[
            expected_event_data["type_id"].isin([1, 3, 7])
        ]
        

        synced_match = self.match_to_sync.synchronise_tracking_and_event_data(
            n_batches_per_half=1
        )
        pd.testing.assert_frame_equal(
            synced_match.tracking_data, expected_tracking_data
        )
        pd.testing.assert_frame_equal(synced_match.event_data, expected_event_data)

    def test_needleman_wunsch(self):
        sim_list = [
            0,
            0,
            0,
            0.9,
            0,
            0,
            0,
            0,
            0,
            0,
            0.9,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.9,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        sim_mat = np.array(sim_list).reshape(10, 3)

        res = _needleman_wunsch(sim_mat)
        expected_res = {0: 1, 1: 3, 2: 7}

        assert res == expected_res

    def test_create_sim_mat(self):
        expected_res = np.array(
            [
                0.36405464,
                0.36405464,
                0.36074384,
                0.39021513,
                0.35807246,
                0.35807246,
                0.35624062,
                0.39082216,
                0.35198235,
                0.35198235,
                0.35163648,
                0.39121355,
                0.33188426,
                0.33188426,
                0.33188426,
                0.33188426,
                0.33740283,
                0.33740283,
                0.34275047,
                0.39173908,
                0.36386003,
                0.36386003,
                0.36532987,
                0.3936558,
                0.3565997,
                0.3565997,
                0.36105897,
                0.39214357,
                0.34917505,
                0.34917505,
                0.35657184,
                0.37802085,
                0.33525554,
                0.33525554,
                0.34660111,
                0.36787944,
                0.36100615,
                0.36100615,
                0.36888422,
                0.36578695,
                0.35317347,
                0.35317347,
                0.36312554,
                0.37869339,
                0.34546462,
                0.34546462,
                0.35667513,
                0.38356556,
                0.33188426,
                0.33188426,
                0.34538215,
                0.38410493,
            ]
        )
        expected_res = expected_res.reshape(13, 4)

        tracking_data = self.match_to_sync.tracking_data
        date = np.datetime64(str(self.match_to_sync.periods.iloc[0, 3])[:10])
        tracking_data["datetime"] = [
            date + np.timedelta64(int(x / self.match_to_sync.frame_rate * 1000), "ms")
            for x in tracking_data["timestamp"]
        ]
        tracking_data.reset_index(inplace=True)
        event_data = self.match_to_sync.event_data
        event_data = event_data[event_data["type_id"].isin([1, 3, 7])].reset_index()
        res = _create_sim_mat(
            tracking_batch=tracking_data,
            event_batch=event_data,
            match=self.match_to_sync,
        )

        np.testing.assert_allclose(expected_res, res)

    def test_create_sim_mat_without_player(self):
        expected_res = np.array(
            [
                0.00204535,
                0.00204535,
                0.36787944,
                0.00312935,
                0.00184787,
                0.00184787,
                0.37415304,
                0.0031593,
                0.00166347,
                0.00166347,
                0.38057966,
                0.00317874,
                0.00116021,
                0.00116021,
                0.00116021,
                0.00116021,
                0.0012836,
                0.0012836,
                0.39314713,
                0.003205,
                0.00203865,
                0.00203865,
                0.39346724,
                0.00330231,
                0.00180178,
                0.00180178,
                0.40017719,
                0.00322533,
                0.00158381,
                0.00158381,
                0.40705082,
                0.00257606,
                0.00123435,
                0.00123435,
                0.41487527,
                0.00218063,
                0.00194262,
                0.00194262,
                0.41521307,
                0.00210572,
                0.00169827,
                0.00169827,
                0.41665267,
                0.00260428,
                0.00148345,
                0.00148345,
                0.41256202,
                0.0028165,
                0.00116021,
                0.00116021,
                0.40933317,
                0.00284086,
            ]
        )
        expected_res = expected_res.reshape(13, 4)

        tracking_data = self.match_to_sync.tracking_data
        date = np.datetime64(str(self.match_to_sync.periods.iloc[0, 3])[:10])
        tracking_data["datetime"] = [
            date + np.timedelta64(int(x / self.match_to_sync.frame_rate * 1000), "ms")
            for x in tracking_data["timestamp"]
        ]
        tracking_data.reset_index(inplace=True)
        event_data = self.match_to_sync.event_data
        event_data = event_data[event_data["type_id"].isin([1, 3, 7])].reset_index()
        event_data.iloc[2, 7] = "not a float"

        res = _create_sim_mat(
            tracking_batch=tracking_data,
            event_batch=event_data,
            match=self.match_to_sync,
        )

        np.testing.assert_allclose(expected_res, res, rtol=1e-05)
