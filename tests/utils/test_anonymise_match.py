import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from databallpy import get_game
from databallpy.utils.anonymise_game import (
    add_new_pseudonym,
    anonymise_datetime,
    anonymise_game,
    anonymise_players,
    anonymise_teams,
    get_player_mappings,
    get_team_mappings,
    rename_tracking_data_columns,
)
from databallpy.utils.errors import DataBallPyError
from databallpy.utils.utils import MISSING_INT


class TestAnonymiseGame(unittest.TestCase):
    def setUp(self):
        self.keys = pd.DataFrame(
            {
                "name": ["real player 1", "real player 2", "real team 1"],
                "pseudonym": ["P-64b32c78", "P-24f9904f", "T-b4ac4483"],
                "salt": ["20e4b3623451260d", "59b74fe77af66edc", "a90a819f61813012"],
                "original_id": [111, 3256, "T222"],
            }
        )
        self.game = get_game(
            tracking_data_loc="tests/test_data/tracab_td_test.dat",
            tracking_metadata_loc="tests/test_data/tracab_metadata_test.xml",
            event_data_loc="tests/test_data/f24_test.xml",
            event_metadata_loc="tests/test_data/f7_test.xml",
            tracking_data_provider="tracab",
            event_data_provider="opta",
            check_quality=False,
        )
        self.game.home_players.loc[
            self.game.home_players["full_name"] == "Jan Boskamp", "shirt_num"
        ] = 34
        self.game.away_players.loc[
            self.game.away_players["full_name"] == "Pepijn Blok", "shirt_num"
        ] = 17
        tracking_data = self.game.tracking_data.copy()
        tracking_data["home_1_x"] = [0, 0, 0, 0, 0]
        tracking_data["home_1_y"] = [0, 0, 0, 0, 0]
        tracking_data["away_2_x"] = [10, 10, 10, 10, 10]
        tracking_data["away_2_y"] = [10, 10, 12, 120, 12]
        self.game.tracking_data = tracking_data.copy()

    @patch("databallpy.utils.anonymise_game.anonymise_players")
    @patch("databallpy.utils.anonymise_game.anonymise_teams")
    @patch("databallpy.utils.anonymise_game.anonymise_datetime")
    def test_anonymise_game(self, mock_an_dt, mock_an_teams, mock_an_players):
        mock_an_teams.return_value = ("an_teams_game", "an_teams_keys")
        mock_an_players.return_value = ("an_players_game", "an_players_keys")
        mock_an_dt.return_value = "an_datetimes_game"

        res_game, res_keys = anonymise_game(
            self.game.copy(),
            self.keys.copy(),
            pd.to_datetime("2020-1-1 13:00:00", utc=True),
        )
        assert res_game == "an_datetimes_game"
        assert res_keys == "an_teams_keys"

        called_args_players = mock_an_players.call_args_list[0][0]
        assert len(mock_an_players.call_args_list) == 1
        assert len(called_args_players) == 2
        assert called_args_players[0] == self.game.copy()
        pd.testing.assert_frame_equal(called_args_players[1], self.keys)

        called_args_teams = mock_an_teams.call_args_list[0][0]
        assert len(mock_an_teams.call_args_list) == 1
        assert len(called_args_teams) == 2
        assert called_args_teams[0] == "an_players_game"
        assert called_args_teams[1] == "an_players_keys"

        called_args_dt = mock_an_dt.call_args_list[0][0]
        assert len(mock_an_dt.call_args_list) == 1
        assert len(called_args_dt) == 2
        assert called_args_dt[0] == "an_teams_game"
        assert called_args_dt[1] == pd.to_datetime("2020-1-1 13:00:00", utc=True)

    def test_anonymise_game_errors(self):
        with self.assertRaises(DataBallPyError):
            anonymise_game(self.game.copy(), self.keys.copy().drop(columns="salt"))

        with self.assertRaises(DataBallPyError):
            keys = self.keys.copy()
            keys["new_col"] = 1
            anonymise_game(self.game.copy(), keys)

        with self.assertRaises(ValueError):
            anonymise_game(
                self.game.copy(), self.keys.copy(), pd.to_datetime("2020-1-1 13:00:00")
            )

    def test_add_new_pseudonym_wrong_key(self):
        with self.assertRaises(ValueError):
            add_new_pseudonym(
                self.keys.copy(), key_type="competition", name="team2", old_id="t123"
            )

    def test_add_new_pseudonym_not_a_new_id(self):
        keys = add_new_pseudonym(
            self.keys.copy(), key_type="team", name="real team 1", old_id="T222"
        )
        pd.testing.assert_frame_equal(keys, self.keys)

    def test_add_new_pseudonym_new_player(self):
        res_keys = add_new_pseudonym(
            self.keys.copy(), key_type="player", name="real player 3", old_id=456
        )
        # can not know the pseudonym since the salt is randomly generated
        assert len(res_keys) == 4
        assert res_keys.isnull().sum().sum() == 0
        assert "real player 3" in res_keys["name"].values
        assert (
            res_keys.loc[res_keys["name"] == "real player 3", "original_id"].iloc[0]
            == 456
        )

    def test_get_player_mappings(self):
        home_players = pd.DataFrame(
            {
                "full_name": ["real player 1", "real player 3"],
                "id": [111, 333],
                "shirt_num": [12, 22],
            }
        )

        away_players = pd.DataFrame(
            {
                "full_name": ["real player 2", "real player 4"],
                "id": [3256, 6543],
                "shirt_num": [4, 12],
            }
        )

        (
            res_player_id_map,
            res_player_name_map,
            res_home_players_jersey_map,
            res_away_players_jersey_map,
            res_keys,
        ) = get_player_mappings(home_players, away_players, self.keys.copy())

        exp_player_id_map = {
            111: "P-64b32c78",
            333: "Unknown",
            3256: "P-24f9904f",
            6543: "Unkown",
        }
        exp_player_name_map = {
            "real player 1": "home_1",
            "real player 3": "home_2",
            "real player 2": "away_1",
            "real player 4": "away_2",
        }
        exp_home_players_jersey_map = {12: 1, 22: 2}
        exp_away_players_jersey_map = {4: 1, 12: 2}

        assert list(res_player_id_map.keys()) == list(exp_player_id_map.keys())
        assert res_player_id_map[111] == exp_player_id_map[111]
        assert res_player_id_map[3256] == exp_player_id_map[3256]
        assert res_player_name_map == exp_player_name_map
        assert res_home_players_jersey_map == exp_home_players_jersey_map
        assert res_away_players_jersey_map == exp_away_players_jersey_map
        assert len(res_keys) == 5
        assert res_keys.isnull().sum().sum() == 0
        assert len(res_keys["pseudonym"].unique()) == 5
        assert len(res_keys["original_id"].unique()) == 5

    def test_anonymise_players(self):
        input_keys = pd.concat(
            [
                self.keys.copy(),
                pd.DataFrame(
                    {
                        "name": "Jan Boskamp2",
                        "pseudonym": "P-0f693aca",
                        "salt": "506dd90aa126b157",
                        "original_id": 45849,
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
            sort=True,
        )
        input_game = self.game.copy()
        input_game.event_data["to_player_id"] = [np.nan] * 11 + [45849]
        input_game.event_data["to_player_name"] = [np.nan] * 11 + ["Jan Boskamp"]
        input_game._passes_df = pd.DataFrame({"player_id": [45849]})
        input_game._dribbles_df = pd.DataFrame({"player_id": [45849]})
        input_game._shots_df = pd.DataFrame({"player_id": [45849]})
        for event in input_game.all_events.values():
            if event.team_side == "home" and event.jersey == 2:
                event.jersey = 34
            elif event.team_side == "away" and event.jersey == 1:
                event.jersey = 17
        res_game, res_keys = anonymise_players(input_game, input_keys)

        # test the update of the keys df
        assert "Jan Boskamp" not in res_keys["name"].values
        assert len(res_keys) == 7
        assert res_keys.isnull().sum().sum() == 0
        assert len(res_keys["pseudonym"].unique()) == 7
        assert len(res_keys["original_id"].unique()) == 7
        assert res_keys["pseudonym"].str.startswith("P-").sum() == 6

        # test the changes in the game event data
        assert res_game.event_data["player_id"].str.startswith("P-").all()
        name_mask = (
            res_game.event_data["player_name"].str.startswith("home")
            | res_game.event_data["player_name"].str.startswith("away")
            | res_game.event_data["player_name"].isnull()
        )
        assert name_mask.all()
        assert res_game.event_data.iloc[5]["player_id"] == "P-0f693aca"
        assert res_game.event_data.iloc[7]["player_name"] == "home_2"
        assert res_game.event_data.loc[11, "to_player_id"] == "P-0f693aca"
        assert res_game.event_data.loc[11, "to_player_name"] == "home_2"

        # test changes in the game databallpy events
        for event in (
            list(res_game.shot_events.values())
            + list(res_game.dribble_events.values())
            + list(res_game.pass_events.values())
        ):
            assert event.player_id[:2] == "P-"

        # test changes in _events df
        assert res_game._passes_df.loc[0, "player_id"] == "P-0f693aca"
        assert res_game._dribbles_df.loc[0, "player_id"] == "P-0f693aca"
        assert res_game._shots_df.loc[0, "player_id"] == "P-0f693aca"

        # test changes in tracking data column names
        for col in ["home_34_x", "away_17_y"]:
            assert col not in res_game.tracking_data.columns

        for event in res_game.all_events.values():
            assert event.jersey <= 2

    def test_rename_tracking_data_columns(self):
        jersey_map = {34: 1, 1: 2}
        res_td = rename_tracking_data_columns(
            self.game.tracking_data.copy(), jersey_map, "home"
        )
        for exp_col in ["home_1_x", "home_1_y", "home_2_x", "home_2_y"]:
            assert exp_col in res_td.columns

        np.testing.assert_allclose(
            res_td["home_1_x"].values, self.game.tracking_data["home_34_x"].values
        )
        np.testing.assert_allclose(
            res_td["home_2_y"].values, self.game.tracking_data["home_1_y"].values
        )

    def test_rename_tracking_data_columns_error(self):
        jersey_map = {35: 1, 1: 2}
        with self.assertRaises(ValueError):
            rename_tracking_data_columns(
                self.game.tracking_data.copy(), jersey_map, "home"
            )

    def test_get_team_mappings(self):
        (res_team_map_id, res_team_name_map, res_keys) = get_team_mappings(
            "real team 1", "new team 2", "T222", 12314, self.keys.copy()
        )

        assert list(res_team_map_id.keys()) == ["T222", 12314]
        assert res_team_map_id["T222"] == "T-b4ac4483"
        assert list(res_team_name_map.keys()) == ["real team 1", "new team 2"]
        assert res_team_name_map["real team 1"] == "T-b4ac4483"
        assert len(res_keys) == 4
        assert res_keys.loc[3, "pseudonym"][:2] == "T-"
        assert res_keys.loc[3, "original_id"] == 12314
        assert res_keys.loc[3, "name"] == "new team 2"

    def test_anonymise_teams(self):
        input_game = self.game.copy()
        input_game._passes_df = pd.DataFrame({"team_id": [3]})
        input_game._dribbles_df = pd.DataFrame({"team_id": [194]})
        input_game._shots_df = pd.DataFrame({"team_id": [194]})
        res_game, res_keys = anonymise_teams(input_game, self.keys.copy())

        for val in [
            res_game.home_team_id,
            res_game.away_team_id,
            res_game.home_team_id,
            res_game.away_team_id,
        ]:
            assert val[:2] == "T-"

        assert res_game.home_team_id == res_game.home_team_name
        assert res_game.away_team_id == res_game.away_team_name

        for event in (
            list(res_game.shot_events.values())
            + list(res_game.pass_events.values())
            + list(res_game.dribble_events.values())
        ):
            event.team_id in [res_game.home_team_id, res_game.away_team_id]

        assert res_game.passes_df.loc[0, "team_id"] == res_game.home_team_id
        assert res_game.dribbles_df.loc[0, "team_id"] == res_game.away_team_id
        assert res_game.shots_df.loc[0, "team_id"] == res_game.away_team_id

        assert len(res_keys) == 5

    def test_anonymise_datetime(self):
        input_game = self.game.copy()
        input_game.event_data["tracking_frame"] = [MISSING_INT] * 11 + [1509996]
        input_game._is_synchronised = True
        dt_diff = (
            input_game.tracking_data["datetime"].iloc[0]
            - input_game.event_data["datetime"].iloc[0]
        )
        input_game.event_data["datetime"] += dt_diff + pd.to_timedelta(5, unit="seconds")
        ed_dt_diff = (
            input_game.event_data["datetime"].iloc[0]
            - input_game.periods["start_datetime_ed"].iloc[0]
        )
        input_game.periods["start_datetime_ed"] += ed_dt_diff
        input_game.periods["end_datetime_ed"] += ed_dt_diff
        for event in (
            list(input_game.shot_events.values())
            + list(input_game.pass_events.values())
            + list(input_game.dribble_events.values())
        ):
            event.datetime = input_game.event_data["datetime"].iloc[0]

        dt_event = input_game.event_data["datetime"].iloc[0]
        input_game._passes_df = pd.DataFrame({"datetime": [dt_event]})
        input_game._dribbles_df = pd.DataFrame({"datetime": [dt_event]})
        input_game._shots_df = pd.DataFrame({"datetime": [dt_event]})

        base_time = pd.to_datetime("2020-10-10 12:00:00.000", utc=True)
        res_game = anonymise_datetime(input_game.copy(), base_time=base_time)

        assert res_game.tracking_data["datetime"].iloc[0] == base_time
        assert (
            res_game.tracking_data["datetime"].iloc[0]
            - res_game.tracking_data["datetime"].iloc[-1]
        ) == (
            input_game.tracking_data["datetime"].iloc[0]
            - input_game.tracking_data["datetime"].iloc[-1]
        )
        assert res_game.periods["start_datetime_td"].iloc[0] == base_time
        np.testing.assert_almost_equal(
            res_game.tracking_data["frame"],
            np.arange(len(res_game.tracking_data)) + 1,
        )
        assert res_game.periods["start_frame"].iloc[0] == 1

        np.testing.assert_almost_equal(
            res_game.away_players["start_frame"].values, [1, 1]
        )
        np.testing.assert_almost_equal(res_game.home_players["end_frame"].values, [5, 3])

        assert res_game.event_data["datetime"].iloc[0] == base_time + pd.to_timedelta(
            5, unit="seconds"
        )
        assert (
            res_game.event_data["datetime"].iloc[0]
            - res_game.event_data["datetime"].iloc[-1]
        ) == (
            input_game.event_data["datetime"].iloc[0]
            - input_game.event_data["datetime"].iloc[-1]
        )
        assert res_game.event_data.loc[11, "tracking_frame"] == 4
        assert res_game.periods["start_datetime_ed"].iloc[
            0
        ] == base_time + pd.to_timedelta(5, unit="seconds")

        for event in (
            list(res_game.shot_events.values())
            + list(res_game.pass_events.values())
            + list(res_game.dribble_events.values())
        ):
            assert event.datetime == base_time + pd.to_timedelta(5, unit="seconds")

        assert res_game.dribbles_df["datetime"].iloc[0] == base_time + pd.to_timedelta(
            5, unit="seconds"
        )
        assert res_game.shots_df["datetime"].iloc[0] == base_time + pd.to_timedelta(
            5, unit="seconds"
        )
        assert res_game.passes_df["datetime"].iloc[0] == base_time + pd.to_timedelta(
            5, unit="seconds"
        )

    def test_anonymise_datetime_no_tracking_data(self):
        input_game = self.game.copy()
        input_game._is_synchronised = False
        input_game.tracking_data = pd.DataFrame()
        input_game.periods.drop(
            columns=[
                "start_frame",
                "end_frame",
                "start_datetime_td",
                "end_datetime_td",
            ],
            inplace=True,
        )
        input_game.home_players.loc[:, ["start_frame", "end_frame"]] = MISSING_INT
        input_game.away_players.loc[:, ["start_frame", "end_frame"]] = MISSING_INT

        input_game.periods.loc[0, "start_datetime_ed"] = input_game.event_data[
            "datetime"
        ].iloc[0] - pd.Timedelta(15, unit="seconds")

        base_time = pd.to_datetime("2020-10-11 12:00:00.000", utc=True)
        ed_dt = input_game.event_data["datetime"].iloc[0]
        for event in (
            list(input_game.shot_events.values())
            + list(input_game.pass_events.values())
            + list(input_game.dribble_events.values())
        ):
            event.datetime = ed_dt + pd.Timedelta(15, unit="seconds")

        input_game._passes_df = pd.DataFrame(
            {"datetime": [ed_dt + pd.Timedelta(15, unit="seconds")]}
        )
        input_game._dribbles_df = pd.DataFrame(
            {"datetime": [ed_dt + pd.Timedelta(15, unit="seconds")]}
        )
        input_game._shots_df = pd.DataFrame(
            {"datetime": [ed_dt + pd.Timedelta(15, unit="seconds")]}
        )

        res_game = anonymise_datetime(input_game.copy(), base_time=base_time)

        assert res_game.event_data["datetime"].iloc[0] == base_time
        assert (
            res_game.event_data["datetime"].iloc[0]
            - res_game.event_data["datetime"].iloc[-1]
        ) == (
            input_game.event_data["datetime"].iloc[0]
            - input_game.event_data["datetime"].iloc[-1]
        )
        assert res_game.periods["start_datetime_ed"].iloc[0] == base_time - pd.Timedelta(
            15, unit="seconds"
        )

        for event in (
            list(res_game.shot_events.values())
            + list(res_game.pass_events.values())
            + list(res_game.dribble_events.values())
        ):
            assert event.datetime == base_time + pd.to_timedelta(15, unit="seconds")

        assert res_game.dribbles_df["datetime"].iloc[0] == base_time + pd.to_timedelta(
            15, unit="seconds"
        )
        assert res_game.shots_df["datetime"].iloc[0] == base_time + pd.to_timedelta(
            15, unit="seconds"
        )
        assert res_game.passes_df["datetime"].iloc[0] == base_time + pd.to_timedelta(
            15, unit="seconds"
        )
