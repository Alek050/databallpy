import datetime as dt
import unittest

import numpy as np
import pandas as pd

from databallpy.features.angle import get_smallest_angle
from databallpy.schemas import EventData
from databallpy.utils.get_game import get_game
from databallpy.utils.synchronise_tracking_and_event_data import (
    _create_sim_mat,
    _get_cost_function,
    _needleman_wunsch,
    _validate_cost,
    _validate_sigmoid_kwargs,
    align_event_data_datetime,
    base_arrival_cost_function,
    base_general_cost_ball_event,
    base_pass_cost_function,
    base_shot_cost_function,
    combine_cost_functions,
    create_smart_batches,
    get_ball_acceleration_cost,
    get_ball_deceleration_cost,
    get_ball_goal_angle_cost,
    get_distance_ball_event_cost,
    get_distance_ball_player_cost,
    get_player_ball_distance_decrease_cost,
    get_player_ball_distance_increase_cost,
    get_time_difference_cost,
    pre_compute_synchronisation_variables,
    synchronise_tracking_and_event_data,
)
from databallpy.utils.utils import MISSING_INT, sigmoid
from tests.expected_outcomes import (
    RES_SIM_MAT,
    RES_SIM_MAT_MISSING_PLAYER,
    RES_SIM_MAT_NO_PLAYER,
)
from tests.pandas_testing import assert_frame_equal


class TestSynchroniseTrackingAndEventData(unittest.TestCase):
    def setUp(self) -> None:
        self.game_to_sync = get_game(
            tracking_data_loc="tests/test_data/sync/tracab_td_sync_test.dat",
            tracking_metadata_loc="tests/test_data/sync/tracab_metadata_sync_test.xml",
            tracking_data_provider="tracab",
            event_data_loc="tests/test_data/sync/opta_events_sync_test.xml",
            event_metadata_loc="tests/test_data/sync/opta_metadata_sync_test.xml",
            event_data_provider="opta",
            check_quality=False,
        )
        self.game_to_sync.allow_synchronise_tracking_and_event_data = True
        self.game_to_sync._tracking_timestamp_is_precise = False
        self.game_with_carry = self.game_to_sync.copy()

        # the expected outcomes below are based on the events without the carry
        # that get_game adds between the two passes
        event_data = self.game_to_sync.event_data
        is_carry = event_data["original_event"] == "carry"
        self.assertEqual(is_carry.sum(), 1)
        self.game_to_sync.event_data = EventData(
            event_data[~is_carry].reset_index(drop=True), provider=event_data.provider
        )
        self.game_to_sync.event_data["event_id"] = range(len(event_data) - 1)

        self.game_to_sync.tracking_data["ball_acceleration"] = np.array(
            [180.0, 180.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150.0, 0]
        )
        self.game_to_sync.tracking_data["ball_velocity"] = np.array(
            [
                1.0,
                80.0,
                81.0,
                82.0,
                83.0,
                84.0,
                85.0,
                86.0,
                87.0,
                88.0,
                89.0,
                240.0,
                239.0,
            ]
        )
        self.game_to_sync.event_data.loc[8, "databallpy_event"] = "shot"
        self.game_to_sync.event_data.loc[8, "action_type"] = "shot"
        self.expected_action_types = [
            None,
            "pass",
            None,
            "pass",
            None,
            None,
            None,
            None,
            None,
            "take_on",
            None,
            "shot",
            None,
        ]
        self.event_3_team_side = "home"
        self.event_3_jersey = 2

    @staticmethod
    def _add_expected_tracking_start(expected_event_data, expected_tracking_data):
        # the ball location in the synced frame
        synced = expected_event_data["tracking_frame"] != MISSING_INT
        frames = expected_event_data.loc[synced, "tracking_frame"]
        for col in ["x", "y"]:
            expected_event_data[f"tracking_start_{col}"] = np.nan
            expected_event_data.loc[synced, f"tracking_start_{col}"] = (
                expected_tracking_data.loc[frames, f"ball_{col}"].values
            )

    def test_synchronise_tracking_and_event_data_normal_condition(self):
        expected_event_data = self.game_to_sync.event_data.copy()
        expected_tracking_data = self.game_to_sync.tracking_data.copy()
        expected_tracking_data["period_id"] = [1] * 13
        expected_tracking_data["datetime"] = [
            pd.to_datetime("2023-01-22 16:46:39.720000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:39.760000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:39.800000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:39.840000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:39.880000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:39.920000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:39.960000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:40+01:00").tz_convert("Europe/Amsterdam"),
            pd.to_datetime("2023-01-22 16:46:40.040000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:40.080000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:40.120000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:40.160000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:40.200000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
        ]

        expected_tracking_data["databallpy_event"] = [
            None,
            "pass",
            None,
            "pass",
            None,
            None,
            None,
            None,
            None,
            "dribble",
            None,
            "shot",
            None,
        ]
        expected_tracking_data["action_type"] = self.expected_action_types
        expected_tracking_data["event_id"] = [
            MISSING_INT,
            3,
            MISSING_INT,
            4,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            7,
            MISSING_INT,
            8,
            MISSING_INT,
        ]
        expected_tracking_data["sync_certainty"] = [
            np.nan,
            0.579847,
            np.nan,
            0.634676,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.391775,
            np.nan,
            0.503890,
            np.nan,
        ]

        expected_event_data.loc[:, "tracking_frame"] = [
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            1,
            3,
            MISSING_INT,
            MISSING_INT,
            9,
            11,
        ]
        expected_event_data.loc[:, "sync_certainty"] = [
            np.nan,
            np.nan,
            np.nan,
            0.579847,
            0.634676,
            np.nan,
            np.nan,
            0.391775,
            0.503890,
        ]
        self._add_expected_tracking_start(expected_event_data, expected_tracking_data)
        game_to_sync = self.game_to_sync.copy()

        game_to_sync.synchronise_tracking_and_event_data(n_batches=2, offset=0)
        assert_frame_equal(game_to_sync.tracking_data, expected_tracking_data)

        assert_frame_equal(game_to_sync.event_data, expected_event_data)

        game_to_sync = self.game_to_sync.copy()
        game_to_sync.tracking_data.loc[:, "ball_status"] = "dead"
        with self.assertRaises(IndexError):
            game_to_sync.synchronise_tracking_and_event_data(n_batches=2, offset=0)

    def test_synchronise_tracking_and_event_data_action_types(self):
        game_to_sync = self.game_to_sync.copy()
        game_to_sync.event_data.loc[4, "action_type"] = "clearance"
        game_to_sync.event_data.loc[7, "action_type"] = "foul"

        game_to_sync.synchronise_tracking_and_event_data(n_batches=2, offset=0)

        # all action types are synced, except for fouls and non actions
        tracking_frame = game_to_sync.event_data.loc[4, "tracking_frame"]
        self.assertNotEqual(tracking_frame, MISSING_INT)
        td = game_to_sync.tracking_data
        self.assertEqual(td.loc[tracking_frame, "action_type"], "clearance")
        self.assertIsNone(td.loc[tracking_frame, "databallpy_event"])
        self.assertEqual(game_to_sync.event_data.loc[7, "tracking_frame"], MISSING_INT)
        self.assertNotIn("foul", td["action_type"].values)

    def test_synchronise_tracking_and_event_data_carry(self):
        game_to_sync = self.game_with_carry.copy()
        game_to_sync.synchronise_tracking_and_event_data(n_batches=2, offset=0)

        event_data = game_to_sync.event_data
        carry = event_data[event_data["original_event"] == "carry"].iloc[0]
        self.assertEqual(carry["action_type"], "dribble")
        self.assertNotEqual(carry["tracking_frame"], MISSING_INT)
        td = game_to_sync.tracking_data
        self.assertEqual(td.loc[carry["tracking_frame"], "event_id"], carry["event_id"])
        self.assertEqual(td.loc[carry["tracking_frame"], "action_type"], "dribble")

    def test_synchonise_tracking_and_event_data_error(self):
        with self.assertRaises(KeyError):
            game = self.game_to_sync.copy()
            game.tracking_data.drop(columns=game.tracking_data.columns, inplace=True)
            synchronise_tracking_and_event_data(
                game.tracking_data,
                game.event_data,
                game.home_players,
                game.away_players,
                verbose=False,
            )

    def test_synchronise_tracking_and_event_data_non_aligned_timestamps(self):
        expected_event_data = self.game_to_sync.event_data.copy()
        expected_tracking_data = self.game_to_sync.tracking_data.copy()
        expected_tracking_data["period_id"] = [1] * 13
        expected_tracking_data["datetime"] = [
            pd.to_datetime("2023-01-22 16:46:39.720000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:39.760000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:39.800000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:39.840000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:39.880000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:39.920000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:39.960000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:40+01:00").tz_convert("Europe/Amsterdam"),
            pd.to_datetime("2023-01-22 16:46:40.040000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:40.080000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:40.120000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:40.160000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:40.200000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
        ]

        expected_tracking_data["databallpy_event"] = [
            None,
            "pass",
            None,
            "pass",
            None,
            None,
            None,
            None,
            None,
            "dribble",
            None,
            "shot",
            None,
        ]
        expected_tracking_data["action_type"] = self.expected_action_types
        expected_tracking_data["event_id"] = [
            MISSING_INT,
            3,
            MISSING_INT,
            4,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            7,
            MISSING_INT,
            8,
            MISSING_INT,
        ]
        expected_tracking_data["sync_certainty"] = [
            np.nan,
            0.579847,
            np.nan,
            0.634676,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.391775,
            np.nan,
            0.503890,
            np.nan,
        ]

        expected_event_data.loc[:, "tracking_frame"] = [
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            1,
            3,
            MISSING_INT,
            MISSING_INT,
            9,
            11,
        ]
        expected_event_data.loc[:, "sync_certainty"] = [
            np.nan,
            np.nan,
            np.nan,
            0.579847,
            0.634676,
            np.nan,
            np.nan,
            0.391775,
            0.503890,
        ]
        self._add_expected_tracking_start(expected_event_data, expected_tracking_data)
        expected_event_data["datetime"] += pd.to_timedelta(1, unit="hours")
        game_to_sync = self.game_to_sync.copy()
        game_to_sync._tracking_timestamp_is_precise = True
        game_to_sync._event_timestamp_is_precise = True
        game_to_sync.event_data["datetime"] += pd.to_timedelta(1, unit="hours")

        game_to_sync.synchronise_tracking_and_event_data(n_batches=2, offset=0)

        assert_frame_equal(game_to_sync.tracking_data, expected_tracking_data)

        assert_frame_equal(game_to_sync.event_data, expected_event_data)

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

    def test_needleman_wunsch_sim_mat_nan(self):
        sim_list = 30 * [np.nan]
        sim_mat = np.array(sim_list).reshape(10, 3)

        with self.assertRaises(ValueError):
            _needleman_wunsch(sim_mat)

    def test_create_sim_mat(self):
        expected_res = RES_SIM_MAT

        tracking_data = self.game_to_sync.tracking_data.copy()
        date = pd.to_datetime(
            str(self.game_to_sync.periods.iloc[0, 3])[:10]
        ).tz_localize("Europe/Amsterdam")
        tracking_data["datetime"] = [
            date
            + dt.timedelta(
                milliseconds=int(x / self.game_to_sync.tracking_data.frame_rate * 1000)
            )
            for x in tracking_data["frame"]
        ]
        tracking_data["ball_acceleration_sqrt"] = 5.0
        tracking_data["goal_angle_home_team"] = 0.0
        tracking_data["goal_angle_away_team"] = 0.0
        tracking_data.reset_index(inplace=True)
        event_data = self.game_to_sync.event_data.copy()
        event_data = event_data[
            event_data["event_type_id"].isin([1, 3, 7])
        ].reset_index()
        res = _create_sim_mat(
            tracking_batch=tracking_data,
            event_batch=event_data,
            home_players=self.game_to_sync.home_players,
            away_players=self.game_to_sync.away_players,
        )
        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

    def test_create_sim_mat_action_type_cost_functions(self):
        tracking_data = self.game_to_sync.tracking_data.copy().reset_index()
        event_data = self.game_to_sync.event_data.copy()
        event_data = event_data[
            event_data["event_type_id"].isin([1, 3, 7])
        ].reset_index()
        n_frames = len(tracking_data)
        # the cost function of the action type has priority over the databallpy event
        cost_functions = {
            "pass": lambda **kwargs: np.full(n_frames, 0.5),
            "take_on": lambda **kwargs: np.full(n_frames, 0.2),
            "dribble": lambda **kwargs: np.full(n_frames, 0.9),
            "shot": lambda **kwargs: np.full(n_frames, 0.0),
        }
        res = _create_sim_mat(
            tracking_batch=tracking_data,
            event_batch=event_data,
            home_players=self.game_to_sync.home_players,
            away_players=self.game_to_sync.away_players,
            cost_functions=cost_functions,
        )
        expected_res = np.repeat([[0.5, 0.5, 0.8, 1.0]], n_frames, axis=0)
        np.testing.assert_allclose(res, expected_res)

    def test_create_sim_mat_missing_player(self):
        expected_res = RES_SIM_MAT_MISSING_PLAYER

        tracking_data = self.game_to_sync.tracking_data.copy()
        date = pd.to_datetime(
            str(self.game_to_sync.periods.iloc[0, 3])[:10]
        ).tz_localize("Europe/Amsterdam")
        tracking_data["datetime"] = [
            date
            + dt.timedelta(
                milliseconds=int(x / self.game_to_sync.tracking_data.frame_rate * 1000)
            )
            for x in tracking_data["frame"]
        ]
        tracking_data.reset_index(inplace=True)
        tracking_data["away_1_x"] = [np.nan] * 13
        tracking_data["away_1_y"] = [np.nan] * 13
        tracking_data["ball_acceleration_sqrt"] = 5.0
        tracking_data["goal_angle_home_team"] = 0.0
        tracking_data["goal_angle_away_team"] = 0.0
        event_data = self.game_to_sync.event_data.copy()
        event_data = event_data[
            event_data["event_type_id"].isin([1, 3, 7])
        ].reset_index()
        res = _create_sim_mat(
            tracking_batch=tracking_data,
            event_batch=event_data,
            home_players=self.game_to_sync.home_players,
            away_players=self.game_to_sync.away_players,
        )
        np.testing.assert_allclose(res, expected_res, rtol=1e-05, atol=1e-05)

    def test_create_sim_mat_without_player(self):
        expected_res = RES_SIM_MAT_NO_PLAYER

        tracking_data = self.game_to_sync.tracking_data.copy()
        date = pd.to_datetime(
            str(self.game_to_sync.periods.iloc[0, 3])[:10]
        ).tz_localize("Europe/Amsterdam")
        tracking_data["datetime"] = [
            date
            + dt.timedelta(
                milliseconds=int(x / self.game_to_sync.tracking_data.frame_rate * 1000)
            )
            for x in tracking_data["frame"]
        ]
        tracking_data["ball_acceleration_sqrt"] = 5.0
        tracking_data["goal_angle_home_team"] = 0.0
        tracking_data["goal_angle_away_team"] = 0.0
        tracking_data.reset_index(inplace=True)
        event_data = self.game_to_sync.event_data.copy()
        event_data = event_data[
            event_data["event_type_id"].isin([1, 3, 7])
        ].reset_index()

        def _assert_sim_mats_equal(tracking_data, event_data):
            res = _create_sim_mat(
                tracking_batch=tracking_data,
                event_batch=event_data,
                home_players=self.game_to_sync.home_players,
                away_players=self.game_to_sync.away_players,
            )
            np.testing.assert_allclose(expected_res, res, rtol=1e-05)

        # Test with player_id = np.nan
        event_data.loc[2, "player_id"] = np.nan
        _assert_sim_mats_equal(tracking_data, event_data)

        # Test with player_id = MISSING_INT (-999)
        event_data.loc[2, "player_id"] = MISSING_INT
        _assert_sim_mats_equal(tracking_data, event_data)

    def test_pre_compute_synchronisation_variables(self):
        expected_td = self.game_to_sync.tracking_data.copy()
        expected_td["goal_angle_home_team"] = [
            0.19953378014112227,
            0.2249295894248988,
            np.nan,
            np.nan,
            2.916567100863839,
            0.19953378014112227,
            0.2249295894248988,
            0.22967695413522296,
            2.916567100863839,
            0.19953378014112227,
            0.2249295894248988,
            0.22967695413522296,
            np.nan,
        ]
        expected_td["goal_angle_away_team"] = [
            0.19953378014112227,
            0.2249295894248988,
            np.nan,
            np.nan,
            2.916567100863839,
            0.19953378014112227,
            0.2249295894248988,
            0.22967695413522296,
            2.916567100863839,
            0.19953378014112227,
            0.2249295894248988,
            0.22967695413522296,
            np.nan,
        ]
        expected_td["datetime"] = [
            pd.to_datetime("2023-01-22 16:46:39.720000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:39.760000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:39.800000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:39.840000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:39.880000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:39.920000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:39.960000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:40+01:00").tz_convert("Europe/Amsterdam"),
            pd.to_datetime("2023-01-22 16:46:40.040000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:40.080000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:40.120000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:40.160000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
            pd.to_datetime("2023-01-22 16:46:40.200000+01:00").tz_convert(
                "Europe/Amsterdam"
            ),
        ]

        res_tracking_data = pre_compute_synchronisation_variables(
            tracking_data=self.game_to_sync.tracking_data.copy(),
            frame_rate=25,
            pitch_dimensions=(106, 68),
        )
        assert_frame_equal(res_tracking_data, expected_td)

    def test_create_smart_batches(self):
        tracking_data = pd.DataFrame(
            {
                "datetime": [
                    "2023-01-22 16:00:00",
                    "2023-01-22 16:00:10",
                    "2023-01-22 16:00:20",
                    "2023-01-22 16:00:30",
                    "2023-01-22 16:00:40",
                    "2023-01-22 16:00:50",
                ],
                "ball_status": ["alive", "alive", "dead", "dead", "alive", "alive"],
                "period_id": [1, 1, 1, 1, 1, 2],
            }
        )

        tracking_data["datetime"] = pd.to_datetime(tracking_data["datetime"])

        expected_res = pd.to_datetime(["2023-01-22 16:00:25", "2023-01-22 17:00:50"])

        res = create_smart_batches(tracking_data)
        assert all(res == expected_res)

    def test_align_event_data_datetime(self):
        event_data = self.game_to_sync.event_data.copy()
        start_dt = event_data.loc[0, "datetime"]
        event_data["datetime"] = start_dt + pd.to_timedelta(1, unit="hours")
        tracking_data = self.game_to_sync.tracking_data.copy()
        tracking_data["datetime"] = [
            start_dt + pd.to_timedelta(x, unit="seconds") for x in range(13)
        ]
        tracking_data.loc[tracking_data.index[-1], "period_id"] = MISSING_INT

        expected_event_data = event_data.copy()
        expected_event_data.loc[1:, "datetime"] -= pd.to_timedelta(1, unit="hours")
        expected_event_data.loc[1:, "datetime"] += pd.to_timedelta(1, unit="seconds")

        # both tz aware
        res_event_data = align_event_data_datetime(
            event_data.copy(), tracking_data, offset=1.0
        )
        assert_frame_equal(res_event_data, expected_event_data)

        ed_no_tz = event_data.copy()
        ed_no_tz["datetime"] = ed_no_tz["datetime"].dt.tz_convert(None)
        td_no_tz = tracking_data.copy()
        td_no_tz["datetime"] = td_no_tz["datetime"].dt.tz_convert(None)

        # ed not tz aware
        res_event_data = align_event_data_datetime(
            ed_no_tz.copy(), tracking_data, offset=1.0
        )
        assert res_event_data["datetime"].dt.tz == tracking_data["datetime"].dt.tz
        assert_frame_equal(res_event_data, expected_event_data)

        # td not tz aware
        res_event_data = align_event_data_datetime(
            event_data.copy(), td_no_tz, offset=1.0
        )
        assert res_event_data["datetime"].dt.tz == td_no_tz["datetime"].dt.tz
        expected_event_data["datetime"] = expected_event_data["datetime"].dt.tz_convert(
            None
        )
        assert_frame_equal(res_event_data, expected_event_data)

    def test_get_time_difference_cost(self):
        tracking_data = self.game_to_sync.tracking_data.copy()
        event = self.game_to_sync.get_event(3)  # pass

        time_diff = (
            (tracking_data["datetime"] - event.datetime).dt.total_seconds().values
        )
        expected_res = sigmoid(np.abs(time_diff), e=5)
        res = get_time_difference_cost(tracking_data, event)
        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

        res2 = get_time_difference_cost(tracking_data, event, time_diff=time_diff, e=2)
        expected_res2 = sigmoid(np.abs(time_diff), e=2)
        np.testing.assert_allclose(res2, expected_res2, rtol=1e-05)

    def test_get_distance_ball_event_cost(self):
        tracking_data = self.game_to_sync.tracking_data.copy()
        event = self.game_to_sync.get_event(3)
        ball_event_distance = np.sqrt(
            (tracking_data["ball_x"] - event.start_x) ** 2
            + (tracking_data["ball_y"] - event.start_y) ** 2
        )

        res = get_distance_ball_event_cost(tracking_data, event)
        expected_res = sigmoid(ball_event_distance, d=5, e=6)

        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

        res2 = get_distance_ball_event_cost(
            tracking_data, event, ball_event_distance=ball_event_distance + 2, d=2, e=3
        )
        expected_res2 = sigmoid(ball_event_distance + 2, d=2, e=3)

        np.testing.assert_allclose(res2, expected_res2, rtol=1e-05)

    def test_get_distance_ball_player_cost(self):
        tracking_data = self.game_to_sync.tracking_data.copy()
        ball_player_distance = np.sqrt(
            (tracking_data["ball_x"] - tracking_data["home_2_x"]) ** 2
            + (tracking_data["ball_y"] - tracking_data["home_2_y"]) ** 2
        )

        res = get_distance_ball_player_cost(
            tracking_data, self.event_3_team_side, self.event_3_jersey
        )
        expected_res = sigmoid(ball_player_distance, d=5, e=2.5)

        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

        res2 = get_distance_ball_player_cost(
            tracking_data, self.event_3_team_side, self.event_3_jersey, d=2, e=3
        )
        expected_res2 = sigmoid(ball_player_distance + 2, d=2, e=3)

        np.testing.assert_allclose(res2, expected_res2, rtol=1e-05)

    def test_get_ball_acceleration_cost(self):
        tracking_data = self.game_to_sync.tracking_data.copy()
        ball_acceleration = tracking_data["ball_acceleration"]

        res = get_ball_acceleration_cost(tracking_data)
        expected_res = sigmoid(-ball_acceleration, d=0.2, e=-25.0)

        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

        res2 = get_ball_acceleration_cost(tracking_data, d=2, e=3)
        expected_res2 = sigmoid(ball_acceleration, d=2, e=3)

        np.testing.assert_allclose(res2, expected_res2, rtol=1e-05)

    def test_get_player_ball_distance_increase_cost(self):
        tracking_data = self.game_to_sync.tracking_data.copy()
        player_ball_diff = np.sqrt(
            (tracking_data["home_2_x"] - tracking_data["ball_x"]) ** 2
            + (tracking_data["home_2_y"] - tracking_data["ball_y"]) ** 2
        )

        res = get_player_ball_distance_increase_cost(
            tracking_data, self.event_3_team_side, self.event_3_jersey
        )
        expected_res = sigmoid(np.gradient(player_ball_diff), d=-8.0)

        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

        res2 = get_player_ball_distance_increase_cost(
            tracking_data, self.event_3_team_side, self.event_3_jersey, d=2
        )
        expected_res2 = sigmoid(np.gradient(player_ball_diff), d=2)

        np.testing.assert_allclose(res2, expected_res2, rtol=1e-05)

    def test_get_ball_deceleration_cost(self):
        tracking_data = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2023-01-01 00:00:00.0", "2023-01-01 00:00:00.5"]
                    + ["2023-01-01 00:00:01.0"]
                ),
                "ball_x": [0.0, 5.0, 6.0],
                "ball_y": [0.0, 0.0, 0.0],
                "ball_velocity": [10.0, 6.0, 2.0],
            }
        )
        # velocity drops 4 m/s every 0.5 s: deceleration of 8 m/s^2
        res = get_ball_deceleration_cost(tracking_data)
        expected_res = sigmoid(-np.array([8.0, 8.0, 8.0]), d=0.2, e=-25.0)
        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

        res2 = get_ball_deceleration_cost(tracking_data, d=2, e=3)
        expected_res2 = sigmoid(np.array([8.0, 8.0, 8.0]), d=2, e=3)
        np.testing.assert_allclose(res2, expected_res2, rtol=1e-05)

        # without ball_velocity, the velocity is derived from the ball location
        res3 = get_ball_deceleration_cost(tracking_data.drop(columns="ball_velocity"))
        velocity = np.gradient(np.array([0.0, 5.0, 6.0]), np.array([0.0, 0.5, 1.0]))
        deceleration = -np.gradient(velocity, np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(
            res3, sigmoid(-deceleration, d=0.2, e=-25.0), rtol=1e-05
        )

        res4 = get_ball_deceleration_cost(tracking_data.iloc[:1])
        self.assertTrue(np.isnan(res4).all())

    def test_get_player_ball_distance_decrease_cost(self):
        tracking_data = self.game_to_sync.tracking_data.copy()
        player_ball_diff = np.sqrt(
            (tracking_data["home_2_x"] - tracking_data["ball_x"]) ** 2
            + (tracking_data["home_2_y"] - tracking_data["ball_y"]) ** 2
        )

        res = get_player_ball_distance_decrease_cost(
            tracking_data, self.event_3_team_side, self.event_3_jersey
        )
        expected_res = sigmoid(np.gradient(player_ball_diff), d=8.0)
        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

        res2 = get_player_ball_distance_decrease_cost(
            tracking_data, self.event_3_team_side, self.event_3_jersey, d=2
        )
        expected_res2 = sigmoid(np.gradient(player_ball_diff), d=2)
        np.testing.assert_allclose(res2, expected_res2, rtol=1e-05)

    def test_base_arrival_cost_function(self):
        tracking_data = self.game_to_sync.tracking_data.copy()
        event = self.game_to_sync.get_event(3)

        res = base_arrival_cost_function(
            tracking_data, event, self.event_3_team_side, self.event_3_jersey
        )

        expected_res = combine_cost_functions(
            [
                get_time_difference_cost(tracking_data, event),
                get_distance_ball_event_cost(tracking_data, event),
                get_distance_ball_player_cost(
                    tracking_data, self.event_3_team_side, self.event_3_jersey
                ),
                get_ball_deceleration_cost(tracking_data),
                get_player_ball_distance_decrease_cost(
                    tracking_data, self.event_3_team_side, self.event_3_jersey
                ),
            ]
        )
        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

    def test_get_cost_function(self):
        def custom(**kwargs):
            pass

        def custom2(**kwargs):
            pass

        cases = [
            # default by action type
            ("cross", "pass", {}, base_pass_cost_function),
            ("clearance", None, {}, base_pass_cost_function),
            ("bad_touch", None, {}, base_pass_cost_function),
            ("shot_penalty", "shot", {}, base_shot_cost_function),
            ("dribble", "dribble", {}, base_arrival_cost_function),
            ("take_on", "dribble", {}, base_arrival_cost_function),
            ("interception", None, {}, base_arrival_cost_function),
            ("keeper_save", None, {}, base_arrival_cost_function),
            ("tackle", "tackle", {}, base_general_cost_ball_event),
            ("pass", "own_goal", {}, base_general_cost_ball_event),
            # default by databallpy event, without action types
            (None, "pass", {}, base_pass_cost_function),
            (None, "shot", {}, base_shot_cost_function),
            (None, "dribble", {}, base_arrival_cost_function),
            (None, "tackle", {}, base_general_cost_ball_event),
            # custom cost functions, action type over databallpy event
            ("cross", "pass", {"pass": custom}, custom),
            ("cross", "pass", {"pass": custom, "cross": custom2}, custom2),
            ("pass", "own_goal", {"own_goal": custom}, custom),
            (None, "dribble", {"dribble": custom}, custom),
        ]
        for action_type, databallpy_event, cost_functions, expected in cases:
            with self.subTest(action_type=action_type, event=databallpy_event):
                self.assertIs(
                    _get_cost_function(action_type, databallpy_event, cost_functions),
                    expected,
                )

    def test_get_ball_goal_angle_cost(self):
        tracking_data = pd.DataFrame(
            {
                "ball_x": [-10, -25, -40],
                "ball_y": [-15, -10, -5],
            }
        )
        goal_loc = [-self.game_to_sync.pitch_dimensions[0] / 2, 0]

        ball_moving_vectors = np.array(
            [
                [-15, -5],
                [-15, -5],
            ]
        )

        ball_goal_vectors = np.array(
            [
                [goal_loc[0] - -10, -15],
                [goal_loc[0] - -25, -10],
            ]
        )
        goal_angle = get_smallest_angle(
            ball_moving_vectors,
            ball_goal_vectors,
            angle_format="radian",
        )

        goal_angle = np.concatenate((goal_angle, [goal_angle[-1]]))
        res = get_ball_goal_angle_cost(
            tracking_data, "away", self.game_to_sync.pitch_dimensions[0]
        )
        expected_res = sigmoid(goal_angle, d=6, e=0.2 * np.pi)

        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

        tracking_data["goal_angle_home_team"] = goal_angle
        tracking_data["goal_angle_away_team"] = goal_angle + np.pi

        res2 = get_ball_goal_angle_cost(
            tracking_data, "away", self.game_to_sync.pitch_dimensions[0], d=2, e=3
        )
        expected_res2 = sigmoid(goal_angle + np.pi, d=2, e=3)

        np.testing.assert_allclose(res2, expected_res2, rtol=1e-05)

    def test_base_general_cost_ball_event(self):
        tracking_data = self.game_to_sync.tracking_data.copy()
        event = self.game_to_sync.get_event(3)

        res = base_general_cost_ball_event(
            tracking_data, event, self.event_3_team_side, self.event_3_jersey
        )

        expected_res = combine_cost_functions(
            [
                get_time_difference_cost(tracking_data, event),
                get_distance_ball_event_cost(tracking_data, event),
                get_distance_ball_player_cost(
                    tracking_data, self.event_3_team_side, self.event_3_jersey
                ),
            ]
        )
        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

    def test_base_pass_cost_function(self):
        tracking_data = self.game_to_sync.tracking_data.copy()
        event = self.game_to_sync.get_event(3)

        res = base_pass_cost_function(
            tracking_data, event, self.event_3_team_side, self.event_3_jersey
        )

        expected_res = combine_cost_functions(
            [
                get_time_difference_cost(tracking_data, event),
                get_distance_ball_event_cost(tracking_data, event),
                get_distance_ball_player_cost(
                    tracking_data, self.event_3_team_side, self.event_3_jersey
                ),
                get_ball_acceleration_cost(tracking_data),
                get_player_ball_distance_increase_cost(
                    tracking_data, self.event_3_team_side, self.event_3_jersey
                ),
            ]
        )
        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

    def test_base_shot_cost_function(self):
        tracking_data = self.game_to_sync.tracking_data.copy()
        event = self.game_to_sync.get_event(3)

        res = base_shot_cost_function(
            tracking_data, event, self.event_3_team_side, self.event_3_jersey
        )

        expected_res = combine_cost_functions(
            [
                get_time_difference_cost(tracking_data, event),
                get_distance_ball_event_cost(tracking_data, event),
                get_distance_ball_player_cost(
                    tracking_data, self.event_3_team_side, self.event_3_jersey
                ),
                get_ball_acceleration_cost(tracking_data),
                get_player_ball_distance_increase_cost(
                    tracking_data, self.event_3_team_side, self.event_3_jersey
                ),
                get_ball_goal_angle_cost(tracking_data, self.event_3_team_side, 105),
            ]
        )

        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

    def test_combine_cost_functions(self):
        cost_functions = [
            [0.2, 0.3, np.nan, 0.4],
            [0.1, np.nan, np.nan, 0.4],
            [0.1, 0.2, np.nan, 0.4],
        ]

        res = combine_cost_functions(cost_functions)
        expected_res = np.array([0.4 / 3, 0.25, 1.0, 0.4])
        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

    def test_validate_cost(self):
        cost = np.array([0.1, 0.2, 0.3, 0.4])
        _validate_cost(cost, 4)

        cost = [0.1, 0.2, 0.3, 0.4]
        with self.assertRaises(TypeError):
            _validate_cost(cost, 4)

        cost = np.array([[0.1, 0.2, 0.3, 0.4]])
        with self.assertRaises(ValueError):
            _validate_cost(cost, 4)

        cost = np.array([0.1, 0.2, 0.3, 0.4])
        with self.assertRaises(ValueError):
            _validate_cost(cost, 3)

        cost = np.array([0.1, 0.2, 0.3, np.nan])
        with self.assertRaises(ValueError):
            _validate_cost(cost, 4)

        cost = np.array([-0.1, 0.2, 0.3, 0.4])
        with self.assertRaises(ValueError):
            _validate_cost(cost, 4)

        cost = np.array([0.1, 0.2, 0.3, 1.4])
        with self.assertRaises(ValueError):
            _validate_cost(cost, 4)

    def test_validate_sigmoid_kwargs(self):
        kwargs = {"d": 0.1, "e": 0.2}
        _validate_sigmoid_kwargs(kwargs)

        kwargs = {"d": 0.1}
        _validate_sigmoid_kwargs(kwargs)

        kwargs = {"e": 0.1}
        _validate_sigmoid_kwargs(kwargs)

        kwargs = {}
        _validate_sigmoid_kwargs(kwargs)

        kwargs = {"a": 3.0, "b": -4.0, "wrong": 2.0}
        with self.assertRaises(ValueError):
            _validate_sigmoid_kwargs(kwargs)

        kwargs = {"d": 0.1, "e": "-0.2"}
        with self.assertRaises(ValueError):
            _validate_sigmoid_kwargs(kwargs)
