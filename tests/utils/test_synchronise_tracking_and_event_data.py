import datetime as dt
import unittest

import numpy as np
import pandas as pd

from databallpy.features.angle import get_smallest_angle
from databallpy.utils.get_match import get_match
from databallpy.utils.synchronise_tracking_and_event_data import (
    _create_sim_mat,
    _needleman_wunsch,
    _validate_cost,
    _validate_sigmoid_kwargs,
    align_event_data_datetime,
    base_general_cost_ball_event,
    base_pass_cost_function,
    base_shot_cost_function,
    combine_cost_functions,
    create_smart_batches,
    get_ball_acceleration_cost,
    get_ball_goal_angle_cost,
    get_distance_ball_event_cost,
    get_distance_ball_player_cost,
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


class TestSynchroniseTrackingAndEventData(unittest.TestCase):
    def setUp(self) -> None:
        self.match_to_sync = get_match(
            tracking_data_loc="tests/test_data/sync/tracab_td_sync_test.dat",
            tracking_metadata_loc="tests/test_data/sync/tracab_metadata_sync_test.xml",
            tracking_data_provider="tracab",
            event_data_loc="tests/test_data/sync/opta_events_sync_test.xml",
            event_metadata_loc="tests/test_data/sync/opta_metadata_sync_test.xml",
            event_data_provider="opta",
            check_quality=False,
        )
        self.match_to_sync.allow_synchronise_tracking_and_event_data = True
        self.match_to_sync._tracking_timestamp_is_precise = False

        self.match_to_sync.tracking_data["ball_acceleration"] = np.array(
            [180.0, 180.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150.0, 0]
        )
        self.match_to_sync.tracking_data["ball_velocity"] = np.array(
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
        self.match_to_sync.event_data.loc[8, "databallpy_event"] = "shot"

    def test_synchronise_tracking_and_event_data_normal_condition(self):
        expected_event_data = self.match_to_sync.event_data.copy()
        expected_tracking_data = self.match_to_sync.tracking_data.copy()
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
            "dribble",
            None,
            None,
            "shot",
            None,
        ]
        expected_tracking_data["event_id"] = [
            MISSING_INT,
            2499594225,
            MISSING_INT,
            2499594243,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            2499594285,
            MISSING_INT,
            MISSING_INT,
            2499594291,
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
            0.331058,
            np.nan,
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
            8,
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
            0.331058,
            0.503890,
        ]
        match_to_sync = self.match_to_sync.copy()

        match_to_sync.synchronise_tracking_and_event_data(n_batches=2, offset=0)
        pd.testing.assert_frame_equal(
            match_to_sync.tracking_data, expected_tracking_data
        )

        pd.testing.assert_frame_equal(match_to_sync.event_data, expected_event_data)

        match_to_sync = self.match_to_sync.copy()
        match_to_sync.tracking_data.loc[:, "ball_status"] = "dead"
        with self.assertRaises(IndexError):
            match_to_sync.synchronise_tracking_and_event_data(n_batches=2, offset=0)

    def test_synchonise_tracking_and_event_data_error(self):
        with self.assertRaises(KeyError):
            match = self.match_to_sync.copy()
            match.tracking_data.drop(columns=match.tracking_data.columns, inplace=True)
            synchronise_tracking_and_event_data(
                match.tracking_data, match.event_data, match.all_events, verbose=False
            )

    def test_synchronise_tracking_and_event_data_non_aligned_timestamps(self):
        expected_event_data = self.match_to_sync.event_data.copy()
        expected_tracking_data = self.match_to_sync.tracking_data.copy()
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
            "dribble",
            None,
            None,
            "shot",
            None,
        ]
        expected_tracking_data["event_id"] = [
            MISSING_INT,
            2499594225,
            MISSING_INT,
            2499594243,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            2499594285,
            MISSING_INT,
            MISSING_INT,
            2499594291,
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
            0.331058,
            np.nan,
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
            8,
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
            0.331058,
            0.503890,
        ]
        expected_event_data["datetime"] += pd.to_timedelta(1, unit="hours")
        match_to_sync = self.match_to_sync.copy()
        match_to_sync._tracking_timestamp_is_precise = True
        match_to_sync._event_timestamp_is_precise = True
        match_to_sync.event_data["datetime"] += pd.to_timedelta(1, unit="hours")

        match_to_sync.synchronise_tracking_and_event_data(n_batches=2, offset=0)

        pd.testing.assert_frame_equal(
            match_to_sync.tracking_data, expected_tracking_data
        )

        pd.testing.assert_frame_equal(match_to_sync.event_data, expected_event_data)

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

        tracking_data = self.match_to_sync.tracking_data.copy()
        date = pd.to_datetime(
            str(self.match_to_sync.periods.iloc[0, 3])[:10]
        ).tz_localize("Europe/Amsterdam")
        tracking_data["datetime"] = [
            date
            + dt.timedelta(milliseconds=int(x / self.match_to_sync.frame_rate * 1000))
            for x in tracking_data["frame"]
        ]
        tracking_data["ball_acceleration_sqrt"] = 5.0
        tracking_data["goal_angle_home_team"] = 0.0
        tracking_data["goal_angle_away_team"] = 0.0
        tracking_data.reset_index(inplace=True)
        event_data = self.match_to_sync.event_data.copy()
        event_data = event_data[event_data["type_id"].isin([1, 3, 7])].reset_index()
        res = _create_sim_mat(
            tracking_batch=tracking_data,
            event_batch=event_data,
            all_events=self.match_to_sync.all_events,
        )
        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

    def test_create_sim_mat_missing_player(self):
        expected_res = RES_SIM_MAT_MISSING_PLAYER

        tracking_data = self.match_to_sync.tracking_data.copy()
        date = pd.to_datetime(
            str(self.match_to_sync.periods.iloc[0, 3])[:10]
        ).tz_localize("Europe/Amsterdam")
        tracking_data["datetime"] = [
            date
            + dt.timedelta(milliseconds=int(x / self.match_to_sync.frame_rate * 1000))
            for x in tracking_data["frame"]
        ]
        tracking_data.reset_index(inplace=True)
        tracking_data["away_1_x"] = [np.nan] * 13
        tracking_data["away_1_y"] = [np.nan] * 13
        tracking_data["ball_acceleration_sqrt"] = 5.0
        tracking_data["goal_angle_home_team"] = 0.0
        tracking_data["goal_angle_away_team"] = 0.0
        event_data = self.match_to_sync.event_data.copy()
        event_data = event_data[event_data["type_id"].isin([1, 3, 7])].reset_index()
        res = _create_sim_mat(
            tracking_batch=tracking_data,
            event_batch=event_data,
            all_events=self.match_to_sync.all_events,
        )
        np.testing.assert_allclose(res, expected_res, rtol=1e-05, atol=1e-05)

    def test_create_sim_mat_without_player(self):
        expected_res = RES_SIM_MAT_NO_PLAYER

        tracking_data = self.match_to_sync.tracking_data.copy()
        date = pd.to_datetime(
            str(self.match_to_sync.periods.iloc[0, 3])[:10]
        ).tz_localize("Europe/Amsterdam")
        tracking_data["datetime"] = [
            date
            + dt.timedelta(milliseconds=int(x / self.match_to_sync.frame_rate * 1000))
            for x in tracking_data["frame"]
        ]
        tracking_data["ball_acceleration_sqrt"] = 5.0
        tracking_data["goal_angle_home_team"] = 0.0
        tracking_data["goal_angle_away_team"] = 0.0
        tracking_data.reset_index(inplace=True)
        event_data = self.match_to_sync.event_data.copy()
        event_data = event_data[event_data["type_id"].isin([1, 3, 7])].reset_index()

        def _assert_sim_mats_equal(tracking_data, event_data):
            res = _create_sim_mat(
                tracking_batch=tracking_data,
                event_batch=event_data,
                all_events=self.match_to_sync.all_events,
            )
            np.testing.assert_allclose(expected_res, res, rtol=1e-05)

        # Test with player_id = np.nan
        event_data.loc[2, "player_id"] = np.nan
        _assert_sim_mats_equal(tracking_data, event_data)

        # Test with player_id = MISSING_INT (-999)
        event_data.loc[2, "player_id"] = MISSING_INT
        _assert_sim_mats_equal(tracking_data, event_data)

    def test_pre_compute_synchronisation_variables(self):
        expected_td = self.match_to_sync.tracking_data.copy()
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
            tracking_data=self.match_to_sync.tracking_data.copy(),
            frame_rate=25,
            pitch_dimensions=(106, 68),
        )
        pd.testing.assert_frame_equal(res_tracking_data, expected_td)

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

        expected_res = pd.to_datetime(["2023-01-22 16:00:25", "2023-01-22 16:00:53"])

        res = create_smart_batches(tracking_data)
        assert all(res == expected_res)

    def test_align_event_data_datetime(self):
        event_data = self.match_to_sync.event_data.copy()
        start_dt = event_data.loc[0, "datetime"]
        event_data["datetime"] = start_dt + pd.to_timedelta(1, unit="hours")
        tracking_data = self.match_to_sync.tracking_data.copy()
        tracking_data["datetime"] = [
            start_dt + pd.to_timedelta(x, unit="seconds") for x in range(13)
        ]
        tracking_data.loc[tracking_data.index[-1], "period_id"] = MISSING_INT

        expected_event_data = event_data.copy()
        expected_event_data.loc[1:, "datetime"] -= pd.to_timedelta(1, unit="hours")
        expected_event_data.loc[1:, "datetime"] += pd.to_timedelta(1, unit="seconds")

        res_event_data = align_event_data_datetime(event_data, tracking_data, offset=1.0)
        pd.testing.assert_frame_equal(res_event_data, expected_event_data)

    def test_get_time_difference_cost(self):
        tracking_data = self.match_to_sync.tracking_data.copy()
        event = self.match_to_sync.get_event(2499594225)  # pass

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
        tracking_data = self.match_to_sync.tracking_data.copy()
        event = self.match_to_sync.get_event(2499594225)
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
        tracking_data = self.match_to_sync.tracking_data.copy()
        event = self.match_to_sync.get_event(2499594225)
        event.jersey = 2
        event.team_side = "home"
        ball_player_distance = np.sqrt(
            (tracking_data["ball_x"] - tracking_data["home_2_x"]) ** 2
            + (tracking_data["ball_y"] - tracking_data["home_2_y"]) ** 2
        )

        res = get_distance_ball_player_cost(tracking_data, event)
        expected_res = sigmoid(ball_player_distance, d=5, e=2.5)

        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

        res2 = get_distance_ball_player_cost(tracking_data, event, d=2, e=3)
        expected_res2 = sigmoid(ball_player_distance + 2, d=2, e=3)

        np.testing.assert_allclose(res2, expected_res2, rtol=1e-05)

    def test_get_ball_acceleration_cost(self):
        tracking_data = self.match_to_sync.tracking_data.copy()
        event = self.match_to_sync.get_event(2499594225)
        ball_acceleration = tracking_data["ball_acceleration"]

        res = get_ball_acceleration_cost(tracking_data, event)
        expected_res = sigmoid(-ball_acceleration, d=0.2, e=-25.0)

        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

        res2 = get_ball_acceleration_cost(tracking_data, event, d=2, e=3)
        expected_res2 = sigmoid(ball_acceleration, d=2, e=3)

        np.testing.assert_allclose(res2, expected_res2, rtol=1e-05)

    def test_get_player_ball_distance_increase_cost(self):
        tracking_data = self.match_to_sync.tracking_data.copy()
        event = self.match_to_sync.get_event(2499594225)
        event.jersey = 2
        event.team_side = "home"
        player_ball_diff = np.sqrt(
            (tracking_data["home_2_x"] - tracking_data["ball_x"]) ** 2
            + (tracking_data["home_2_y"] - tracking_data["ball_y"]) ** 2
        )

        res = get_player_ball_distance_increase_cost(tracking_data, event)
        expected_res = sigmoid(np.gradient(player_ball_diff), d=-8.0)

        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

        res2 = get_player_ball_distance_increase_cost(tracking_data, event, d=2)
        expected_res2 = sigmoid(np.gradient(player_ball_diff), d=2)

        np.testing.assert_allclose(res2, expected_res2, rtol=1e-05)

    def test_get_ball_goal_angle_cost(self):
        tracking_data = pd.DataFrame(
            {
                "ball_x": [-10, -25, -40],
                "ball_y": [-15, -10, -5],
            }
        )
        event = self.match_to_sync.get_event(2499594225)
        event.jersey = 2
        event.team_side = "away"

        goal_loc = [-event.pitch_size[0] / 2, 0]

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
        res = get_ball_goal_angle_cost(tracking_data, event)
        expected_res = sigmoid(goal_angle, d=6, e=0.2 * np.pi)

        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

        tracking_data["goal_angle_home_team"] = goal_angle
        tracking_data["goal_angle_away_team"] = goal_angle + np.pi

        res2 = get_ball_goal_angle_cost(tracking_data, event, d=2, e=3)
        expected_res2 = sigmoid(goal_angle + np.pi, d=2, e=3)

        np.testing.assert_allclose(res2, expected_res2, rtol=1e-05)

    def test_base_general_cost_ball_event(self):
        tracking_data = self.match_to_sync.tracking_data.copy()
        event = self.match_to_sync.get_event(2499594225)

        res = base_general_cost_ball_event(tracking_data, event)
        expected_res = combine_cost_functions(
            [
                get_time_difference_cost(tracking_data, event),
                get_distance_ball_event_cost(tracking_data, event),
                get_distance_ball_player_cost(tracking_data, event),
            ]
        )
        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

    def test_base_pass_cost_function(self):
        tracking_data = self.match_to_sync.tracking_data.copy()
        event = self.match_to_sync.get_event(2499594225)

        res = base_pass_cost_function(tracking_data, event)

        expected_res = combine_cost_functions(
            [
                get_time_difference_cost(tracking_data, event),
                get_distance_ball_event_cost(tracking_data, event),
                get_distance_ball_player_cost(tracking_data, event),
                get_ball_acceleration_cost(tracking_data, event),
                get_player_ball_distance_increase_cost(tracking_data, event),
            ]
        )
        np.testing.assert_allclose(res, expected_res, rtol=1e-05)

    def test_base_shot_cost_function(self):
        tracking_data = self.match_to_sync.tracking_data.copy()
        event = self.match_to_sync.get_event(2499594225)

        res = base_shot_cost_function(tracking_data, event)

        expected_res = combine_cost_functions(
            [
                get_time_difference_cost(tracking_data, event),
                get_distance_ball_event_cost(tracking_data, event),
                get_distance_ball_player_cost(tracking_data, event),
                get_ball_acceleration_cost(tracking_data, event),
                get_player_ball_distance_increase_cost(tracking_data, event),
                get_ball_goal_angle_cost(tracking_data, event),
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
