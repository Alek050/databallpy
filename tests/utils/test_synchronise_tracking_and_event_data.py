import datetime as dt
import unittest

import numpy as np
import pandas as pd

from databallpy.get_match import get_match
from databallpy.utils.synchronise_tracking_and_event_data import (
    _create_sim_mat,
    _needleman_wunsch,
    pre_compute_synchronisation_variables,
)
from databallpy.utils.utils import MISSING_INT
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
        expected_tracking_data["period"] = [1] * 13
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
            "pass",
            "pass",
            None,
            None,
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
        expected_tracking_data["event_id"] = [
            2499594225,
            2499594243,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            2499594285,
            MISSING_INT,
            2499594291,
            MISSING_INT,
        ]

        expected_event_data.loc[:, "tracking_frame"] = [
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            0,
            1,
            MISSING_INT,
            MISSING_INT,
            9,
            11,
        ]
        match_to_sync = self.match_to_sync.copy()

        match_to_sync.synchronise_tracking_and_event_data(n_batches_per_half=1)

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
        expected_res = expected_res.reshape(13, 4)

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
            match=self.match_to_sync,
        )

        np.testing.assert_allclose(expected_res, res, rtol=1e-05)

    def test_create_sim_mat_missing_player(self):
        expected_res = RES_SIM_MAT_MISSING_PLAYER
        expected_res = expected_res.reshape(13, 4)

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
            match=self.match_to_sync,
        )

        np.testing.assert_allclose(res, expected_res, rtol=1e-05, atol=1e-05)

    def test_create_sim_mat_without_player(self):
        expected_res = RES_SIM_MAT_NO_PLAYER
        expected_res = expected_res.reshape(13, 4)

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
                match=self.match_to_sync,
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
        expected_td["ball_acceleration_sqrt"] = np.sqrt(
            expected_td["ball_acceleration"]
        )
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
        expected_td["databallpy_event"] = [None] * 13
        expected_td["event_id"] = [MISSING_INT] * 13

        res_tracking_data = pre_compute_synchronisation_variables(
            tracking_data=self.match_to_sync.tracking_data.copy(),
            frame_rate=25,
            pitch_dimensions=(106, 68),
            periods=self.match_to_sync.periods.copy(),
        )
        pd.testing.assert_frame_equal(res_tracking_data, expected_td)
