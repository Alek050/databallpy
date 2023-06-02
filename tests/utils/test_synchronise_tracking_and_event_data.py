import datetime as dt
import unittest

import numpy as np
import pandas as pd

from databallpy.get_match import get_match
from databallpy.utils.synchronise_tracking_and_event_data import (
    _create_sim_mat,
    _needleman_wunsch,
)
from databallpy.utils.utils import MISSING_INT
from tests.expected_outcomes import RES_SIM_MAT, RES_SIM_MAT_NO_PLAYER


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

    def test_synchronise_tracking_and_event_data(self):
        expected_event_data = self.match_to_sync.event_data.copy()
        expected_tracking_data = self.match_to_sync.tracking_data.copy()
        expected_tracking_data["period"] = [1] * 13
        expected_tracking_data["event"] = [
            "pass",
            "pass",
            np.nan,
            np.nan,
            np.nan,
            "take on",
            "tackle",
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]
        expected_tracking_data["event_id"] = [
            2499594225,
            2499594243,
            np.nan,
            np.nan,
            np.nan,
            2499594285,
            2499594291,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]

        expected_event_data.loc[:, "tracking_frame"] = [
            np.nan,
            np.nan,
            np.nan,
            0.0,
            1.0,
            np.nan,
            np.nan,
            5.0,
            6.0,
        ]
        expected_event_data = expected_event_data[
            expected_event_data["type_id"].isin([1, 3, 7])
        ]

        self.match_to_sync.synchronise_tracking_and_event_data(n_batches_per_half=1)

        pd.testing.assert_frame_equal(
            self.match_to_sync.tracking_data, expected_tracking_data
        )
        pd.testing.assert_frame_equal(
            self.match_to_sync.event_data, expected_event_data
        )

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
        tracking_data.reset_index(inplace=True)
        event_data = self.match_to_sync.event_data.copy()
        event_data = event_data[event_data["type_id"].isin([1, 3, 7])].reset_index()
        res = _create_sim_mat(
            tracking_batch=tracking_data,
            event_batch=event_data,
            match=self.match_to_sync,
        )

        np.testing.assert_allclose(expected_res, res)

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
