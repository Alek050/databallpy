import unittest

import numpy as np
import pandas as pd

from databallpy.load_data.tracking_data.utils._quality_check_tracking_data import (
    _check_ball_velocity,
    _check_missing_ball_data,
    _check_missing_player_data,
    _check_player_velocity,
    _max_sequence_invalid_frames,
    _quality_check_tracking_data,
)
from databallpy.warnings import DataBallPyWarning


class TestQualityCheckTrackingData(unittest.TestCase):
    def setUp(self) -> None:
        self.framerate = 2

        self.tracking_data_warning = pd.DataFrame(
            {
                "ball_x": [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    1,
                    1,
                    50,
                ],
                "ball_y": [1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan, np.nan, 2, 40],
                "ball_status": ["alive"] * 13,
                "home_1_x": [1, 1, 40, 1, 4, 42, 4, 43, 3, 41, 0, 40, 40],
                "home_1_y": [1, 1, 40, 1, 4, 41, 4, 43, 3, 41, 0, 40, 40],
                "matchtime_td": ["00:00:00"] * 13,
            }
        )

        self.tracking_data_no_warning = pd.DataFrame(
            {
                "ball_x": [1] * 13,
                "ball_y": [1] * 13,
                "ball_status": ["alive"] * 13,
                "home_1_x": [1] * 13,
                "home_1_y": [1] * 13,
                "matchtime_td": ["00:00:00"] * 13,
            }
        )

        self.periods = pd.DataFrame(
            {"period_id": [1], "start_frame": [0], "end_frame": [13]}
        )

        self.valid_frames = pd.Series([False, False, True, False, True, True])

    def test_quality_check_tracking_data(self):
        result = _quality_check_tracking_data(
            self.tracking_data_no_warning, self.framerate, self.periods
        )
        self.assertEqual(result, True)

        with self.assertWarns(DataBallPyWarning):
            _quality_check_tracking_data(
                self.tracking_data_warning, self.framerate, self.periods
            )

    def test_check_missing_ball_data(self):
        self.assertIsNone(
            _check_missing_ball_data(self.tracking_data_no_warning, self.framerate)
        )
        with self.assertWarns(DataBallPyWarning):
            _check_missing_ball_data(self.tracking_data_warning, self.framerate)

    def test_check_ball_velocity(self):
        self.assertIsNone(
            _check_ball_velocity(self.tracking_data_no_warning, self.framerate)
        )
        with self.assertWarns(DataBallPyWarning):
            _check_ball_velocity(self.tracking_data_warning, self.framerate)

    def test_check_player_velocity(self):
        self.assertIsNone(
            _check_player_velocity(
                self.tracking_data_no_warning, self.framerate, self.periods
            )
        )
        with self.assertWarns(DataBallPyWarning):
            _check_player_velocity(
                self.tracking_data_warning, self.framerate, self.periods
            )

    def test_max_sequence_invalid_frames(self):
        assert _max_sequence_invalid_frames(self.valid_frames) == 2
        assert _max_sequence_invalid_frames(self.valid_frames, False) == 1

    def test_check_missing_player_data(self):
        with self.assertWarns(DataBallPyWarning):
            allowed_to_sync = _check_missing_player_data(
                self.tracking_data_warning, self.framerate, n_seconds=2.0
            )
            assert allowed_to_sync is False
