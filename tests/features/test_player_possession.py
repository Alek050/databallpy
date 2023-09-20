import unittest

import numpy as np
import pandas as pd

from databallpy.features.player_possession import (
    get_distance_between_ball_and_players,
    get_duels,
    get_individual_player_possessions_and_duels,
    get_initial_possessions,
    get_lost_possession_idx,
    get_valid_gains,
)


class TestPlayerPossession(unittest.TestCase):
    def setUp(self):
        self.tracking_data_full = pd.DataFrame(
            {
                "ball_x": [0, 0, 5, 10, 10, 11, 22, 24, 26, 26],
                "ball_y": [0, 0, 5, 10, 10, 10, 10, 10, 10, 15],
                "home_1_x": [0, 0, 5, 0, 0, 0, 0, 0, 26, 0],
                "home_1_y": [0, 0, 5, 0, 0, 0, 0, 0, 10.5, 0],
                "away_1_x": [0.5, 0.5, 0.5, 0.5, 10, 11, 18, 24, 26, 26],
                "away_1_y": [0.5, 0.5, 0.5, 0.5, 10, 10, 10, 10, 10, 15],
                "ball_status": [
                    "dead",
                    "alive",
                    "alive",
                    "alive",
                    "alive",
                    "alive",
                    "alive",
                    "alive",
                    "alive",
                    "alive",
                ],
            }
        )

        self.tracking_data = pd.DataFrame(
            {
                "ball_x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "ball_y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "ball_velocity": [
                    np.nan,
                    1.41,
                    1.41,
                    1.41,
                    1.41,
                    1.41,
                    1.41,
                    1.41,
                    1.41,
                    1.41,
                ],
                "home_1_x": [0, 5, 2, 3, 4, 5, 1, 7, 8, 9],
                "home_1_y": [0, 5, 2, 3, 4, 5, 1, 7, 8, 9],
                "home_2_vx": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "home_2_vy": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "away_2_x": [0, 1, 6, 3, 4, 5, 6, 11, 8, 9],
                "away_2_y": [0, 1, 6, 3, 4, 5, 6, 11, 8, 9],
                "databallpy_event": [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
            }
        )
        self.distances = pd.DataFrame(
            {
                "home_1": [10, 1, 2.2, 12, 14, 20, 20, 20, 20, 20],
                "home_2": [11, 1.3, 12, 1, 2, 20, 20, 20, 20, 20],
                "away_2": [10.5, 0.3, 14, 15, 16, 20, 20, 1, 0.5, 0.5],
                "away_3": [13, 14, 1.3, 16, 17, 20, 20, 20, 20, 20],
            }
        )
        self.duels = pd.Series(
            [None, None, None, None, None, None, None, None, None, None]
        )
        self.possession_start_idxs = np.array([0, 6])
        self.possession_end_idxs = np.array([4, 9])

    def test_get_individual_player_possession(self):
        td = self.tracking_data_full.copy()
        possessions, duels = get_individual_player_possessions_and_duels(
            td, 1, bv_threshold=3
        )
        expected_possessions = pd.Series(
            [
                None,
                "home_1",
                "home_1",
                None,
                "away_1",
                "away_1",
                "away_1",
                "away_1",
                "away_1",
                "away_1",
            ]
        )
        expected_duels = pd.Series(
            [
                None,
                "home_1-away_1",
                None,
                None,
                None,
                None,
                None,
                None,
                "home_1-away_1",
                None,
            ]
        )
        pd.testing.assert_series_equal(possessions, expected_possessions)
        pd.testing.assert_series_equal(duels, expected_duels)

    def test_get_initial_possessions(self):
        possessions_no_min_frames = get_initial_possessions(
            self.tracking_data, self.distances, pz_radius=1.5, min_frames=0
        )
        expected_no_min_frames = pd.Series(
            [
                None,
                "away_2",
                "away_3",
                "home_2",
                None,
                None,
                None,
                "away_2",
                "away_2",
                "away_2",
            ]
        )
        pd.testing.assert_series_equal(
            possessions_no_min_frames, expected_no_min_frames
        )

        possessions_min_frames = get_initial_possessions(
            self.tracking_data, self.distances, pz_radius=1.5, min_frames=2
        )
        expected_min_frames = pd.Series(
            [None, None, None, None, None, None, None, "away_2", "away_2", "away_2"]
        )
        pd.testing.assert_series_equal(possessions_min_frames, expected_min_frames)

    def test_get_duels(self):
        duels = get_duels(
            self.tracking_data.iloc[:5], self.distances.iloc[:5], dz_radius=1.5
        )
        expected = pd.Series([None, "home_1-away_2", None, None, None])
        pd.testing.assert_series_equal(duels, expected)

    def test_get_distance_between_ball_and_players(self):
        distances = get_distance_between_ball_and_players(self.tracking_data)
        expected = pd.DataFrame(
            {
                "home_1": [0, np.sqrt(32), 0, 0, 0, 0, np.sqrt(50), 0, 0, 0],
                "away_2": [0, 0, np.sqrt(32), 0, 0, 0, 0, np.sqrt(32), 0, 0],
            }
        )
        pd.testing.assert_frame_equal(distances, expected)

    def test_get_lost_possession_idx(self):
        # ball does not go faster than 2m/s, thus last index should be returned
        lost_possession_idx = get_lost_possession_idx(self.tracking_data, 2)
        self.assertEqual(lost_possession_idx, 9)

        td = self.tracking_data.copy()
        td.loc[[5, 7], "ball_velocity"] = [3, 3]
        lost_possession_idx = get_lost_possession_idx(td, 2)
        self.assertEqual(lost_possession_idx, 5)

    def test_get_valid_gains_event_found(self):
        td = pd.DataFrame(
            {
                "ball_x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "ball_y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "ball_velocity": [np.nan, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "home_1_x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "home_1_y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "databallpy_event": [
                    None,
                    "pass",
                    None,
                    None,
                    None,
                    None,
                    "shot",
                    "tackle",
                    None,
                    None,
                ],
            }
        )
        duels = pd.Series(
            [None, None, None, None, None, None, None, "home_1-away_2", None, None]
        )

        valid_gains = get_valid_gains(
            td, self.possession_start_idxs, self.possession_end_idxs, 5.0, 10.0, duels
        )
        np.testing.assert_allclose(valid_gains, np.array([True, False]))

        # if duels not passed, should not check on databallpy_event
        valid_gains = get_valid_gains(
            td, self.possession_start_idxs, self.possession_end_idxs, 5.0, 10.0
        )
        np.testing.assert_allclose(valid_gains, np.array([False, False]))

    def test_get_valid_gains_ball_angle_found(self):
        td = pd.DataFrame(
            {
                "ball_x": [0, 1, 2, 3, 4, 5, 6, 7, 8, -9],
                "ball_y": [0, 1, 2, 3, 4, 5, 6, 7, 8, -9],
                "ball_velocity": [np.nan, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "home_1_x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "home_1_y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "databallpy_event": [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
            }
        )

        valid_gains = get_valid_gains(
            td,
            self.possession_start_idxs,
            self.possession_end_idxs,
            5.0,
            10.0,
            self.duels,
        )
        np.testing.assert_allclose(valid_gains, np.array([False, True]))

    def test_get_valid_gains_ball_speed_found(self):
        td = pd.DataFrame(
            {
                "ball_x": [0, 1, -20, 3, 4, 5, 6, 7, 8, 9],
                "ball_y": [0, 1, -20, 3, 4, 5, 6, 7, 8, 9],
                "ball_velocity": [np.nan, 1, np.sqrt(21**2 * 2), 1, 1, 1, 1, 1, 1, 1],
                "home_1_x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "home_1_y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "databallpy_event": [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
            }
        )

        valid_gains = get_valid_gains(
            td,
            self.possession_start_idxs,
            self.possession_end_idxs,
            5.0,
            10.0,
            self.duels,
        )
        np.testing.assert_allclose(valid_gains, np.array([True, False]))
