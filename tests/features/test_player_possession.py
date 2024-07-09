import unittest

import numpy as np
import pandas as pd

from databallpy.features import add_velocity
from databallpy.features.player_possession import (
    get_ball_angle_condition,
    get_ball_losses_and_updated_gain_idxs,
    get_ball_speed_condition,
    get_distance_between_ball_and_players,
    get_individual_player_possession,
    get_initial_possessions,
    get_start_end_idxs,
    get_valid_gains,
)


class TestPlayerPossession(unittest.TestCase):
    def setUp(self) -> None:
        self.tracking_data = pd.DataFrame(
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
        add_velocity(self.tracking_data, "ball", 1, inplace=True)

        self.expected_distances = pd.DataFrame(
            {
                "home_1": [
                    0.0,
                    0.0,
                    0.0,
                    np.sqrt(200),
                    np.sqrt(200),
                    np.sqrt(100 + 11**2),
                    np.sqrt(100 + 22**2),
                    np.sqrt(100 + 24**2),
                    0.5,
                    np.sqrt(26**2 + 15**2),
                ],
                "away_1": [
                    np.sqrt(0.5),
                    np.sqrt(0.5),
                    np.sqrt(4.5**2 + 4.5**2),
                    np.sqrt(9.5**2 + 9.5**2),
                    0,
                    0,
                    4,
                    0,
                    0,
                    0,
                ],
            }
        )

    def test_get_individual_player_possession(self):
        td = self.tracking_data.copy()
        expected_possession = np.array(
            [
                None,
                "home_1",
                None,
                None,
                "away_1",
                "away_1",
                "away_1",
                "away_1",
                "away_1",
                None,
            ]
        )
        individual_possessions = get_individual_player_possession(td)
        np.testing.assert_array_equal(individual_possessions, expected_possession)

        assert "player_possession" not in td.columns
        get_individual_player_possession(td, inplace=True)
        assert "player_possession" in td.columns
        np.testing.assert_array_equal(td["player_possession"], expected_possession)

        with self.assertRaises(ValueError):
            get_individual_player_possession(td.drop(columns=["ball_velocity"]))

    def test_get_distance_between_ball_and_players(self):
        td = self.tracking_data.copy()
        distances = get_distance_between_ball_and_players(td)
        pd.testing.assert_frame_equal(distances, self.expected_distances)

    def test_get_initial_possessions(self):
        distances_df = pd.DataFrame(
            {
                "home_1": [0, 0, 1, 2, 3, 4, 3, 2, 1.0],
                "away_1": [4, 1.0, 0, 1, 2, 3, 4, 3, 2],
            }
        )
        initial_possessions = get_initial_possessions(2, distances_df)
        expected_initial_possessions = np.array(
            ["home_1", "home_1", "away_1", "away_1", None, None, None, None, "home_1"]
        )
        np.testing.assert_array_equal(initial_possessions, expected_initial_possessions)

    def test_get_valid_gains(self):
        start_idxs = np.array([0, 5, 6, 9])
        end_idxs = np.array([1, 5, 7, 9])
        td = self.tracking_data.copy()

        valid_gains = get_valid_gains(td, start_idxs, end_idxs, 3.0, 10.0, 1)
        np.testing.assert_array_equal(valid_gains, [True, True, True, True])

        valid_gains = get_valid_gains(td, start_idxs, end_idxs, 3.0, 10.0, 2)
        np.testing.assert_array_equal(valid_gains, [True, False, True, False])

    def test_get_start_end_idxs(self):
        pz_intial = np.array(
            [
                "home_2",
                "home_2",
                None,
                None,
                None,
                "away_6",
                "away_7",
                "away_7",
                None,
                "home_3",
            ]
        )
        start_idxs, end_idxs = get_start_end_idxs(pz_intial)
        np.testing.assert_array_equal(start_idxs, [0, 5, 6, 9])
        np.testing.assert_array_equal(end_idxs, [1, 5, 7, 9])

    def test_get_ball_speed_condition(self):
        td = self.tracking_data.copy()
        start_idxs = np.array([1, 4])
        end_idxs = np.array([3, 6])

        ball_speed_condition = get_ball_speed_condition(td, start_idxs, end_idxs, 5.0)
        np.testing.assert_array_equal(ball_speed_condition, [False, True])

        ball_speed_condition = get_ball_speed_condition(td, start_idxs, end_idxs, 1.0)
        np.testing.assert_array_equal(ball_speed_condition, [True, True])

    def test_ball_angle_condition(self):
        td = self.tracking_data.copy()
        start_idxs = np.array([2, 5])
        end_idxs = np.array([4, 7])

        ball_angle_condition = get_ball_angle_condition(td, start_idxs, end_idxs, 10)
        np.testing.assert_array_equal(ball_angle_condition, [True, False])

        ball_angle_condition = get_ball_angle_condition(td, start_idxs, end_idxs, 60)
        np.testing.assert_array_equal(ball_angle_condition, [False, False])

    def test_get_ball_losses_and_updated_gain_idxs(self):
        start_idxs = np.array([0, 5, 6, 9])
        end_idxs = np.array([1, 5, 7, 9])
        valid_gains = np.array([True, True, True, False])
        initial_possession = np.array(
            [
                "home_1",
                "home_1",
                None,
                None,
                None,
                "home_1",
                "away_1",
                "away_1",
                None,
                "home_1",
            ]
        )

        valid_gains_idxs, ball_losses_idxs = get_ball_losses_and_updated_gain_idxs(
            start_idxs,
            end_idxs,
            valid_gains,
            initial_possession,
        )

        expected_valid_gains_idxs = np.array([0, 6])
        expected_ball_losses_idxs = np.array([5, 7])

        np.testing.assert_array_equal(valid_gains_idxs, expected_valid_gains_idxs)
        np.testing.assert_array_equal(ball_losses_idxs, expected_ball_losses_idxs)
