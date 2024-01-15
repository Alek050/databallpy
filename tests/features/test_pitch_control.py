import unittest

import numpy as np
import pandas as pd

from databallpy.features.pitch_control import (
    calculate_covariance_matrix,
    calculate_pitch_control_surface_radius,
    calculate_scaling_matrix,
    get_mean_position_of_influence,
    get_pitch_control_period,
    get_pitch_control_single_frame,
    get_player_influence,
    get_team_influence,
    normalize_values,
)


class TestPitchControl(unittest.TestCase):
    def setUp(self):
        self.td = pd.DataFrame(
            {
                "home_1_x": [1.0, 1.0],
                "home_1_y": [2.0, 2.0],
                "home_1_vx": [0.5, 0.5],
                "home_1_vy": [0.5, 2.0],
                "away_1_x": [3.0, 4.0],
                "away_1_y": [4.0, 3.0],
                "away_1_vx": [0.5, 6.0],
                "away_1_vy": [0.5, 5.0],
                "ball_x": [5.0, 22],
                "ball_y": [6.0, 22],
                "home_2_x": [23.0, 19.0],
                "home_2_y": [42.0, 12.0],
                "home_2_vx": [np.nan, 12.0],
                "home_2_vy": [np.nan, -8.0],
                "home_3_x": [-23.0, 21],
                "home_3_y": [-42.0, 1],
                "home_3_vx": [11, -6.5],
                "home_3_vy": [2, 2.22],
            },
            index=[1, 2],
        )

    def test_get_pitch_control_period(self):
        td = self.td.copy()
        grid = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
        output = get_pitch_control_period(td, grid)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, (2, grid[0].shape[0], grid[0].shape[1]))
        self.assertAlmostEqual(output[0].sum(), 1.0)
        self.assertAlmostEqual(output[1].sum(), 2.0)

    def test_get_pitch_control_period_errors(self):
        td = self.td.copy()
        grid = np.array([[[0, 0], [0, 0]]])
        with self.assertRaises(IndexError):
            get_pitch_control_period(td, grid)

    def test_get_pitch_control_single_frame(self):
        frame = self.td.iloc[0].copy()
        grid = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
        output = get_pitch_control_single_frame(frame, grid)

        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, grid[0].shape)
        self.assertAlmostEqual(output.sum(), 1.0)  # 2 home - 1 away

    def test_get_team_influence(self):
        frame = self.td.iloc[0].copy()
        distances = pd.Series(
            {
                "home_1": np.sqrt(32),
                "home_2": np.sqrt(8.0),
                "home_3": np.sqrt(28 * 28 + 48 * 48),
            }
        )
        col_ids = ["home_1", "home_2", "home_3"]
        grid = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
        home_1 = get_player_influence(1.0, 2.0, 0.5, 0.5, np.sqrt(32), grid)
        home_3 = get_player_influence(
            -23.0, -42.0, 11, 2, np.sqrt(28 * 28 + 48 * 48), grid
        )

        output = get_team_influence(frame, col_ids, grid)
        output_dist = get_team_influence(
            frame, col_ids, grid, player_ball_distances=distances
        )

        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, grid[0].shape)
        np.testing.assert_array_almost_equal(output, home_1 + home_3)
        self.assertAlmostEqual(output.sum(), 2.0)
        np.testing.assert_array_almost_equal(output_dist, home_1 + home_3)

    def test_get_player_influence(self):
        x_val = 2.0
        y_val = 3.0
        vx_val = 1.0
        vy_val = 1.0
        distance_to_ball = 5.0
        grid = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
        output = get_player_influence(
            x_val, y_val, vx_val, vy_val, distance_to_ball, grid
        )

        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, grid[0].shape)
        self.assertEqual(output.sum(), 1.0)

    def test_normalize_values(self):
        res = normalize_values(np.array([[1, 3, 5], [0, 1, 0]]))
        np.testing.assert_array_almost_equal(
            res, np.array([[0.1, 0.3, 0.5], [0.0, 0.1, 0.0]])
        )

    def test_get_mean_position_of_influence(self):
        res = get_mean_position_of_influence(22.0, 23.5, -8.0, 11.0)
        self.assertEqual(res[0], 18.0)
        self.assertEqual(res[1], 29.0)

    def test_calculate_covariance_matrix(self):
        vx_val = 2.0
        vy_val = 2.0
        scaling_matrix = np.array([[1.0, 0.0], [0.0, 1.5]])

        rotation_matrix = np.array(
            [
                [np.sqrt(2) / 2, -np.sqrt(2) / 2],
                [np.sqrt(2) / 2, np.sqrt(2) / 2],
            ]
        )
        expected_output = np.dot(
            rotation_matrix,
            np.dot(
                scaling_matrix, np.dot(scaling_matrix, np.linalg.inv(rotation_matrix))
            ),
        )

        output = calculate_covariance_matrix(vx_val, vy_val, scaling_matrix)

        np.testing.assert_array_almost_equal(output, expected_output)

    def test_calculate_scaling_matrix(self):
        speed_magnitude = 10.0
        distance_to_ball = 0.0
        max_speed = 13.0
        influence_radius = calculate_pitch_control_surface_radius(distance_to_ball)
        assert influence_radius == 4.0 * 1.8
        ratio_of_max_speed = 100 / 169.0
        expected_output = np.array(
            [
                [(influence_radius + (influence_radius * ratio_of_max_speed)) / 2, 0],
                [0, (influence_radius - (influence_radius * ratio_of_max_speed)) / 2],
            ]
        )

        output = calculate_scaling_matrix(speed_magnitude, distance_to_ball, max_speed)
        np.testing.assert_array_equal(output, expected_output)

    def test_calculate_pitch_control_surface_radius(self):
        distance_to_ball = 10.0
        min_r = 4.0
        max_r = 10.0
        expected_output = (
            min(min_r + 0.00025 * np.power(distance_to_ball, 3.5), max_r) * 1.8
        )
        output = calculate_pitch_control_surface_radius(distance_to_ball, min_r, max_r)
        self.assertEqual(output, expected_output)

        result1 = calculate_pitch_control_surface_radius(1, min_r, max_r)
        result2 = calculate_pitch_control_surface_radius(15, min_r, max_r)
        assert min_r * 1.8 < result1 and result1 < result2 and result2 < max_r * 1.8


if __name__ == "__main__":
    unittest.main()
