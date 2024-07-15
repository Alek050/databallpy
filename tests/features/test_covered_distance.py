import unittest

import numpy as np
import pandas as pd

from databallpy.features.covered_distance import (
    _parse_intervals,
    get_covered_distance
)

class TestCoveredDistance(unittest.TestCase):
    def setUp(self):
        self.framerate = 1
        self.vel_intervals = ((-1, 13), (30, 17))
        self.acc_intervals = ((37, 15), (0, 0.75))
        self.input = pd.DataFrame(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "home_1_vx": [10, -20, 10, np.nan, 10, np.nan],
                "home_1_vy": [7, -12.5, 9, np.nan, 15, np.nan],                
                "home_1_velocity": [
                    np.sqrt(149), 
                    np.sqrt(556.25), 
                    np.sqrt(181), 
                    np.nan, 
                    np.sqrt(325), 
                    np.nan
                ],
                "home_1_ax": [-30, 0, np.nan, 0, np.nan, np.nan],
                "home_1_ay": [-19.5, 1, np.nan, 3, np.nan, np.nan],
                "home_1_acceleration": [
                    np.sqrt(1280.25), 
                    1, 
                    np.nan, 
                    3, 
                    np.nan, 
                    np.nan
                ],
                "away_2_x": [2, 3, 1, 1, 3, 2],
                "away_2_y": [1, 2, 3, 4, 5, 6],
                "away_2_vx": [2, 3, 1, 1, 3, 2],
                "away_2_vy": [1, 1, 1, 1, 1, 1],
                "away_2_velocity": [
                    np.sqrt(5), 
                    np.sqrt(10), 
                    np.sqrt(2), 
                    np.sqrt(2), 
                    np.sqrt(10), 
                    np.sqrt(5)
                ],
                "away_2_ax": [1, -0.5, -1, 1, 0.5, -1],
                "away_2_ay": [0, 0, 0, 0, 0, 0],
                "away_2_acceleration": [
                    1, 
                    0.5, 
                    1, 
                    1, 
                    0.5, 
                    1]
            }
        )

        self.expected_total_distance = {
            'home_1': {
                'total_distance': (np.sqrt(149) + np.sqrt(556.25) + np.sqrt(181) + np.sqrt(325))
            },
            'away_2': {
                'total_distance': (np.sqrt(5) + np.sqrt(10) + np.sqrt(2) + np.sqrt(2) + np.sqrt(10) + np.sqrt(5))
                }
        }

        self.expected_output_total_and_vel_distance = {
            'home_1': {
                'total_distance': (np.sqrt(149) + np.sqrt(556.25) + np.sqrt(181) + np.sqrt(325)),
                'total_distance_velocity': [((-1, 13), np.sqrt(149)),((17, 30), (np.sqrt(556.25) + np.sqrt(325)))] 
            },
            'away_2': {
                'total_distance': (np.sqrt(5) + np.sqrt(10) + np.sqrt(2) + np.sqrt(2) + np.sqrt(10) + np.sqrt(5)), 
                'total_distance_velocity': [((-1, 13), (np.sqrt(5) + np.sqrt(10) + np.sqrt(2) + np.sqrt(2) + np.sqrt(10) + np.sqrt(5))),((17, 30), 0)] 
            }
        }

        self.expected_output_total_and_vel_and_acc_distance = {
            'home_1': {
                'total_distance': (np.sqrt(149) + np.sqrt(556.25) + np.sqrt(181) + np.sqrt(325)),
                'total_distance_velocity': [((-1, 13), np.sqrt(149)), ((17, 30), (np.sqrt(556.25) + np.sqrt(325)))], 
                'total_distance_acceleration': [((15, 37), np.sqrt(149)),((0, 0.75), 0)]
            },
            'away_2': {
                'total_distance': (np.sqrt(5) + np.sqrt(10) + np.sqrt(2) + np.sqrt(2) + np.sqrt(10) + np.sqrt(5)), 
                'total_distance_velocity': [((-1, 13), (np.sqrt(5) + np.sqrt(10) + np.sqrt(2) + np.sqrt(2) + np.sqrt(10) + np.sqrt(5))), ((17, 30), 0)] , 
                'total_distance_acceleration': [((15, 37), 0), ((0, 0.75), (np.sqrt(10) + np.sqrt(10)))]
            }
        }

    def test_get_total_distance(self):
        input_df = self.input.copy()
        output = get_covered_distance(input_df, ["home_1","away_2"], self.framerate)
        self.assertDictEqual(output, self.expected_total_distance)

    def test_get_total_and_vel_distance(self):
        input_df = self.input.copy()
        output = get_covered_distance(input_df, ["home_1","away_2"], self.framerate, velocity_intervals=self.vel_intervals)
        self.assertDictEqual(output, self.expected_output_total_and_vel_distance)

    def test_get_total_and_vel_and_acc_distance(self):
        input_df = self.input.copy()
        output = get_covered_distance(input_df, ["home_1","away_2"], self.framerate, velocity_intervals=self.vel_intervals, acceleration_intervals=self.acc_intervals)
        self.assertDictEqual(output, self.expected_output_total_and_vel_and_acc_distance)

    def test_get_covered_distance_wrong_input(self):
        input_df = self.input.copy()
        input_df.drop(columns=["home_1_vx"], inplace=True)
        with self.assertRaises(ValueError):
            get_covered_distance(input_df, ["home_1","away_2"], self.framerate)

        input_df = self.input.copy()
        input_df.drop(columns=["home_1_vx"], inplace=True)
        with self.assertRaises(ValueError):
            get_covered_distance(input_df, ["home_1","away_2"], self.framerate, velocity_intervals=self.vel_intervals)

        input_df = self.input.copy()
        input_df.drop(columns=["home_1_ax"], inplace=True)
        with self.assertRaises(ValueError):
            get_covered_distance(input_df, ["home_1","away_2"], self.framerate, velocity_intervals=self.vel_intervals, acceleration_intervals=self.acc_intervals)

        input_df = self.input.copy()
        with self.assertRaises(TypeError) as cm:
            data = {"ball_x": [1, 2, 3, 4]}
            get_covered_distance(data, ["home_1", "away_2"], 1)
        self.assertEqual(
            str(cm.exception), 
            f"tracking data must be a pandas DataFrame, not a {type(data).__name__}"
        )
        
        with self.assertRaises(TypeError) as cm:
            players = "home_1"
            get_covered_distance(self.input, players, 1)
        self.assertEqual(
            str(cm.exception), 
            f"player_ids must be a list, not a {type(players).__name__}"
        )

        with self.assertRaises(TypeError) as cm:
            players = ["home_1", 123]
            get_covered_distance(self.input, [players], 1)
        self.assertEqual(
            str(cm.exception), 
            "All elements in player_ids must be strings"
        )
        
        with self.assertRaises(TypeError) as cm:
            framerate = "1"
            get_covered_distance(self.input, ["home_1", "away_2"], framerate)
        self.assertEqual(
            str(cm.exception), 
            f"framerate must be a int, not a {type(framerate).__name__}"
        )

    def test_parse_intervals(self):
        velocity = (1, 4, 2, 9, 0)
        expected_output_interval = [(1, 4), (2, 4), (2, 9), (0, 9)]
        output = _parse_intervals(velocity)
        self.assertListEqual(output, expected_output_interval)

        acceleration = ((8, -2), [3, 3.17])
        expected_output_interval = [(-2, 8), (3, 3.17)]
        output = _parse_intervals(acceleration)
        self.assertListEqual(output, expected_output_interval)

    def test_parse_intervals_wrong_input(self):
        intervals = (0, 15.0, 90, 'a', 3, -1)
        with self.assertRaises(TypeError) as cm:
            _parse_intervals(intervals)
        self.assertEqual(
            str(cm.exception), 
            "Intervals must contain either all floats/integers or all tuples/lists."
        )

        intervals = ((0, 15.0), 90, 3, (3, -1))
        with self.assertRaises(TypeError) as cm:
            _parse_intervals(intervals)
        self.assertEqual(
            str(cm.exception), 
            "Intervals must contain either all floats/integers or all tuples/lists."
        )
