import unittest

import numpy as np
import pandas as pd

from databallpy.features.covered_distance import (
    get_covered_distance
)


class TestCoveredDistance(unittest.TestCase):
    def setUp(self):

        # set input and expected data
        self.framerate = 1
        self.vel_intervals = (0, 15, 90, 50, 3, -1)
        self.acc_intervals = (-2, 30, 0, 10, -2, -50)

        self.input = pd.DataFrame(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "away_2_x": [0, 2, 4, 6, 8, 10],
                "away_2_y": [1, 2, 3, 4, 5, 6],
            }
        )
        self.expected_covered_distance = {
            'home_1': {'total_distance': 193.65, 
                       'total_distance_velocity': [], 
                       'total_distance_acceleration': []},
            'away_2': {'total_distance': 11.2, 
                       'total_distance_velocity': [], 
                       'total_distance_acceleration': []},
        }

        
        self.expected_output_vel = {
            'home_1': {'total_distance': 193.65, 
                       'total_distance_velocity': [((0, 15), 12.21),((50, 90), 145.4)], 
                       'total_distance_acceleration': []},
            'away_2': {'total_distance': 11.2, 
                       'total_distance_velocity': [((0, 15), 11.2),((50, 90), 0)], 
                       'total_distance_acceleration': []},
        }

        self.expected_output_acc = {
            'home_1': {'total_distance': 193.65, 
                       'total_distance_velocity': [], 
                       'total_distance_acceleration': [((-2, 30), 26.66),((0, 10), 0),((-2, -50), 49.967)]},
            'away_2': {'total_distance': 11.2, 
                       'total_distance_velocity': [], 
                       'total_distance_acceleration': [((-2, 30), 0),((0, 10), 0),((-2, -50), 0)]},
        }

    # test covered distance
    def test_get_covered_distance(self):
        input_df = self.input.copy()
        output = get_covered_distance(input_df, ["home_1","away_2"], self.framerate)
        pd.testing.assert_frame_equal(output, self.expected_covered_distance)

    # test covered distance in velocity interval
    def test_get_covered_distance_vel(self):
        input_df = self.input.copy()
        vel_intervals = self.vel_intervals.copy()
        output = get_covered_distance(input_df, ["home_1","away_2"], self.framerate, vel_intervals=vel_intervals)
        pd.testing.assert_frame_equal(output, self.expected_output_vel)

    # test covered distance in acceleration interval
    def test_get_covered_distance_acc(self):
        input_df = self.input.copy()
        acc_intervals = self.acc_intervals.copy()
        output = get_covered_distance(input_df, ["home_1","away_2"], self.framerate, acc_intervals=acc_intervals)
        pd.testing.assert_frame_equal(output, self.expected_output_acc)

    # test check intervals function
    def test_check_intervals(self):
        # check type
        intervals = (0, 15, 90, 'a', 3, -1)
        with self.assertRaises(TypeError) as cm:
            get_covered_distance(self.input, "home_1", 1,vel_intervals=intervals)
        self.assertEqual(
            str(cm.exception), 
            "All elements in the tuple must be integers"
        )

        # check even number
        intervals = (0, 15, 90, 3, -1)
        with self.assertRaises(ValueError) as cm:
            get_covered_distance(self.input, "home_1", 1,vel_intervals=intervals)
        self.assertEqual(
            str(cm.exception), 
            "Intervals must contain an even number of elements."
        )
        """
    # test frames function
    def _check_frames(self):
            # check type
        intervals = (0, 15, 90, 'a', 3, -1)
        with self.assertRaises(TypeError) as cm:
            get_covered_distance(self.input, "home_1", 1,vel_intervals=intervals)
        self.assertEqual(
            str(cm.exception), 
            "All elements in the tuple must be integers"
        )

        # check even number
        intervals = (0, 15, 90, 3, -1)
        with self.assertRaises(ValueError) as cm:
            get_covered_distance(self.input, "home_1", 1,vel_intervals=intervals)
        self.assertEqual(
            str(cm.exception), 
            "Intervals must contain an even number of elements."
        )
"""
    # check input covered distance
    def test_get_covered_distance_input(self):
        
        # tracking_data
        with self.assertRaises(TypeError) as cm:
            get_covered_distance({"ball_x": [1, 2, 3, 4]}, ["home_1", "away_2"], 1)
        self.assertEqual(
            str(cm.exception), 
            "tracking data must be a pandas DataFrame, not a str"
        )
        
        # player_ids
        with self.assertRaises(TypeError) as cm:
            get_covered_distance(self.input, "home_1", 1)
        self.assertEqual(
            str(cm.exception), 
            "player_ids must be a list, not a str"
        )

        with self.assertRaises(TypeError) as cm:
            get_covered_distance(self.input, ["home_1", 123], 1)
        self.assertEqual(
            str(cm.exception), 
            "All elements in player_ids must be strings"
        )
        
        # framerate
        with self.assertRaises(TypeError) as cm:
            get_covered_distance(self.input, ["home_1", "away_2"], "1")
        self.assertEqual(
            str(cm.exception), 
            "framerate must be a int, not a str"
        )

        # intervals




    def test_get_acceleration(self):
        input_df = self.input.copy()
        input_df.drop(columns=["home_1_vx"], inplace=True)
        with self.assertRaises(ValueError):
            get_acceleration(input_df, ["home_1"], self.framerate)
       
        # checkt of snelheid aanwezig is, bij mij niet nodig want snelheid is nodig voor afstand (default)
        
        input_df = self.input.copy()
        output = get_acceleration(input_df, ["home_1"], self.framerate)
        pd.testing.assert_frame_equal(output, self.expected_output_acc)

        with self.assertRaises(ValueError):
            get_acceleration(input_df, ["home_1"], self.framerate, filter_type="wrong")

    def test_differentiate_sg_filter(self):
        output = _differentiate(
            self.input.copy(),
            new_name="velocity",
            metric="",
            frame_rate=self.framerate,
            filter_type="savitzky_golay",
            window=2,
            max_val=np.nan,
            poly_order=1,
            column_ids=["home_1"],
        )

        expected_output = pd.DataFrame(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "home_1_vx": [10.0, -5.0, np.nan, np.nan, np.nan, np.nan],
                "home_1_vy": [7.0, -1.75, np.nan, np.nan, np.nan, np.nan],
                "home_1_velocity": [
                    np.sqrt(149),
                    np.sqrt(25 + 1.75**2),
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
            }
        )
        pd.testing.assert_frame_equal(output, expected_output)

    def test_differentiate_ma_filter(self):
        output = _differentiate(
            self.input.copy(),
            new_name="velocity",
            metric="",
            frame_rate=self.framerate,
            filter_type="moving_average",
            window=2,
            max_val=51,
            poly_order=1,
            column_ids=None,
        )












        expected_output = pd.DataFrame(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "home_1_vx": [5.0, -5.0, -5.0, np.nan, np.nan, np.nan],
                "home_1_vy": [3.5, -2.75, -1.75, np.nan, np.nan, np.nan],
                "home_1_velocity": [
                    np.sqrt(25 + 3.5**2),
                    np.sqrt(25 + 2.75**2),
                    np.sqrt(25 + 1.75**2),
                    np.nan,
                    np.nan,
                    np.nan,
                ],
            }
        )
        pd.testing.assert_frame_equal(output, expected_output)