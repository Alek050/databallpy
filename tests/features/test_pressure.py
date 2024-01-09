import math
import unittest

import numpy as np
import pandas as pd

from databallpy.features.angle import get_smallest_angle
from databallpy.features.pressure import (
    calculate_L,
    calculate_variable_dfront,
    calculate_z,
    get_pressure_on_player,
)


class TestPressure(unittest.TestCase):
    def setUp(self):
        self.td_frame = pd.Series(
            {
                "home_1_x": 1.0,
                "home_1_y": 1.0,
                "away_1_x": 2.0,
                "away_1_y": 2.0,
                "away_2_x": 3.0,
                "away_2_y": 3.0,
                "away_3_x": 40.0,
                "away_3_y": 30.0,
            }
        )

    def test_get_pressure_on_player_specified_d_front(self):
        d_front = calculate_variable_dfront(
            self.td_frame, "home_1", max_d_front=9, pitch_length=100.0
        )
        z1 = calculate_z(self.td_frame, "home_1", "away_1", pitch_length=100.0)
        z2 = calculate_z(self.td_frame, "home_1", "away_2", pitch_length=100.0)
        l1 = calculate_L(d_back=3.0, d_front=d_front, z=z1)
        l2 = calculate_L(d_back=3.0, d_front=d_front, z=z2)
        dist1 = math.dist([1, 1], [2, 2])
        dist2 = math.dist([1, 1], [3, 3])

        pres1 = (1 - dist1 / l1) ** 1.75 * 100
        pres2 = (1 - dist2 / l2) ** 1.75 * 100
        expected_pressure = pres1 + pres2

        result = get_pressure_on_player(
            self.td_frame,
            "home_1",
            pitch_size=[100.0, 50.0],
            d_front="variable",
            d_back=3.0,
            q=1.75,
        )
        self.assertAlmostEqual(result, expected_pressure, places=4)

    def test_get_pressure_on_player_variable_d_front(self):
        z1 = calculate_z(self.td_frame, "home_1", "away_1", pitch_length=100.0)
        z2 = calculate_z(self.td_frame, "home_1", "away_2", pitch_length=100.0)
        l1 = calculate_L(d_back=3.0, d_front=9.0, z=z1)
        l2 = calculate_L(d_back=3.0, d_front=9.0, z=z2)
        dist1 = math.dist([1, 1], [2, 2])
        dist2 = math.dist([1, 1], [3, 3])

        pres1 = (1 - dist1 / l1) ** 1.75 * 100
        pres2 = (1 - dist2 / l2) ** 1.75 * 100
        expected_pressure = pres1 + pres2

        result = get_pressure_on_player(
            self.td_frame,
            "home_1",
            pitch_size=[100.0, 50.0],
            d_front=9.0,
            d_back=3.0,
            q=1.75,
        )
        self.assertAlmostEqual(result, expected_pressure, places=4)

    def test_calculate_variable_dfront(self):
        expected = 7.4505
        res = calculate_variable_dfront(
            self.td_frame, "home_1", max_d_front=10.0, pitch_length=100.0
        )
        self.assertAlmostEqual(res, expected, places=4)

    def test_calculate_z(self):
        # vector from home_1 to goal
        a = [49, -1]
        # vector from home_1 to away_1
        b = [1, 1]
        angle = get_smallest_angle(a, b, angle_format="radian")
        expected = (1 + np.cos(angle)) / 2
        res = calculate_z(self.td_frame, "home_1", "away_1", pitch_length=100.0)
        self.assertAlmostEqual(res, expected, places=4)

    def test_calculate_L(self):
        expected = 3.1046
        res = calculate_L(d_back=3.0, d_front=5.0, z=0.2)
        self.assertAlmostEqual(res, expected, places=4)
    
    def test_get_pressure_on_player_wrong_input(self):
        with self.assertRaises(TypeError):
            get_pressure_on_player(
                "td_frame",
                "home_1",
                pitch_size=[100.0, 50.0],
                d_front="variable",
                d_back=3.0,
                q=1.75,
            )

