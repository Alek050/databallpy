import unittest

import numpy as np
import pandas as pd

from databallpy.features.angle import get_smallest_angle
from databallpy.features.pressure import (
    calculate_l,
    calculate_variable_dfront,
    calculate_z,
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

    def test_calculate_l(self):
        expected = 3.1046
        res = calculate_l(d_back=3.0, d_front=5.0, z=0.2)
        self.assertAlmostEqual(res, expected, places=4)
