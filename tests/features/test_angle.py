import unittest

import numpy as np

from databallpy.features.angle import get_smallest_angle


class TestGetSmallestAngle(unittest.TestCase):
    def test_get_smallest_angle_radian_90(self):
        a = [1, 1]
        b = [-1, 1]
        expected_result = np.pi / 2
        self.assertAlmostEqual(get_smallest_angle(a, b), expected_result)

    def test_get_smallest_angle_degree_90(self):
        a = [1, 1]
        b = [-1, 1]
        expected_result = 90
        self.assertAlmostEqual(
            get_smallest_angle(a, b, angle_format="degree"), expected_result
        )

    def test_get_smallest_angle_degree_75(self):
        a = [1, 1]
        b = [-1, 2]
        expected_result = 71.56505117707799
        self.assertAlmostEqual(
            get_smallest_angle(a, b, angle_format="degree"), expected_result
        )

    def test_get_smallest_angle_invalid_format(self):
        a = [1, 1]
        b = [-1, 1]
        with self.assertRaises(ValueError):
            get_smallest_angle(a, b, angle_format="invalid_format")

    def test_get_smallest_angle_invalid_length(self):
        a = [1, 1, 0]
        b = [-1, 1]
        with self.assertRaises(ValueError):
            get_smallest_angle(a, b)

        a = [[1, 1, 0], [1, 1, 0]]
        b = [[-1, 1, 1], [-1, 1, 1]]
        with self.assertRaises(ValueError):
            get_smallest_angle(a, b)

    def test_get_smallest_angle_invalid_type(self):
        a = {"list1": [1, 1, 0]}
        b = [-1, 1]
        with self.assertRaises(TypeError):
            get_smallest_angle(a, b)
