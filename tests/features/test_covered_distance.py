import unittest

from databallpy.features.covered_distance import _parse_intervals


class TestCoveredDistance(unittest.TestCase):
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
        intervals = (0, 15.0, 90, "a", 3, -1)
        with self.assertRaises(TypeError):
            _parse_intervals(intervals)

        intervals = ((0, 15.0), 90, 3, (3, -1))
        with self.assertRaises(TypeError):
            _parse_intervals(intervals)
