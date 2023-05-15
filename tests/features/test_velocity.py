import unittest

import numpy as np
import pandas as pd

from databallpy.features.velocity import get_velocity


class TestGetVelocity(unittest.TestCase):
    def setUp(self):
        self.input = pd.DataFrame(
            {
                "ball": [10, 20, -30, 40, np.nan, 60],
            }
        )
        self.expected_output = pd.DataFrame(
            {
                "ball": [10, 20, -30, 40, np.nan, 60],
                "ball_v": [np.nan, 10, -50, 70, np.nan, np.nan],
            }
        )
        self.framerate = 1

    def test_get_velocity(self):
        output = get_velocity(self.input, ["ball"], self.framerate)
        pd.testing.assert_frame_equal(output, self.expected_output)
