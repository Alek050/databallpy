import unittest

import numpy as np
import pandas as pd

from databallpy.features.feature_utils import _check_column_ids


class TestFeatureUtils(unittest.TestCase):
    def test_check_column_ids(self):
        tracking_data = pd.DataFrame(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "ball_x": [10, 20, -30, 40, np.nan, 60],
                "ball_y": [5, 12, -20, 30, np.nan, 60],
                "away_13_x": [10, 20, -30, 40, np.nan, 60],
                "away_13_y": [5, 12, -20, 30, np.nan, 60],
            }
        )

        with self.assertRaises(TypeError):
            _check_column_ids(tracking_data, column_ids="home_1_x")

        with self.assertRaises(ValueError):
            _check_column_ids(tracking_data, column_ids=[])

        with self.assertRaises(TypeError):
            _check_column_ids(tracking_data, column_ids=[1, 2, 3])

        with self.assertRaises(ValueError):
            _check_column_ids(tracking_data, column_ids=["home_2"])

        _check_column_ids(tracking_data, column_ids=["home_1", "ball", "away_13"])
