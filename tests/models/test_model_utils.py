import unittest

import numpy as np

from databallpy.models.utils import get_xT_prediction, scale_and_predict_logreg


class TestUtils(unittest.TestCase):
    def test_scale_and_predict_logreg(self):
        input = np.array([[1, 2, 3]])
        params = {
            "standard_scaler": {
                "mean": {"x": 2, "y": 3, "z": 4},
                "var": {"x": 1, "y": 1, "z": 1},
            },
            "logreg": {"coefs": {"x": 1, "y": 2, "z": 3}, "intercept": -1},
        }

        np.testing.assert_almost_equal(
            scale_and_predict_logreg(input, params)[0], 0.0009, decimal=4
        )

        with self.assertRaises(TypeError):
            scale_and_predict_logreg([[1, 2, 3]], params)
        with self.assertRaises(ValueError):
            scale_and_predict_logreg(np.array([[1, 2]]), params)
        with self.assertRaises(ValueError):
            scale_and_predict_logreg(np.array([1, 2, 3]), params)

    def test_get_xT_prediction(self):
        xT_model = np.array([[0.1, 0.2], [0.3, 0.4]])
        x = 25.0

        y = 17.0

        np.testing.assert_almost_equal(get_xT_prediction(x, y, xT_model), 0.4)
