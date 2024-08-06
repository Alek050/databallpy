import unittest

import numpy as np
import pandas as pd

from databallpy.utils.utils import (
    MISSING_INT,
    _to_float,
    _to_int,
    _values_are_equal_,
    get_next_possession_frame,
    sigmoid,
)


class TestUtils(unittest.TestCase):
    def test_to_float(self):
        assert pd.isnull(_to_float("s2"))
        assert pd.isnull(_to_float(None))
        assert 2.0 == _to_float("2")
        assert 3.3 == _to_float("3.3")

    def test_to_int(self):
        assert MISSING_INT == _to_int("s2.3")
        assert MISSING_INT == _to_int(None)
        assert 2 == _to_int("2")
        assert 3 == _to_int("3.3")

    def test_missing_int(self):
        assert MISSING_INT == -999

    def test_get_next_possession_frame(self):
        tracking_data = pd.DataFrame(
            {
                "ball_status": ["dead", "alive", "alive", "dead", "alive", "alive"],
                "player_possession": [None, None, "home_1", None, None, None],
            },
            index=[0, 1, 2, 3, 4, 5],
        )
        frame = tracking_data.iloc[0]
        res1 = get_next_possession_frame(tracking_data, frame, "home_2")

        pd.testing.assert_series_equal(res1, tracking_data.iloc[2])

        res2 = get_next_possession_frame(tracking_data, frame, "home_1")
        pd.testing.assert_series_equal(res2, tracking_data.iloc[3])

        frame = tracking_data.iloc[4]
        tracking_data["ball_status"] = [
            "dead",
            "alive",
            "alive",
            "dead",
            "dead",
            "dead",
        ]
        res3 = get_next_possession_frame(tracking_data, frame, "home_2")
        pd.testing.assert_series_equal(res3, tracking_data.iloc[5])

        tracking_data["ball_status"] = [
            "dead",
            "alive",
            "alive",
            "alive",
            "alive",
            "alive",
        ]
        res4 = get_next_possession_frame(tracking_data, frame, "home_2")
        pd.testing.assert_series_equal(res4, tracking_data.iloc[5])

    def test_sigmoid(self):
        # Test with default parameters
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertAlmostEqual(sigmoid(1), 0.7310585786300049)

        # Test with custom parameters
        self.assertAlmostEqual(sigmoid(0, a=1, b=2, c=3, d=4, e=5), 1.0)
        self.assertAlmostEqual(sigmoid(5, a=1, b=2, c=3, d=4, e=5), 1.5)

        # Test with numpy array input
        np.testing.assert_almost_equal(
            sigmoid(np.array([0, 1])), np.array([0.5, 0.7310585786300049])
        )
        np.testing.assert_almost_equal(
            sigmoid(np.array([0, 5]), a=1, b=2, c=3, d=4, e=5), np.array([1.0, 1.5])
        )

    def test_values_are_equal(self):
        # floats and ints
        self.assertTrue(_values_are_equal_(1.000000, 1.000001))
        self.assertTrue(_values_are_equal_(1.000000, 1))
        self.assertTrue(_values_are_equal_(1, 1.0000001))
        self.assertTrue(_values_are_equal_(1.000000, 1.000000))
        self.assertTrue(_values_are_equal_(np.nan, np.nan))
        self.assertFalse(_values_are_equal_(1.000000, 1.0001))
        self.assertFalse(_values_are_equal_(1.000000, np.nan))
        self.assertFalse(_values_are_equal_(np.nan, 1))
        self.assertFalse(_values_are_equal_(1, 1.0001))
        self.assertFalse(_values_are_equal_(1, 2))
        self.assertFalse(_values_are_equal_(1.00, "1"))

        # pd.timestamps
        self.assertTrue(
            _values_are_equal_(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-01"))
        )
        self.assertFalse(
            _values_are_equal_(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-02"))
        )
        self.assertFalse(_values_are_equal_(pd.Timestamp("2021-01-01"), "2021-01-01"))

        # list and tuple
        self.assertTrue(_values_are_equal_([1, 2], (1, 2)))
        self.assertTrue(_values_are_equal_([1, 2], (1, 2.0000001)))
        self.assertTrue(_values_are_equal_([[1, 2]], [(1, 2)]))
        self.assertFalse(_values_are_equal_([1, 2], (1, 2, 3)))
        self.assertFalse(_values_are_equal_([1, 2], (1, 2.0001)))
        self.assertFalse(_values_are_equal_([[1, 2]], [(1, 2, "nan")]))
        self.assertFalse(_values_are_equal_([1, 2], "[1, 2]"))

        # dict
        self.assertTrue(_values_are_equal_({"a": 1, "b": 2}, {"b": 2, "a": 1}))
        self.assertTrue(
            _values_are_equal_({"a": [1, 2], "b": 2}, {"a": [1, 2], "b": 2})
        )
        self.assertFalse(_values_are_equal_({"a": 1, "b": 2}, {"a": 1, "b": 3}))
        self.assertFalse(_values_are_equal_({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3}))
        self.assertFalse(_values_are_equal_({"a": 1, "b": 2}, {"a": 1, "b": 2.0001}))
        self.assertFalse(_values_are_equal_({"a": 1, "b": 2}, {"a": 1, "b": "2"}))
        self.assertFalse(_values_are_equal_({"a": 1, "b": 2}, {"a": 1, "c": 2}))
        self.assertFalse(_values_are_equal_({"a": 1, "b": 2}, "[1, 2]"))

        # np.ndarray
        self.assertTrue(_values_are_equal_(np.array([1, 2]), np.array([1, 2])))
        self.assertFalse(_values_are_equal_(np.array([1, 2]), np.array([1, 3])))
        self.assertFalse(_values_are_equal_(np.array([1, 2]), np.array([1, 2.0001])))
        self.assertFalse(_values_are_equal_(np.array([1, 2]), np.array(["1", "2"])))
        self.assertFalse(_values_are_equal_(np.array([1, 2]), [1, 2]))
        self.assertFalse(_values_are_equal_(np.array([1, 2]), np.array([[1, 2]])))

        # pd.Series or pd.DataFrame
        self.assertTrue(_values_are_equal_(pd.Series([1, 2]), pd.Series([1, 2])))
        self.assertFalse(_values_are_equal_(pd.Series([1, 2]), pd.Series([1, 3])))
        self.assertTrue(
            _values_are_equal_(
                pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
                pd.DataFrame({"b": [3, 4], "a": [1, 2]}),
            )
        )
        self.assertFalse(
            _values_are_equal_(
                pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
                pd.DataFrame({"b": [3, 4], "a": [1, 3]}),
            )
        )
        self.assertFalse(
            _values_are_equal_(pd.Series([1, 2]), pd.DataFrame({"a": [1, 2]}))
        )
        self.assertFalse(_values_are_equal_(pd.Series([1, 2]), "[1, 2]"))
        self.assertFalse(
            _values_are_equal_(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), "[1, 2]")
        )
        self.assertFalse(
            _values_are_equal_(
                pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
                pd.DataFrame({"b": [3, 4], "c": [3, 4], "a": [1, 2]}),
            )
        )
        self.assertFalse(
            _values_are_equal_(
                pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
                pd.DataFrame({"c": [3, 4], "a": [1, 2]}),
            )
        )

        # NaT
        self.assertTrue(
            _values_are_equal_(pd.to_datetime("NaT"), pd.to_datetime("NaT"))
        )
        self.assertFalse(
            _values_are_equal_(pd.to_datetime("NaT"), pd.to_datetime("2021-01-01"))
        )
        self.assertFalse(_values_are_equal_(pd.to_datetime("NaT"), "NaT"))

        # NotImplementedError
        class CustomType:
            def __init__(self, value):
                self.value = value

        with self.assertRaises(NotImplementedError):
            _values_are_equal_(CustomType(1), CustomType(1))
