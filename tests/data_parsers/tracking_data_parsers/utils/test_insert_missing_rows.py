import unittest

import numpy as np
import pandas as pd

from databallpy.data_parsers.tracking_data_parsers.utils._insert_missing_rows import (
    _insert_missing_rows,
)
from databallpy.utils.utils import MISSING_INT


class TestInsertMissingRows(unittest.TestCase):
    def setUp(self):
        self.input = pd.DataFrame(
            {
                "frame": [0, 1, 2, 3, 4, 6, 9, 10, 15, 16, 17, 18, 21],
                "values_int": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "values_float": [1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "values_object": [
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                ],
                "values_datetime": ["2023-02-02 00:00:00"] * 13,
                "values_other": [[1]] * 13,
            }
        )
        self.input["values_datetime"] = pd.to_datetime(self.input["values_datetime"])
        self.expected_output = pd.DataFrame(
            {
                "frame": [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                ],
                "values_int": [
                    1,
                    1,
                    1,
                    1,
                    1,
                    MISSING_INT,
                    1,
                    MISSING_INT,
                    MISSING_INT,
                    1,
                    1,
                    MISSING_INT,
                    MISSING_INT,
                    MISSING_INT,
                    MISSING_INT,
                    1,
                    1,
                    1,
                    1,
                    MISSING_INT,
                    MISSING_INT,
                    1,
                ],
                "values_float": [
                    1,
                    1,
                    1,
                    1,
                    1,
                    np.nan,
                    1,
                    np.nan,
                    np.nan,
                    1,
                    1,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    1,
                    1,
                    1,
                    1,
                    np.nan,
                    np.nan,
                    1,
                ],
                "values_object": [
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    None,
                    "1",
                    None,
                    None,
                    "1",
                    "1",
                    None,
                    None,
                    None,
                    None,
                    "1",
                    "1",
                    "1",
                    "1",
                    None,
                    None,
                    "1",
                ],
                "values_datetime": [
                    pd.to_datetime("2023-02-02 00:00:00"),
                    pd.to_datetime("2023-02-02 00:00:00"),
                    pd.to_datetime("2023-02-02 00:00:00"),
                    pd.to_datetime("2023-02-02 00:00:00"),
                    pd.to_datetime("2023-02-02 00:00:00"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("2023-02-02 00:00:00"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("2023-02-02 00:00:00"),
                    pd.to_datetime("2023-02-02 00:00:00"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("2023-02-02 00:00:00"),
                    pd.to_datetime("2023-02-02 00:00:00"),
                    pd.to_datetime("2023-02-02 00:00:00"),
                    pd.to_datetime("2023-02-02 00:00:00"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("2023-02-02 00:00:00"),
                ],
                "values_other": [
                    [1],
                    [1],
                    [1],
                    [1],
                    [1],
                    None,
                    [1],
                    None,
                    None,
                    [1],
                    [1],
                    None,
                    None,
                    None,
                    None,
                    [1],
                    [1],
                    [1],
                    [1],
                    None,
                    None,
                    [1],
                ],
            }
        )

    def test_insert_missing_rows(self):
        output = _insert_missing_rows(self.input, "frame")
        pd.testing.assert_frame_equal(output, self.expected_output)

    def test_insert_missing_rows_with_selection(self):
        input_df = pd.DataFrame(
            {
                "frame": [0, 1, 4, 10],
                "values_int": [1, 1, 1, 1],
                "values_float": [1.0, 1.0, 1.0, 1.0],
            }
        )
        expected_output = pd.DataFrame(
            {
                "frame": [0, 1, 2, 3, 4, 5, 10],
                "values_int": [
                    1,
                    1,
                    MISSING_INT,
                    MISSING_INT,
                    1,
                    MISSING_INT,
                    1,
                ],
                "values_float": [1.0, 1.0, np.nan, np.nan, 1.0, np.nan, 1.0],
            }
        )
        output = _insert_missing_rows(input_df, "frame", selection=(1, 5))
        pd.testing.assert_frame_equal(output, expected_output)
