import unittest

import numpy as np
import pandas as pd

from databallpy.load_data.tracking_data._insert_missing_rows import _insert_missing_rows


class TestInsertMissingRows(unittest.TestCase):
    def setUp(self):
        self.input = pd.DataFrame(
            {
                "frame": [0, 1, 2, 3, 4, 6, 9, 10, 15, 16, 17, 18, 21],
                "values": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
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
                "values": [
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
            }
        )

    def test_insert_missing_rows(self):
        output = _insert_missing_rows(self.input, "frame")
        pd.testing.assert_frame_equal(output, self.expected_output)
