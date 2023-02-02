import unittest

import pandas as pd

from databallpy.load_data.tracking_data._insert_missing_rows import _insert_missing_rows

class TestTracab(unittest.TestCase):
    def setUp(self):
        self.input = pd.DataFrame(
            {
                "timestamp": [0,1,2,3,4,6,9,10,15,16,17,18,21]
            }
        )
        self.expected_output = pd.DataFrame(
            {
                "timestamp": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
            }
        )

    def test_insert_missing_rows(self):
        output = _insert_missing_rows(self.input, "timestamp")
        pd.testing.assert_frame_equal(output, self.expected_output)