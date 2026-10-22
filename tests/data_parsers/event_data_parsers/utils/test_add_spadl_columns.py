import unittest

import pandas as pd

from databallpy.data_parsers.event_data_parsers.utils import _add_spadl_columns


class TestAddSpadlColumns(unittest.TestCase):
    def test_add_spadl_columns(self):
        event_data = pd.DataFrame({"event_id": [0, 1, 2]}, index=[5, 6, 7])
        res = _add_spadl_columns(event_data)

        self.assertListEqual(
            list(res.columns),
            ["event_id", "action_type", "action_result", "action_bodypart"],
        )
        for col in ["action_type", "action_result", "action_bodypart"]:
            self.assertEqual(res[col].dtype, object)
            self.assertTrue(res[col].isna().all())

    def test_add_spadl_columns_keeps_existing(self):
        event_data = pd.DataFrame(
            {"event_id": [0, 1], "action_type": ["pass", None]}, index=[3, 4]
        )
        res = _add_spadl_columns(event_data)

        pd.testing.assert_series_equal(
            res["action_type"],
            pd.Series(["pass", None], index=[3, 4], name="action_type"),
        )
        self.assertTrue(res["action_result"].isna().all())
        self.assertTrue(res["action_bodypart"].isna().all())
