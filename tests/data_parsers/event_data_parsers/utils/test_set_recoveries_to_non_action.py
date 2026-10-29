import unittest

import numpy as np
import pandas as pd

from databallpy.data_parsers.event_data_parsers.utils import (
    _set_recoveries_to_non_action,
)


class TestSetRecoveriesToNonAction(unittest.TestCase):
    def test_set_recoveries_to_non_action(self):
        event_data = pd.DataFrame(
            {
                "period_id": [1, 1, 1],
                "start_x": [0.0, 10.0, 20.0],
                "start_y": [0.0, 0.0, 0.0],
                "end_x": [np.nan, 20.0, np.nan],
                "end_y": [np.nan, 0.0, np.nan],
                "action_type": ["dribble", "pass", "dribble"],
                "action_result": ["success"] * 3,
                "action_bodypart": ["foot"] * 3,
            }
        )
        is_recovery = pd.Series([True, False, True], index=event_data.index)

        res = _set_recoveries_to_non_action(event_data.copy(), is_recovery)

        self.assertListEqual(
            res["action_type"].tolist(), ["non_action", "pass", "non_action"]
        )
        self.assertTrue(
            res.loc[is_recovery, ["action_result", "action_bodypart"]]
            .isna()
            .all(axis=None)
        )
        # the locations and the other actions are not changed
        locations = ["start_x", "start_y", "end_x", "end_y"]
        pd.testing.assert_frame_equal(res[locations], event_data[locations])
        pd.testing.assert_frame_equal(
            res.loc[~is_recovery], event_data.loc[~is_recovery]
        )
