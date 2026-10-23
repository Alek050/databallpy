import unittest

import numpy as np
import pandas as pd

from databallpy.data_parsers.event_data_parsers.utils import _add_recovery_dribbles


class TestAddRecoveryDribbles(unittest.TestCase):
    def test_add_recovery_dribbles(self):
        event_data = pd.DataFrame(
            {
                "period_id": [1, 1, 1, 1, 1, 1, 2, 2],
                "start_x": [0.0, 10.0, 2.0, 2.0, 20.0, 30.0, 0.0, 5.0],
                "start_y": [0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                "end_x": [np.nan] * 8,
                "end_y": [np.nan] * 8,
                "action_type": [
                    "dribble",
                    "pass",
                    "dribble",
                    "dribble",
                    "pass",
                    "dribble",
                    "pass",
                    "dribble",
                ],
                "action_result": ["success"] * 8,
                "action_bodypart": ["foot"] * 8,
            }
        )
        is_recovery = pd.Series(
            [True, False, True, True, False, True, False, True], index=event_data.index
        )

        res = _add_recovery_dribbles(event_data.copy(), is_recovery)

        # moved to the next action, skipping the other recovery
        self.assertEqual(res.loc[0, "action_type"], "dribble")
        self.assertListEqual(res.loc[0, ["end_x", "end_y"]].tolist(), [10.0, 0.0])
        self.assertListEqual(res.loc[2, ["end_x", "end_y"]].tolist(), [20.0, 0.0])
        self.assertListEqual(res.loc[3, ["end_x", "end_y"]].tolist(), [20.0, 0.0])
        # the next action is in the next period, or there is no next action
        for idx in [5, 7]:
            self.assertTrue(res.loc[idx, ["action_type", "action_result"]].isna().all())
        # other actions are not changed
        pd.testing.assert_frame_equal(
            res.loc[~is_recovery], event_data.loc[~is_recovery]
        )

    def test_add_recovery_dribbles_euclidean_distance(self):
        # 2.5 m in both directions is less than 3 m per axis, but more in distance
        event_data = pd.DataFrame(
            {
                "period_id": [1, 1, 1],
                "start_x": [0.0, 2.5, 4.0],
                "start_y": [0.0, 2.5, 3.0],
                "end_x": [np.nan] * 3,
                "end_y": [np.nan] * 3,
                "action_type": ["dribble", "pass", "dribble"],
                "action_result": ["success"] * 3,
                "action_bodypart": ["foot"] * 3,
            }
        )
        res = _add_recovery_dribbles(
            event_data, pd.Series([True, False, False], index=event_data.index)
        )
        self.assertEqual(res.loc[0, "action_type"], "dribble")
        self.assertListEqual(res.loc[0, ["end_x", "end_y"]].tolist(), [2.5, 2.5])

    def test_add_recovery_dribbles_not_moved(self):
        event_data = pd.DataFrame(
            {
                "period_id": [1, 1],
                "start_x": [0.0, 2.0],
                "start_y": [0.0, 2.0],
                "end_x": [np.nan] * 2,
                "end_y": [np.nan] * 2,
                "action_type": ["dribble", "pass"],
                "action_result": ["success"] * 2,
                "action_bodypart": ["foot"] * 2,
            }
        )
        res = _add_recovery_dribbles(
            event_data, pd.Series([True, False], index=event_data.index)
        )
        self.assertTrue(res.loc[0, ["action_type", "action_bodypart"]].isna().all())
