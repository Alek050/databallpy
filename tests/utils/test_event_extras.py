import unittest

import numpy as np
import pandas as pd

from databallpy.features.expected_values import get_xt
from databallpy.utils.constants import MISSING_INT
from databallpy.utils.event_extras import (
    EXTRA_COLUMNS,
    create_all_event_extras,
    create_event_extras,
    get_event_extras_from_dataframe,
    get_event_type_dataframe,
)


class TestEventExtras(unittest.TestCase):
    def setUp(self):
        self.event_data = pd.DataFrame(
            {
                "event_id": [0, 1, 2],
                "databallpy_event": ["pass", "shot", None],
                "period_id": [1, 1, 1],
                "minutes": [0, 1, 2],
                "seconds": [1.0, 2.0, 3.0],
                "datetime": pd.to_datetime(
                    ["2023-01-01 20:00:01", "2023-01-01 20:01:02", "2023-01-01 20:02:03"]
                ),
                "start_x": [0.0, 40.0, -10.0],
                "start_y": [0.0, 5.0, 3.0],
                "end_x": [10.0, np.nan, np.nan],
                "end_y": [-5.0, np.nan, np.nan],
                "team_id": [1, 2, 1],
                "player_id": [11, 21, 12],
                "action_type": ["pass", "shot", "tackle"],
                "action_result": ["success", "fail", "success"],
                "action_bodypart": ["foot", "head", "foot"],
            }
        )
        self.players = pd.DataFrame({"id": [11, 12, 21], "shirt_num": [7, 8, 9]})
        self.pass_extras = create_event_extras(
            {
                0: {
                    "outcome": True,
                    "related_event_id": MISSING_INT,
                    "body_part": "right_foot",
                    "possession_type": "open_play",
                    "set_piece": "kick_off",
                    "outcome_str": "successful",
                    "pass_type": "long_ball",
                    "receiver_player_id": 12,
                }
            },
            "pass",
        )
        self.shot_extras = create_event_extras(
            {
                1: {
                    "outcome": False,
                    "related_event_id": MISSING_INT,
                    "body_part": "head",
                    "possession_type": "open_play",
                    "set_piece": "no_set_piece",
                    "outcome_str": "miss_off_target",
                }
            },
            "shot",
        )

    def test_create_event_extras(self):
        self.assertEqual(list(self.pass_extras.columns), EXTRA_COLUMNS["pass"])
        self.assertEqual(self.pass_extras.loc[0, "receiver_player_id"], 12)

        # missing extras get the default values
        self.assertEqual(list(self.shot_extras.columns), EXTRA_COLUMNS["shot"])
        self.assertTrue(np.isnan(self.shot_extras.loc[0, "y_target"]))
        self.assertTrue(np.isnan(self.shot_extras.loc[0, "z_target"]))
        self.assertFalse(self.shot_extras.loc[0, "first_touch"])

        empty = create_event_extras({}, "dribble")
        self.assertEqual(list(empty.columns), EXTRA_COLUMNS["dribble"])
        self.assertEqual(len(empty), 0)
        self.assertEqual(empty["event_id"].dtype, "int64")

    def test_create_all_event_extras(self):
        res = create_all_event_extras({"pass_events": {0: {"outcome": True}}})
        self.assertEqual(
            list(res.keys()), ["shot_events", "pass_events", "dribble_events"]
        )
        self.assertEqual(len(res["pass_events"]), 1)
        self.assertEqual(len(res["shot_events"]), 0)
        self.assertEqual(len(res["dribble_events"]), 0)
        self.assertEqual(res["pass_events"].loc[0, "receiver_player_id"], MISSING_INT)

    def test_get_event_extras_from_dataframe(self):
        pass_events = get_event_type_dataframe(
            self.event_data, self.pass_extras, "pass", 1, self.players, [105, 68]
        )
        pass_events.index = [5]
        res = get_event_extras_from_dataframe(pass_events, "pass")
        pd.testing.assert_frame_equal(res, self.pass_extras)

        # missing columns are added as NaN
        res = get_event_extras_from_dataframe(
            pd.DataFrame({"event_id": [3], "outcome": [False]}), "dribble"
        )
        self.assertEqual(list(res.columns), EXTRA_COLUMNS["dribble"])
        self.assertTrue(res["duel_type"].isna().all())

        for empty in [None, pd.DataFrame()]:
            res = get_event_extras_from_dataframe(empty, "shot")
            self.assertEqual(list(res.columns), EXTRA_COLUMNS["shot"])
            self.assertEqual(len(res), 0)
            self.assertEqual(res["event_id"].dtype, "int64")

    def test_get_event_type_dataframe_pass(self):
        res = get_event_type_dataframe(
            self.event_data, self.pass_extras, "pass", 1, self.players, [105, 68]
        )
        self.assertEqual(
            list(res.columns),
            [
                "event_id",
                "period_id",
                "minutes",
                "seconds",
                "datetime",
                "start_x",
                "start_y",
                "team_id",
                "team_side",
                "player_id",
                "jersey",
                "outcome",
                "related_event_id",
                "xt",
                "body_part",
                "possession_type",
                "set_piece",
                "outcome_str",
                "end_x",
                "end_y",
                "pass_type",
                "receiver_player_id",
            ],
        )
        self.assertEqual(len(res), 1)
        self.assertEqual(res.loc[0, "team_side"], "home")
        self.assertEqual(res.loc[0, "jersey"], 7)
        self.assertEqual(res.loc[0, "end_x"], 10.0)
        # xt is computed when the event data has no xt column
        self.assertEqual(res.loc[0, "xt"], 0.001)

    def test_get_event_type_dataframe_shot(self):
        event_data = self.event_data.copy()
        event_data["xt"] = [0.1, 0.2, np.nan]
        res = get_event_type_dataframe(
            event_data, self.shot_extras, "shot", 1, self.players, [105, 68]
        )
        self.assertEqual(len(res), 1)
        self.assertEqual(res.loc[0, "team_side"], "away")
        self.assertEqual(res.loc[0, "jersey"], 9)
        self.assertEqual(res.loc[0, "xt"], 0.2)
        self.assertEqual(
            list(res.columns[-3:]), ["ball_goal_distance", "shot_angle", "xg"]
        )
        # the away team shoots at the goal on the left
        np.testing.assert_allclose(
            res.loc[0, "ball_goal_distance"], np.hypot(52.5 + 40.0, 5.0)
        )
        self.assertNotIn("end_x", res.columns)

    def test_get_event_type_dataframe_empty(self):
        empty_extras = create_event_extras({}, "dribble")
        res = get_event_type_dataframe(
            self.event_data, empty_extras, "dribble", 1, self.players, [105, 68]
        )
        pd.testing.assert_frame_equal(res, pd.DataFrame())

        res = get_event_type_dataframe(
            self.event_data.iloc[0:0], self.pass_extras, "pass", 1, self.players, [1, 1]
        )
        pd.testing.assert_frame_equal(res, pd.DataFrame())

    def test_get_event_type_dataframe_xt_matches_get_xt(self):
        res = get_event_type_dataframe(
            self.event_data, self.shot_extras, "shot", 1, self.players, [105, 68]
        )
        expected = get_xt(self.event_data, 1).loc[1]
        self.assertAlmostEqual(res.loc[0, "xt"], expected)
