import unittest

import numpy as np
import pandas as pd

from databallpy.features.expected_values import (
    FREE_KICK_XT,
    OPEN_PLAY_XT,
    THROW_IN_XT,
    XG_PARAMS,
    get_shot_features,
    get_xt,
)
from databallpy.models.utils import get_xt_prediction, scale_and_predict_logreg


class TestGetXt(unittest.TestCase):
    def test_get_xt(self):
        event_data = pd.DataFrame(
            {
                "action_type": [
                    "pass",
                    "pass",
                    "shot_penalty",
                    "corner_crossed",
                    "goalkick",
                    "throw_in",
                    "freekick_short",
                    "dribble",
                    "tackle",
                    None,
                    "shot",
                ],
                "team_id": [1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1],
                "start_x": [
                    0.2,
                    20.0,
                    41.5,
                    -52.5,
                    -47.0,
                    10.0,
                    -30.0,
                    -20.0,
                    5.0,
                    5.0,
                    np.nan,
                ],
                "start_y": [0.1, 5.0, 0.0, 34.0, 0.0, 34.0, 10.0, -12.0, 5.0, 5.0, 1.0],
            },
            index=range(10, 21),
        )
        res = get_xt(event_data, home_team_id=1)

        def xt(x, y, grid):
            return get_xt_prediction(np.array([x]), np.array([y]), grid)[0]

        expected = pd.Series(
            [
                0.001,  # kick off
                xt(-20.0, -5.0, OPEN_PLAY_XT),  # away team, mirrored
                0.797,
                0.049,
                0.0,
                xt(10.0, 34.0, THROW_IN_XT),
                xt(30.0, -10.0, FREE_KICK_XT),
                xt(20.0, 12.0, OPEN_PLAY_XT),
                np.nan,  # no on-ball action
                np.nan,  # no action type
                np.nan,  # missing location
            ],
            index=range(10, 21),
            dtype=float,
        )
        pd.testing.assert_series_equal(res, expected)

    def test_get_xt_empty(self):
        event_data = pd.DataFrame(
            columns=["action_type", "team_id", "start_x", "start_y"]
        ).astype({"start_x": float, "start_y": float})
        res = get_xt(event_data, home_team_id=1)
        self.assertEqual(len(res), 0)
        self.assertEqual(res.dtype, float)


class TestGetShotFeatures(unittest.TestCase):
    def test_get_shot_features(self):
        shots = pd.DataFrame(
            {
                "team_id": [1, 2, 1, 1, 1, 1, 1],
                "start_x": [40.0, -40.0, 41.5, 30.0, 45.0, 40.0, np.nan],
                "start_y": [5.0, -5.0, 0.0, 3.0, 2.0, 5.0, 0.0],
                "action_type": [
                    "shot",
                    "shot",
                    "shot_penalty",
                    "shot_freekick",
                    "shot",
                    "shot",
                    "shot",
                ],
                "action_result": [
                    "fail",
                    "success",
                    "success",
                    "fail",
                    "fail",
                    "owngoal",
                    "fail",
                ],
                "action_bodypart": [
                    "foot_left",
                    None,
                    "foot",
                    "foot_right",
                    "head",
                    "foot",
                    "foot",
                ],
            },
            index=range(5, 12),
        )
        res = get_shot_features(shots, [105.0, 68.0], home_team_id=1)

        self.assertEqual(list(res.columns), ["ball_goal_distance", "shot_angle", "xg"])
        self.assertEqual(list(res.index), list(range(5, 12)))

        distance = np.hypot(12.5, 5.0)
        np.testing.assert_allclose(res["ball_goal_distance"].iloc[:2], distance)
        # the away team shoots at the other goal, so the features are mirrored
        np.testing.assert_allclose(res.iloc[0, :2], res.iloc[1, :2])

        def xg(row, model):
            features = res.loc[[row], ["ball_goal_distance", "shot_angle"]].to_numpy()
            return scale_and_predict_logreg(features, XG_PARAMS[model])[0]

        np.testing.assert_allclose(res.loc[5, "xg"], xg(5, "xg_by_foot"))
        # missing bodypart uses the foot model
        np.testing.assert_allclose(res.loc[6, "xg"], xg(6, "xg_by_foot"))
        self.assertEqual(res.loc[7, "xg"], 0.79)
        np.testing.assert_allclose(res.loc[8, "xg"], xg(8, "xg_by_free_kick"))
        np.testing.assert_allclose(res.loc[9, "xg"], xg(9, "xg_by_head"))
        self.assertEqual(res.loc[10, "xg"], 0.0)
        self.assertTrue(np.isnan(res.loc[11, "xg"]))

    def test_get_shot_features_empty(self):
        shots = pd.DataFrame(
            columns=[
                "team_id",
                "start_x",
                "start_y",
                "action_type",
                "action_result",
                "action_bodypart",
            ]
        )
        res = get_shot_features(shots, [105.0, 68.0], home_team_id=1)
        self.assertEqual(len(res), 0)
        self.assertEqual(list(res.columns), ["ball_goal_distance", "shot_angle", "xg"])
