import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    import pandera.pandas as pa
except ModuleNotFoundError:
    import pandera as pa

from databallpy.events.spadl import SPADL_ACTION_COLUMNS
from databallpy.schemas import EventData
from databallpy.utils.constants import MISSING_INT, SPADL_ACTION_TYPES
from databallpy.utils.get_game import get_game
from databallpy.utils.warnings import DataBallPyWarning

PITCH_DIMENSIONS = [100.0, 50.0]


def _get_event_data():
    return EventData(
        {
            "event_id": [0, 1, 2, 3, 4, 5],
            "original_event_id": ["a", "b", "c", "d", "e", "f"],
            "period_id": [1, 1, 1, 1, 1, 2],
            "minutes": [0, 0, 0, 0, 0, 45],
            "seconds": [10.0, 14.0, 15.0, 16.0, 20.0, 5.0],
            "team_id": [1, 1, 2, 2, 2, 1],
            "player_id": [11, 12, 21, 22, 23, 13],
            "start_x": [0.0, 20.0, 0.0, -30.0, 0.0, 40.0],
            "start_y": [0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
            "end_x": [10.0, 30.0, np.nan, np.nan, np.nan, 50.0],
            "end_y": [0.0, 10.0, np.nan, np.nan, np.nan, 0.0],
            "action_type": ["pass", "pass", "non_action", "clearance", None, "shot"],
            "action_result": ["success", "success", None, "success", None, "fail"],
            "action_bodypart": ["foot", "foot_left", None, "head", None, "head"],
        },
        provider="test",
    )


class TestToSpadl(unittest.TestCase):
    def setUp(self):
        self.event_data = _get_event_data()

    def test_to_spadl_spadl_coordinates(self):
        res = self.event_data.to_spadl(
            home_team_id=1,
            pitch_dimensions=PITCH_DIMENSIONS,
            game_id=99,
            coordinate_system="spadl",
        )
        expected = pd.DataFrame(
            {
                "game_id": [99] * 5,
                "original_event_id": ["a", None, "b", "d", "f"],
                "event_id": pd.array([0, pd.NA, 1, 3, 5], dtype="Int64"),
                "action_id": [0, 1, 2, 3, 4],
                "period_id": [1, 1, 1, 1, 2],
                "time_seconds": [10.0, 12.0, 14.0, 16.0, 5.0],
                "team_id": [1, 1, 1, 2, 1],
                "player_id": [11, 12, 12, 22, 13],
                "start_x": [52.5, 63.0, 73.5, 21.0, 94.5],
                "start_y": [34.0, 34.0, 34.0, 47.6, 34.0],
                # the clearance ends at the start of the next action
                "end_x": [63.0, 73.5, 84.0, 94.5, 105.0],
                "end_y": [34.0, 34.0, 47.6, 34.0, 34.0],
                "type_id": [0, 21, 0, 18, 11],
                "result_id": [1, 1, 1, 1, 0],
                "bodypart_id": [0, 0, 4, 1, 1],
            }
        )
        expected[["game_id", "original_event_id", "team_id", "player_id"]] = expected[
            ["game_id", "original_event_id", "team_id", "player_id"]
        ].astype(object)

        self.assertListEqual(list(res.columns), SPADL_ACTION_COLUMNS)
        pd.testing.assert_frame_equal(res, expected, check_dtype=False)

    def test_to_spadl_databallpy_coordinates(self):
        res = self.event_data.to_spadl(home_team_id=1, pitch_dimensions=PITCH_DIMENSIONS)
        spadl = self.event_data.to_spadl(
            home_team_id=1, pitch_dimensions=PITCH_DIMENSIONS, coordinate_system="spadl"
        )

        np.testing.assert_allclose(res["start_x"], [0.0, 10.0, 20.0, -30.0, 40.0])
        np.testing.assert_allclose(res["start_y"], [0.0, 0.0, 0.0, 10.0, 0.0])
        np.testing.assert_allclose(res["end_x"], [10.0, 20.0, 30.0, 40.0, 50.0])
        np.testing.assert_allclose(res["end_y"], [0.0, 0.0, 10.0, 0.0, 0.0])
        cols = [c for c in SPADL_ACTION_COLUMNS if c[-2:] not in ("_x", "_y")]
        pd.testing.assert_frame_equal(res[cols], spadl[cols])
        self.assertTrue(res["game_id"].isna().all())

    def test_to_spadl_playing_direction(self):
        home = self.event_data.to_spadl(
            home_team_id=1, pitch_dimensions=PITCH_DIMENSIONS
        )
        cols = ["start_x", "start_y", "end_x", "end_y"]
        is_home = home["team_id"] == 1
        for direction, to_flip in [
            ("away_oriented", pd.Series(True, index=home.index)),
            ("possession_oriented", ~is_home),
            ("out_of_possession_oriented", is_home),
        ]:
            with self.subTest(direction=direction):
                res = self.event_data.to_spadl(
                    home_team_id=1,
                    pitch_dimensions=PITCH_DIMENSIONS,
                    playing_direction=direction,
                )
                expected = home[cols].copy()
                expected.loc[to_flip] *= -1
                pd.testing.assert_frame_equal(res[cols], expected)

    def test_to_spadl_team_oriented_deprecated(self):
        with self.assertWarns(DataBallPyWarning):
            res = self.event_data.to_spadl(
                home_team_id=1,
                pitch_dimensions=PITCH_DIMENSIONS,
                playing_direction="team_oriented",
            )
        pd.testing.assert_frame_equal(
            res,
            self.event_data.to_spadl(home_team_id=1, pitch_dimensions=PITCH_DIMENSIONS),
        )

    def test_to_spadl_wrong_input(self):
        with self.assertRaises(ValueError):
            self.event_data.to_spadl(
                home_team_id=1,
                pitch_dimensions=PITCH_DIMENSIONS,
                playing_direction="wrong",
            )
        with self.assertRaises(ValueError):
            self.event_data.to_spadl(
                home_team_id=1,
                pitch_dimensions=PITCH_DIMENSIONS,
                coordinate_system="wrong",
            )

    def test_to_spadl_missing_values(self):
        for col, value in [
            ("start_x", np.nan),
            ("team_id", None),
            ("team_id", MISSING_INT),
            ("player_id", None),
            ("player_id", MISSING_INT),
        ]:
            with self.subTest(col=col, value=value):
                event_data = self.event_data.copy()
                event_data["team_id"] = event_data["team_id"].astype(object)
                event_data["player_id"] = event_data["player_id"].astype(object)
                event_data.loc[0, col] = value
                with self.assertWarnsRegex(DataBallPyWarning, "1 action"):
                    res = event_data.to_spadl(
                        home_team_id=1, pitch_dimensions=PITCH_DIMENSIONS
                    )
                self.assertNotIn(0, res["event_id"].dropna().tolist())
                self.assertFalse(res["player_id"].isna().any())

        # actions with a type but without a result are invalid SPADL
        event_data = self.event_data.copy()
        event_data.loc[0, "action_result"] = None
        with self.assertRaises(pa.errors.SchemaError):
            event_data.to_spadl(home_team_id=1, pitch_dimensions=PITCH_DIMENSIONS)

    def test_to_spadl_dribble_rules(self):
        dribble_id = SPADL_ACTION_TYPES.index("dribble")
        cases = {
            "too_late": ("seconds", 1, 25.0),
            "other_team": ("team_id", 1, 2),
            "same_location": ("start_x", 1, 10.0),
            "other_period": ("period_id", 1, 2),
            "next_is_foul": ("action_type", 1, "foul"),
        }
        for name, (col, idx, value) in cases.items():
            with self.subTest(case=name):
                event_data = self.event_data.copy()
                event_data.loc[idx, col] = value
                if name == "other_period":
                    # drop the period 2 shot, which would add a dribble to the pass
                    event_data.loc[idx, "minutes"] = 45
                    event_data = event_data.drop(index=5)
                res = event_data.to_spadl(
                    home_team_id=1, pitch_dimensions=PITCH_DIMENSIONS
                )
                self.assertNotIn(dribble_id, res["type_id"].tolist())

    def test_to_spadl_no_actions(self):
        event_data = self.event_data.copy()
        event_data["action_type"] = None
        res = event_data.to_spadl(home_team_id=1, pitch_dimensions=PITCH_DIMENSIONS)
        self.assertEqual(len(res), 0)
        self.assertListEqual(list(res.columns), SPADL_ACTION_COLUMNS)


class TestGameToSpadl(unittest.TestCase):
    def test_game_to_spadl(self):
        game = get_game(
            event_data_loc="tests/test_data/f24_test.xml",
            event_metadata_loc="tests/test_data/f7_test.xml",
            event_data_provider="opta",
        )
        with patch.object(EventData, "to_spadl", return_value="spadl") as mock:
            res = game.to_spadl(game_id=5, playing_direction="possession_oriented")

        self.assertEqual(res, "spadl")
        mock.assert_called_once_with(
            home_team_id=game.home_team_id,
            pitch_dimensions=game.pitch_dimensions,
            game_id=5,
            playing_direction="possession_oriented",
            coordinate_system="databallpy",
        )
