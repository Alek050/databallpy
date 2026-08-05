import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from databallpy.optimization.objectives import (
    PressureObjective,
    WeightedPitchControlObjective,
)
from databallpy.optimization.optimization import ObjectiveType
from databallpy.utils.get_game import get_game

GRID_SIZE = (106, 68)


def _load_test_game():
    return get_game(
        tracking_data_loc="tests/test_data/tracab_td_test.dat",
        tracking_metadata_loc="tests/test_data/tracab_metadata_test.xml",
        tracking_data_provider="tracab",
        event_data_loc="tests/test_data/f24_test.xml",
        event_metadata_loc="tests/test_data/f7_test.xml",
        event_data_provider="opta",
        check_quality=False,
    )


class TestWeightedPitchControlObjective(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = _load_test_game()

    def setUp(self):
        self.game.get_column_ids = MagicMock(return_value=["home_1", "away_1"])
        self.frame = self.game.tracking_data.loc[1].copy()
        self.frame["team_possession"] = "home"
        self.xt_array = np.full(GRID_SIZE, 0.5)

    def test_init(self):
        influence = "databallpy.optimization.objectives.get_team_influence"

        # home in possession: home attacks, away defends, xt array is kept as-is
        with patch(influence, return_value=np.ones(GRID_SIZE)):
            objective = WeightedPitchControlObjective(
                self.game, self.frame, xt_array=self.xt_array
            )
        self.assertEqual(objective.computation_type, ObjectiveType.GRID)
        self.assertEqual(objective.attacking_team, "home")
        self.assertEqual(objective.defending_team, "away")
        np.testing.assert_array_equal(objective.xt_array, self.xt_array)

        # away in possession: teams swap and the xt array is flipped horizontally
        self.frame["team_possession"] = "away"
        with patch(influence, return_value=np.ones(GRID_SIZE)):
            objective = WeightedPitchControlObjective(
                self.game, self.frame, xt_array=self.xt_array
            )
        self.assertEqual(objective.attacking_team, "away")
        self.assertEqual(objective.defending_team, "home")
        np.testing.assert_array_equal(objective.xt_array, np.fliplr(self.xt_array))

        # no xt array given: the default model is loaded and resized to the grid
        self.frame["team_possession"] = "home"
        with (
            patch(influence, return_value=np.ones(GRID_SIZE)),
            patch(
                "databallpy.optimization.objectives.np.load",
                return_value=np.ones((264, 196)),
            ) as mock_load,
        ):
            objective = WeightedPitchControlObjective(self.game, self.frame)
        mock_load.assert_called_once()
        # the array shape is stored in (y, x) orientation, i.e. the transpose of GRID_SIZE
        self.assertEqual(objective.xt_array.shape, (GRID_SIZE[1], GRID_SIZE[0]))

    def test_compute(self):
        # run the real pipeline: default xt model, real grid and team influence
        frame = self.frame.copy()
        frame["team_possession"] = "home"
        frame["ball_x"], frame["ball_y"] = 0.0, 0.0
        players = {
            "home_34": (0.0, 0.0, 1.0, 0.0),
            "away_17": (5.0, 0.0, -1.0, 0.0),
        }
        for player, (x, y, vx, vy) in players.items():
            frame[f"{player}_x"], frame[f"{player}_y"] = x, y
            frame[f"{player}_vx"], frame[f"{player}_vy"] = vx, vy

        # defending (away) is resolved first in __init__, then attacking (home)
        self.game.get_column_ids = MagicMock(side_effect=[["away_17"], ["home_34"]])
        objective = WeightedPitchControlObjective(self.game, frame)

        result = objective.compute(frame)
        self.assertAlmostEqual(result, 64.57281218558984)


class TestPressureObjective(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = _load_test_game()

    def setUp(self):
        self.frame = self.game.tracking_data.loc[1].copy()
        self.frame["team_possession"] = "home"

    def test_init(self):
        # players_to_press defaults to the possessing team's players
        self.game.get_column_ids = MagicMock(return_value=["home_1", "home_2"])
        objective = PressureObjective(self.game, self.frame)
        self.assertEqual(objective.computation_type, ObjectiveType.PLAYER)
        self.assertIs(objective.game, self.game)
        self.assertEqual(objective.players_to_press, ["home_1", "home_2"])
        self.game.get_column_ids.assert_called_once_with(team="home")

        # an explicit players_to_press list is used as-is
        players = ["away_5", "away_9"]
        objective = PressureObjective(self.game, self.frame, players_to_press=players)
        self.assertEqual(objective.players_to_press, players)

    def test_compute(self):
        # pressure is the mean of the per-player pressure over the pressed players
        cases = [
            ("two_players", ["home_1", "home_2"], [2.0, 4.0], 3.0),
            ("single_player", ["home_1"], [7.5], 7.5),
            ("uniform_pressure", ["home_1", "home_2", "home_3"], [3.0, 3.0, 3.0], 3.0),
        ]
        for name, players, pressures, expected in cases:
            with self.subTest(name=name):
                objective = PressureObjective(
                    self.game, self.frame, players_to_press=players
                )
                with patch(
                    "databallpy.optimization.objectives.TrackingData"
                ) as mock_tracking_data:
                    instance = mock_tracking_data.return_value
                    instance.index = [self.frame.name]
                    instance.get_pressure_on_player.side_effect = pressures
                    result = objective.compute(self.frame)
                self.assertAlmostEqual(result, expected)
