import unittest
from copy import deepcopy
from unittest.mock import MagicMock

import numpy as np

from databallpy.optimization.constraints import TTIConstraint
from databallpy.utils.get_game import get_game


class TestTTIConstraint(unittest.TestCase):
    def setUp(self):
        self.game = get_game(
            tracking_data_loc="tests/test_data/tracab_td_test.dat",
            tracking_metadata_loc="tests/test_data/tracab_metadata_test.xml",
            tracking_data_provider="tracab",
            event_data_loc="tests/test_data/f24_test.xml",
            event_metadata_loc="tests/test_data/f7_test.xml",
            event_data_provider="opta",
            check_quality=False,
        )
        self.player_ids = ["home_34", "away_17"]
        self.game.get_column_ids = MagicMock(return_value=self.player_ids)

        self.frame = self.game.tracking_data.loc[1].copy()
        for player_id in self.player_ids:
            self.frame[f"{player_id}_x"] = 0.0
            self.frame[f"{player_id}_y"] = 0.0
            self.frame[f"{player_id}_vx"] = 0.0
            self.frame[f"{player_id}_vy"] = 0.0

        self.player_id = self.player_ids[0]
        self.constraint = TTIConstraint(
            self.game,
            self.frame,
            max_time_to_intercept_seconds=1.0,
            reaction_time=0.1,
            max_velocity=5.0,
        )

    def test_init_stores_parameters(self):
        self.assertEqual(self.constraint.max_time_to_intercept_seconds, 1.0)
        self.assertEqual(self.constraint.reaction_time, 0.1)
        self.assertEqual(self.constraint.max_velocity, 5.0)
        self.assertIn(
            "home_34_x", self.constraint.player_to_starting_pos_and_vel_map.index
        )
        self.assertIn(
            "away_17_vy", self.constraint.player_to_starting_pos_and_vel_map.index
        )

    def test_tti(self):
        cases = [
            (
                "stationary_player",
                np.array([0.0, 0.0]),
                np.array([0.0, 0.0]),
                np.array([5.0, 0.0]),
                0.1 + 5.0 / 5.0,
            ),
            (
                "moving_player",
                np.array([0.0, 0.0]),
                np.array([1.0, 0.0]),
                np.array([10.0, 0.0]),
                0.1 + 9.9 / 5.0,
            ),
        ]
        for name, origin, velocity, destination, expected in cases:
            with self.subTest(name=name):
                tti = self.constraint.tti(origin, destination, velocity)
                self.assertAlmostEqual(tti, expected)

    def test_check(self):
        cases = [
            ("same_position", {}, 1.0, True),
            ("reachable_move", {"_x": 2.0}, 1.0, True),
            ("unreachable_move", {"_x": 100.0, "_y": 100.0}, 1.0, False),
            ("exceeds_max_time_to_intercept", {"_x": 2.0}, 0.2, False),
        ]
        for name, overrides, max_tti, expected in cases:
            with self.subTest(name=name):
                constraint = TTIConstraint(
                    self.game,
                    self.frame,
                    max_time_to_intercept_seconds=max_tti,
                    reaction_time=0.1,
                    max_velocity=5.0,
                )
                proposed_frame = deepcopy(self.frame)
                for suffix, value in overrides.items():
                    proposed_frame[f"{self.player_id}{suffix}"] = value
                self.assertEqual(
                    constraint.check(proposed_frame, self.player_id), expected
                )
