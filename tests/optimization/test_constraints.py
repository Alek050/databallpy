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
        self.assertEqual(self.constraint.turning_penalty, "shaw")
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
            (
                # cosine rounds to 1.0000000000000002, see issue #401
                "destination_on_line_of_travel",
                np.array([-34.11000061035156, -5.53000020980835]),
                np.array([2.524995803833008, -1.0500001907348633]),
                np.array([-34.11000061035156, -5.53000020980835])
                + 0.7 * np.array([2.524995803833008, -1.0500001907348633]),
                0.1
                + 0.6
                * np.linalg.norm(np.array([2.524995803833008, -1.0500001907348633]))
                / 5.0,
            ),
        ]
        # The turning term is zero in all these cases, so both options agree.
        for turning_penalty in ["shaw", "pleuler"]:
            constraint = self._constraint(turning_penalty)
            for name, origin, velocity, destination, expected in cases:
                with self.subTest(name=name, turning_penalty=turning_penalty):
                    tti = constraint.tti(origin, destination, velocity)
                    self.assertAlmostEqual(tti, expected)

    def _constraint(self, turning_penalty, max_time_to_intercept_seconds=1.0):
        return TTIConstraint(
            self.game,
            self.frame,
            max_time_to_intercept_seconds=max_time_to_intercept_seconds,
            reaction_time=0.1,
            max_velocity=5.0,
            turning_penalty=turning_penalty,
        )

    def test_turning_penalty_accepts_any_capitalization(self):
        for value, expected in [
            ("shaw", "shaw"),
            ("Shaw", "shaw"),
            ("SHAW", "shaw"),
            ("pleuler", "pleuler"),
            ("Pleuler", "pleuler"),
            ("PLEULER", "pleuler"),
        ]:
            with self.subTest(value=value):
                self.assertEqual(self._constraint(value).turning_penalty, expected)

    def test_turning_penalty_invalid_value(self):
        for value in ["spearman", "", None, 1]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as cm:
                    self._constraint(value)
                self.assertIn('"shaw"', str(cm.exception))
                self.assertIn('"pleuler"', str(cm.exception))

    def test_tti_turning_penalty_hand_computed(self):
        # reaction_time 0.1 s and max_velocity 5 m/s; the player starts at the origin.
        # A velocity of (5, 0) moves the player to (0.5, 0) during the reaction time.
        cases = [
            (
                # destination (0, 1.2), at 90 degrees to the direction of travel:
                # shaw = 0.1 + |(0, 1.2) - (0.5, 0)| / 5 = 0.1 + 1.3 / 5 = 0.36
                # pleuler = 5 * (pi / 2) / pi + 0.36 = 2.5 + 0.36 = 2.86
                "destination_to_the_side",
                np.array([5.0, 0.0]),
                np.array([0.0, 1.2]),
                0.36,
                2.86,
            ),
            (
                # destination (-2, 0), directly behind the player:
                # shaw = 0.1 + |(-2, 0) - (0.5, 0)| / 5 = 0.1 + 2.5 / 5 = 0.6
                # pleuler = 5 * pi / pi + 0.6 = 5.6
                "destination_behind",
                np.array([5.0, 0.0]),
                np.array([-2.0, 0.0]),
                0.6,
                5.6,
            ),
            (
                # destination (10, 0), straight ahead: no turning term
                # shaw = pleuler = 0.1 + |(10, 0) - (0.5, 0)| / 5 = 0.1 + 9.5 / 5 = 2.0
                "destination_straight_ahead",
                np.array([5.0, 0.0]),
                np.array([10.0, 0.0]),
                2.0,
                2.0,
            ),
            (
                # stationary player: no turning term
                # shaw = pleuler = 0.1 + |(3, 4)| / 5 = 0.1 + 5 / 5 = 1.1
                "stationary_player",
                np.array([0.0, 0.0]),
                np.array([3.0, 4.0]),
                1.1,
                1.1,
            ),
        ]
        origin = np.array([0.0, 0.0])
        shaw = self._constraint("shaw")
        pleuler = self._constraint("pleuler")
        for name, velocity, destination, expected_shaw, expected_pleuler in cases:
            with self.subTest(name=name):
                tti_shaw = shaw.tti(origin, destination, velocity)
                tti_pleuler = pleuler.tti(origin, destination, velocity)
                self.assertAlmostEqual(tti_shaw, expected_shaw)
                self.assertAlmostEqual(tti_pleuler, expected_pleuler)
                if name == "destination_behind":
                    self.assertGreater(tti_pleuler, tti_shaw)
                if name in ["destination_straight_ahead", "stationary_player"]:
                    self.assertAlmostEqual(tti_pleuler, tti_shaw)

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

    def test_check_turning_penalty(self):
        # A player moving at 5 m/s along x, with a proposed position 2 m behind them:
        # 0.6 s with "shaw" (reachable within 1 s), 5.6 s with "pleuler" (not).
        frame = deepcopy(self.frame)
        frame[f"{self.player_id}_vx"] = 5.0
        proposed_frame = deepcopy(frame)
        proposed_frame[f"{self.player_id}_x"] = -2.0
        for turning_penalty, expected in [("shaw", True), ("pleuler", False)]:
            with self.subTest(turning_penalty=turning_penalty):
                constraint = TTIConstraint(
                    self.game,
                    frame,
                    max_time_to_intercept_seconds=1.0,
                    reaction_time=0.1,
                    max_velocity=5.0,
                    turning_penalty=turning_penalty,
                )
                self.assertEqual(
                    constraint.check(proposed_frame, self.player_id), expected
                )
