import unittest
from unittest.mock import MagicMock

import pandas as pd

from databallpy.optimization.optimization import (
    Constraint,
    ObjectiveTerm,
    ObjectiveType,
    OptimizationAlgorithm,
    OptimizationResult,
    optimize_tracking_frame,
)
from databallpy.utils.get_game import get_game


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


class _ConstantObjective(ObjectiveTerm):
    def __init__(self, value: float):
        super().__init__(ObjectiveType.PLAYER)
        self.value = value

    def compute(self, input_frame: pd.Series) -> float:
        return self.value


class _DummyAlgorithm(OptimizationAlgorithm):
    def __init__(
        self,
        game,
        selected_frame_idx,
        objective_terms,
        weights,
        constraints=None,
    ):
        super().__init__(
            game=game,
            selected_frame_idx=selected_frame_idx,
            objective_terms=objective_terms,
            weights=weights,
            constraints=constraints,
        )

    def run(self):
        return OptimizationResult(best_frame=self.frame, best_result=0.0)


class TestOptimizationAlgorithm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = _load_test_game()

    def test_init(self):
        terms = [_ConstantObjective(1.0)]
        weights = [1.0]
        algorithm = _DummyAlgorithm(self.game, 1, terms, weights)

        # it stores its inputs and selects the requested frame from the tracking data
        self.assertIs(algorithm.game, self.game)
        self.assertEqual(algorithm.objective_terms, terms)
        self.assertEqual(algorithm.weights, weights)
        self.assertTrue(algorithm.frame.equals(self.game.tracking_data.loc[[1]].iloc[0]))

        # constraints default to an empty list when not provided
        self.assertEqual(algorithm.constraints, [])

        # objective_terms and weights must have equal length
        with self.assertRaises(ValueError):
            _DummyAlgorithm(self.game, 1, terms, weights=[1.0, 2.0])


class TestOptimizeTrackingFrame(unittest.TestCase):
    def test_runs_algorithm_and_returns_its_result(self):
        game = MagicMock()
        terms = [_ConstantObjective(1.0)]
        weights = [1.0]
        constraints = []
        expected_result = OptimizationResult(
            best_frame=pd.Series(dtype=float), best_result=42.0
        )

        algorithm = MagicMock()
        algorithm.__name__ = "SpyAlgorithm"  # optimize_tracking_frame logs this
        algorithm.return_value.run.return_value = expected_result

        result = optimize_tracking_frame(
            game=game,
            selected_frame_idx=3,
            objective_terms=terms,
            weights=weights,
            constraints=constraints,
            algorithm=algorithm,
            num_iterations=5,
            random_state=1,
        )

        # the extra kwargs must be forwarded to the algorithm constructor
        algorithm.assert_called_once_with(
            game=game,
            selected_frame_idx=3,
            objective_terms=terms,
            weights=weights,
            constraints=constraints,
            num_iterations=5,
            random_state=1,
        )
        algorithm.return_value.run.assert_called_once_with()
        self.assertIs(result, expected_result)


class TestAbstractNotImplemented(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = _load_test_game()

    def test_objective_term_compute_not_implemented(self):
        term = ObjectiveTerm(ObjectiveType.GRID)
        with self.assertRaises(NotImplementedError):
            term.compute(pd.Series(dtype=float))

    def test_constraint_check_not_implemented(self):
        # a subclass delegating to the base implementation hits Constraint.check
        class _SuperCallingConstraint(Constraint):
            def check(self, proposed_new_frame, player_id):
                return super().check(proposed_new_frame, player_id)

        with self.assertRaises(NotImplementedError):
            _SuperCallingConstraint().check(pd.Series(dtype=float), "home_1")

    def test_algorithm_run_not_implemented(self):
        # a subclass delegating to the base implementation hits OptimizationAlgorithm.run
        class _SuperCallingAlgorithm(OptimizationAlgorithm):
            def __init__(
                self,
                game,
                selected_frame_idx,
                objective_terms,
                weights,
                constraints=None,
            ):
                super().__init__(
                    game=game,
                    selected_frame_idx=selected_frame_idx,
                    objective_terms=objective_terms,
                    weights=weights,
                    constraints=constraints,
                )

            def run(self):
                return super().run()

        algorithm = _SuperCallingAlgorithm(
            self.game, 1, [_ConstantObjective(1.0)], [1.0]
        )
        with self.assertRaises(NotImplementedError):
            algorithm.run()
