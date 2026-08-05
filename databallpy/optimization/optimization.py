from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

from databallpy.game import Game
from databallpy.utils.logging import create_logger

LOGGER = create_logger(__name__)


class ObjectiveType(str, Enum):
    GRID = "grid"  # computed for each grid cell
    PLAYER = "player"  # computed for each player


class ObjectiveTerm:
    def __init__(self, computation_type: ObjectiveType):
        self.computation_type = computation_type

    def compute(self, input_frame: pd.Series) -> float:
        raise NotImplementedError


@dataclass
class OptimizationResult:
    """The result of running an optimization algorithm on a tracking data frame.

    Attributes:
        best_frame (pd.Series): The tracking data frame with the highest objective
            score found during the optimization.
        best_result (float): The objective score of ``best_frame``.
    """

    best_frame: pd.Series
    best_result: float


class Constraint(ABC):
    @abstractmethod
    def check(self, proposed_new_frame, player_id) -> bool:
        raise NotImplementedError


class OptimizationAlgorithm(ABC):
    """Abstract base class for algorithms that optimize a single tracking data frame.

    Subclasses must implement ``run``, which performs the search and returns an ``OptimizationResult``.

    Args:
        game (Game): The game whose tracking data is being optimized.
        selected_frame_idx (int): Index of the tracking data frame to optimize.
        objective_terms (list[ObjectiveTerm]): The objective objects the subclass will
            need to compute the objective score for the frame.
        weights (list[float]): The weight of each objective term. Must have the same
            length as ``objective_terms``.
        constraints (list[Constraint] | None, optional): Constraints that a proposed
            frame must satisfy. If None, no constraints are applied. Defaults to None.

    Raises:
        ValueError: If ``objective_terms`` and ``weights`` do not have equal length.
    """

    @abstractmethod
    def __init__(
        self,
        game: Game,
        selected_frame_idx: int,
        objective_terms: list[ObjectiveTerm],
        weights: list[float],
        constraints: list[Constraint] | None = None,
    ):
        if len(objective_terms) != len(weights):
            raise ValueError("objective_terms and weights must have equal length")

        self.game = game
        self.frame = game.tracking_data.loc[[selected_frame_idx]].iloc[0]
        self.constraints = constraints or []

        self.objective_terms = objective_terms
        self.weights = weights

    @abstractmethod
    def run(self) -> OptimizationResult:
        raise NotImplementedError


def optimize_tracking_frame(
    game: Game,
    selected_frame_idx: int,
    objective_terms: list[ObjectiveTerm],
    weights: list[float],
    constraints: list[Constraint],
    algorithm: type[OptimizationAlgorithm],
    **algorithm_kwargs: Any,
) -> OptimizationResult:
    """A wrapper function that runs an optimization algorithm on a single tracking data frame.
    Instantiates ``algorithm`` with the provided arguments and runs it.

    Args:
        game (Game): The game whose tracking data is being optimized.
        selected_frame_idx (int): Index of the tracking data frame to optimize.
        objective_terms (list[ObjectiveTerm]): The objective objects the ``algorithm`` will
            need to compute the objective score for the frame.
        weights (list[float]): The weight of each objective term. Must have the same
            length as ``objective_terms``.
        constraints (list[Constraint]): Constraints that a proposed frame must satisfy.
        algorithm (type[OptimizationAlgorithm]): The optimization algorithm class to
            instantiate and run.
        **algorithm_kwargs (Any): Additional keyword parameters for the ``algorithm`` constructor.

    Returns:
        OptimizationResult: The best frame found and its objective score.
    """
    LOGGER.info("Running optimization with %s", algorithm.__name__)

    optimizer = algorithm(
        game=game,
        selected_frame_idx=selected_frame_idx,
        objective_terms=objective_terms,
        weights=weights,
        constraints=constraints,
        **algorithm_kwargs,
    )
    return optimizer.run()
