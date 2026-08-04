# Annealer
import math
import random
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from databallpy import Game
from databallpy.optimization.optimization import (
    Constraint,
    ObjectiveTerm,
    OptimizationAlgorithm,
    OptimizationResult,
)
from databallpy.utils.logging import create_logger

LOGGER = create_logger(__name__)


class SimulatedAnnealing(OptimizationAlgorithm):
    """Optimize the positions of defending players in a single tracking data frame
    using simulated annealing.

    See https://www.geeksforgeeks.org/artificial-intelligence/what-is-simulated-annealing/
    for a description of the annealing parameters.

    Args:
        game (Game): The game object whose tracking data is being optimized.
        selected_frame_idx (int): Index of the tracking data frame to optimize.
        objective_terms (list[ObjectiveTerm]): The objective terms whose weighted sum
            is maximized.
        weights (list[float]): The weight of each objective term. Must have the same
            length as ``objective_terms``.
        constraints (list[Constraint], optional): Constraints that a perturbed frame
            must satisfy to be accepted. Defaults to [].
        defending_players_to_optimize (list[str] | None, optional): Player ids of the
            players whose positions may be perturbed. If None, defaults to all players
            of the team that is not in possession. Defaults to None.
        distance_perturbation (float, optional): Maximum distance a player
            may be moved in each of the x and y directions per perturbation. Defaults
            to 0.2.
        num_iterations (int, optional): The maximum number of annealing iterations.
            Defaults to 1000.
        patience (int, optional): The number of iterations without improvement in the
            best score after which the search stops early. Defaults to 200.
        log_interval (int | None, optional): How often (in iterations) to log the best
            score. If None, logs roughly five times over the run. Defaults to None.
        random_state (int | None, optional): Seed for the random number generator.
            If None, the random number generator is not seeded. Defaults to None.
        verbose (bool, optional): Whether to show a progress bar and log progress.
            Defaults to True.
    """

    def __init__(
        self,
        game: Game,
        selected_frame_idx: int,
        objective_terms: list[ObjectiveTerm],
        weights: list[float],
        constraints: list[Constraint] = [],
        defending_players_to_optimize: list[str] | None = None,
        distance_perturbation: float = 0.2,
        num_iterations: int = 1000,
        patience: int = 200,
        log_interval: int | None = None,
        random_state: int | None = None,
        verbose: bool = True,
    ):
        super().__init__(
            game=game,
            selected_frame_idx=selected_frame_idx,
            objective_terms=objective_terms,
            constraints=constraints,
            weights=weights,
        )
        # annealing params
        self.distance_perturbation = distance_perturbation
        self.p_0 = 0.5
        self.T_0 = -100 / (math.log(self.p_0))
        self.T = self.T_0
        self.cooling_rate = 0.9
        self.num_iterations = num_iterations
        self.patience = patience
        # log the best score 5 times over the course of the run by default
        self.log_interval = (
            log_interval if log_interval is not None else max(1, num_iterations // 5)
        )
        self._rng = (
            random.Random(random_state) if random_state is not None else random.Random()
        )
        self.verbose = verbose
        self.defending_players_to_optimize = (
            defending_players_to_optimize
            if defending_players_to_optimize
            else self.game.get_column_ids(
                team="home" if self.frame["team_possession"] == "away" else "away"
            )
        )

    def perturbation(self, input_frame):
        """Perturb the position of a randomly chosen defending player and return the new frame.

        Args:
            input_frame (pd.Series): The frame to perturb.

        Returns:
            pd.Series: The perturbed frame.
        """
        proposed_new_frame = deepcopy(input_frame)
        # randomly choose 1 of the defenders
        self.selected_players = self._rng.sample(self.defending_players_to_optimize, 1)
        for player_id in self.selected_players:
            # move each player up to a maximal distance from their starting positions
            x_col = player_id + "_x"
            y_col = player_id + "_y"

            new_x_pos = proposed_new_frame[x_col] + self._rng.uniform(
                -self.distance_perturbation, self.distance_perturbation
            )
            new_y_pos = proposed_new_frame[y_col] + self._rng.uniform(
                -self.distance_perturbation, self.distance_perturbation
            )
            proposed_new_frame[y_col] = new_y_pos
            proposed_new_frame[x_col] = new_x_pos
            # check if the new position is reachable within max time to intercept (tti)
            if all(
                constraint.check(proposed_new_frame, player_id)
                for constraint in self.constraints
            ):
                return proposed_new_frame
            return input_frame

    def compute_objective(self, input_frame):
        objective_total = 0
        for i, objective_term in enumerate(self.objective_terms):
            score = objective_term.compute(input_frame)
            objective_total += score * self.weights[i]

        return objective_total

    def run(self):
        best_score = 0
        best_solution = deepcopy(self.frame)
        latest_frame = deepcopy(self.frame)
        last_checkpoint_score = 0
        t = self.T

        progress_iter = tqdm(
            range(1, self.num_iterations),
            desc="Simulated Annealing Optimization",
            leave=True,
            disable=not self.verbose,
        )

        for i in progress_iter:
            perturbed_frame = self.perturbation(latest_frame)
            new_score = self.compute_objective(perturbed_frame)
            # take the new result if it's better, or randomly take a worse result with decaying probability
            if (new_score > best_score) or math.exp(
                np.clip((new_score - best_score) / t, -700, 700)
            ) > self._rng.random():
                latest_frame = perturbed_frame
            if new_score > best_score:
                best_score = new_score
                best_solution = perturbed_frame
                latest_frame = perturbed_frame
            t = t * self.cooling_rate

            # logging
            if self.verbose and i % self.log_interval == 0:
                progress_iter.set_postfix(best_score=best_score)
                LOGGER.info(
                    "Iteration %s / %s, best_score=%s",
                    i,
                    self.num_iterations,
                    best_score,
                )

            # early stopping
            if i % self.patience == 0:
                if last_checkpoint_score == best_score:
                    LOGGER.warning(
                        "No improvement found in last %s iterations. Exiting early.",
                        self.patience,
                    )
                    break
                last_checkpoint_score = best_score

        LOGGER.info("Finished simulated annealing. Best score=%s", best_score)
        return OptimizationResult(best_frame=best_solution, best_result=best_score)
