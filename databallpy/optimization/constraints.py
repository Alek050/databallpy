import numpy as np
import pandas as pd

from databallpy.game import Game
from databallpy.optimization.optimization import Constraint


class TTIConstraint(Constraint):
    """Constraint that limits how far a player may be moved by an optimization algorithm
    based on their time to intercept (TTI) the proposed position.

    A proposed position is only allowed if the player can reach it from their starting
    position and velocity within ``max_time_to_intercept_seconds``. The time to
    intercept accounts for the player's reaction time, the turn required to change
    direction, and travel at ``max_velocity``.

    Args:
        game (Game): The game whose tracking data is being optimized.
        frame (pd.Series): The initial tracking data frame.
        max_time_to_intercept_seconds (float, optional): The maximum time, in seconds,
            a player is allowed to take to reach a proposed position. Defaults to 1.
        reaction_time (float, optional): The player's reaction time in seconds before
            starting to move. Defaults to 0.7.
        max_velocity (float, optional): The maximum player velocity in meters per
            second used to estimate travel time. Defaults to 5.0.
    """

    def __init__(
        self,
        game: Game,
        frame: pd.Series,
        max_time_to_intercept_seconds: float = 1,
        reaction_time: float = 0.7,
        max_velocity: float = 5.0,
    ):
        self.max_time_to_intercept_seconds = max_time_to_intercept_seconds
        self.reaction_time = reaction_time
        self.max_velocity = max_velocity

        self.player_to_starting_pos_and_vel_map = frame[
            [c + "_x" for c in game.get_column_ids()]
            + [c + "_y" for c in game.get_column_ids()]
            + [c + "_vx" for c in game.get_column_ids()]
            + [c + "_vy" for c in game.get_column_ids()]
        ]

    # TTI Implementation from https://github.com/devinpleuler/analytics-handbook/blob/master/soccer_analytics_handbook.ipynb
    def tti(self, origin, destination, velocity):
        u = (origin + velocity) - origin
        v = destination - origin
        u_mag = np.sqrt(np.sum(u**2, axis=-1))
        v_mag = np.sqrt(np.sum(v**2, axis=-1))
        dot_product = np.sum(u * v, axis=-1)

        denom = u_mag * v_mag
        # stationary player edge case
        if denom == 0:
            angle = 0.0
        else:
            angle = np.arccos(dot_product / denom)
        r_reaction = origin + velocity * self.reaction_time
        d = destination - r_reaction
        t = (
            u_mag * angle / np.pi
            + self.reaction_time
            + np.linalg.norm(d, axis=-1) / self.max_velocity
        )

        return t

    def check(self, proposed_new_frame, player_id) -> bool:
        x_col, y_col, vx_col, vy_col = [
            player_id + suffix for suffix in ["_x", "_y", "_vx", "_vy"]
        ]
        origin = np.array(
            [
                self.player_to_starting_pos_and_vel_map[x_col],
                self.player_to_starting_pos_and_vel_map[y_col],
            ]
        )
        velocity = np.array(
            [
                self.player_to_starting_pos_and_vel_map[vx_col],
                self.player_to_starting_pos_and_vel_map[vy_col],
            ]
        )
        destination = np.array([proposed_new_frame[x_col], proposed_new_frame[y_col]])

        return (
            self.tti(origin, destination, velocity) < self.max_time_to_intercept_seconds
        )
