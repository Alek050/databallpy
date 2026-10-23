import numpy as np
import pandas as pd

from databallpy.game import Game
from databallpy.optimization.optimization import Constraint


class TTIConstraint(Constraint):
    """Constraint that limits how far a player may be moved by an optimization algorithm
    based on their time to intercept (TTI) the proposed position.

    A proposed position is only allowed if the player can reach it from their starting
    position and velocity within ``max_time_to_intercept_seconds``. The time to
    intercept accounts for the player's reaction time and travel at ``max_velocity``,
    and, with ``turning_penalty="pleuler"``, for the turn required to change
    direction.

    Two time to intercept estimates are available through ``turning_penalty``:

    - ``"shaw"`` (default): Laurie Shaw's approximation, ``simple_time_to_intercept``
      in ``Metrica_PitchControl.py`` of the LaurieOnTracking repository
      (https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking). The player
      keeps moving with their current velocity for ``reaction_time`` seconds and then
      runs straight to the destination at ``max_velocity``::

          t = reaction_time
              + |destination - (origin + velocity * reaction_time)| / max_velocity

    - ``"pleuler"``: Devin Pleuler's modification of Shaw's approximation, from the
      "Time to Intercept" section of his soccer analytics handbook
      (https://github.com/devinpleuler/analytics-handbook). It adds a turning term to
      Shaw's time::

          t = |velocity| * angle / pi + (Shaw's time)

      where ``angle`` (in radians) is the angle between the player's velocity and the
      direction from the player to the destination.

    The difference is the turning term. It is zero for a stationary player and for a
    destination straight ahead, and grows with the player's speed and with how far the
    destination lies away from the direction of travel: for a destination directly
    behind a player moving at 5 m/s it adds 5 seconds. Destinations away from a moving
    player's direction therefore cost more with ``"pleuler"`` than with ``"shaw"``.

    Args:
        game (Game): The game whose tracking data is being optimized.
        frame (pd.Series): The initial tracking data frame.
        max_time_to_intercept_seconds (float, optional): The maximum time, in seconds,
            a player is allowed to take to reach a proposed position. Defaults to 1.
        reaction_time (float, optional): The player's reaction time in seconds before
            starting to move. Defaults to 0.7.
        max_velocity (float, optional): The maximum player velocity in meters per
            second used to estimate travel time. Defaults to 5.0.
        turning_penalty (str, optional): The time to intercept estimate, ``"shaw"``
            (no turning term) or ``"pleuler"`` (with the turning term), matched without
            regard to capitalization. Defaults to ``"shaw"``.

    Raises:
        ValueError: If ``turning_penalty`` is not ``"shaw"`` or ``"pleuler"``.
    """

    def __init__(
        self,
        game: Game,
        frame: pd.Series,
        max_time_to_intercept_seconds: float = 1,
        reaction_time: float = 0.7,
        max_velocity: float = 5.0,
        turning_penalty: str = "shaw",
    ):
        if not isinstance(turning_penalty, str) or turning_penalty.lower() not in (
            "shaw",
            "pleuler",
        ):
            raise ValueError(
                'turning_penalty must be "shaw" or "pleuler", '
                f"not {turning_penalty!r}"
            )

        self.max_time_to_intercept_seconds = max_time_to_intercept_seconds
        self.reaction_time = reaction_time
        self.max_velocity = max_velocity
        self.turning_penalty = turning_penalty.lower()

        self.player_to_starting_pos_and_vel_map = frame[
            [c + "_x" for c in game.get_column_ids()]
            + [c + "_y" for c in game.get_column_ids()]
            + [c + "_vx" for c in game.get_column_ids()]
            + [c + "_vy" for c in game.get_column_ids()]
        ]

    def tti(self, origin, destination, velocity):
        """Time, in seconds, for a player to reach ``destination``.

        Uses the estimate set by ``turning_penalty``; see the class docstring for both
        formulas and their sources.

        Args:
            origin (np.ndarray): The player's starting position (x, y) in meters.
            destination (np.ndarray): The proposed position (x, y) in meters.
            velocity (np.ndarray): The player's starting velocity (vx, vy) in meters
                per second.

        Returns:
            float: The estimated time to intercept in seconds.
        """
        if self.turning_penalty == "shaw":
            # Laurie Shaw's simple_time_to_intercept (LaurieOnTracking)
            r_reaction = origin + velocity * self.reaction_time
            d = destination - r_reaction
            return self.reaction_time + np.linalg.norm(d, axis=-1) / self.max_velocity

        # Devin Pleuler's modification, from
        # https://github.com/devinpleuler/analytics-handbook/blob/master/soccer_analytics_handbook.ipynb
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
            # clip to avoid NaN when rounding pushes the cosine outside [-1, 1]
            angle = np.arccos(np.clip(dot_product / denom, -1.0, 1.0))
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
