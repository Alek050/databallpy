from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import zoom

from databallpy.features.pitch_control import get_team_influence
from databallpy.game import Game
from databallpy.optimization.optimization import ObjectiveTerm, ObjectiveType
from databallpy.schemas.tracking_data import TrackingData
from databallpy.utils.utils import sigmoid

XT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "open_play_xT.npy"
GRID_SIZE = (106, 68)


class WeightedPitchControlObjective(ObjectiveTerm):
    """Objective term that scores the net pitch control of the defending team over the
    attacking team, weighted by the expected threat (xT) of each area of the pitch.

    For every cell of a grid over the pitch, the difference between the defending and
    attacking teams' influence is squashed with a steep sigmoid (so each cell counts
    as roughly controlled by one team or the other) and multiplied by the xT value of
    that cell. The score is the sum over the grid, so higher values mean the defending
    team controls more of the high-threat areas.

    Args:
        game (Game): The game whose tracking data is being optimized.
        frame (pd.Series): The initial tracking data frame.
        xt_array (np.ndarray | None, optional): The expected threat values over the
            grid. Uses the default open-play xT model if None.
    """

    def __init__(
        self,
        game: Game,
        frame: pd.Series,
        xt_array: np.ndarray | None = None,
    ):
        super().__init__(computation_type=ObjectiveType.GRID)
        self.grid = np.meshgrid(
            np.linspace(
                -game.pitch_dimensions[0] / 2, game.pitch_dimensions[0] / 2, GRID_SIZE[0]
            ),
            np.linspace(
                -game.pitch_dimensions[1] / 2, game.pitch_dimensions[1] / 2, GRID_SIZE[1]
            ),
        )

        self.attacking_team = frame["team_possession"]
        self.defending_team = "home" if self.attacking_team == "away" else "away"
        self.defending_player_ids = game.get_column_ids(team=self.defending_team)

        self.attacking_team_influence = get_team_influence(
            frame,
            col_ids=game.get_column_ids(team=self.attacking_team),
            grid=self.grid,
            player_ball_distances=None,
        )
        if xt_array is None:
            open_play_xt = np.load(XT_MODEL_PATH)
            # we are using (y, x) orientation instead of (x, y) so that it matches
            self.xt_array = zoom(
                open_play_xt, (GRID_SIZE[0] / 264, GRID_SIZE[1] / 196), order=1
            ).T
        else:
            self.xt_array = xt_array

        if self.attacking_team == "away":
            self.xt_array = np.fliplr(self.xt_array)

    def compute(
        self,
        input_frame: pd.Series,
    ) -> float:
        team_influence_defending = get_team_influence(
            input_frame,
            col_ids=self.defending_player_ids,
            grid=self.grid,
            player_ball_distances=None,
        )
        # +ve is defending team, -ve is attacking team
        net_sigmoid_diff = sigmoid(
            team_influence_defending - self.attacking_team_influence, d=100
        )  # this makes the sigmoid steeper and more binary

        return sum(sum(self.xt_array * net_sigmoid_diff))


class PressureObjective(ObjectiveTerm):
    """Objective term that scores the mean pressure exerted on a set of players.

    The pressure on every player in ``players_to_press`` is computed as defined in ``TrackingData.get_pressure_on_player``.

    Args:
        game (Game): The game whose tracking data is being optimized.
        frame (pd.Series): The initial tracking data frame.
        players_to_press (list[str] | None, optional): Column ids of the players whose
            pressure is measured. If None, defaults to all players of the team in
            possession.
    """

    def __init__(
        self,
        game: Game,
        frame: pd.Series,
        players_to_press: list[str] | None = None,
    ):
        super().__init__(computation_type=ObjectiveType.PLAYER)

        self.game = game
        self.players_to_press = (
            players_to_press
            if players_to_press
            else game.get_column_ids(team=frame["team_possession"])
        )

    def compute(self, input_frame: pd.Series) -> float:
        # pressure method only works on tracking data object, need to temporarily reconstruct it with the new data
        pressure_score = 0
        temp_tracking_df = pd.DataFrame(input_frame).T
        temp_tracking_df = TrackingData(
            temp_tracking_df.astype(self.game.tracking_data.dtypes)
        )

        for attacking_player in self.players_to_press:
            pressure = temp_tracking_df.get_pressure_on_player(
                temp_tracking_df.index[0], attacking_player, GRID_SIZE, d_front=9
            )
            # compute the mean pressure on a player by dividing by number of players being pressed
            pressure_score += pressure / len(self.players_to_press)
        return pressure_score
