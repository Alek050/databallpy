import json
import math
import os
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from databallpy.events.base_event import IndividualOnBallEvent
from databallpy.features.angle import get_smallest_angle
from databallpy.models.utils import scale_and_predict_logreg
from databallpy.utils.constants import DATABALLPY_SHOT_OUTCOMES
from databallpy.utils.utils import _copy_value_, _values_are_equal_


@dataclass
class ShotEvent(IndividualOnBallEvent):
    """Class for shot events, inherits from BaseEvent. Saves all information from a
    shot from the event data, and adds information about the shot using the tracking
    data if available.

     Args:
        event_id (int): distinct id of the event
        period_id (int): id of the period
        minutes (int): minute in which the event occurs
        seconds (int): seconds within the aforementioned minute where the event occurs
        datetime (pd.Timestamp): datetime at which the event occured
        start_x (float): x coordinate of the start location of the event
        start_y (float): y coordinate of the start location of the event
        team_id (int): id of the team that performed the event
        team_side (str): side of the team that performed the event, either
            ["home", "away"]
        pitch_size (tuple): size of the pitch in meters
        player_id (int | str): id of the player that performed the event
        jersey (int): jersey number of the player that performed the event
        outcome (bool): whether the event was successful or not
        related_event_id (int | str | None): id of the event that the event is related
            to the current event.
        body_part (str): body part that the event is related to. Should be in
            databallpy.utils.constants.DATBALLPY_BODY_PARTS
        possession_type (str): type of possession that the event is related to.
            Should be in databallpy.utils.constants.DATABALLPY_POSSESSION_TYPES
        set_piece (str): type of set piece that the event is related to. Should be in
            databallpy.utils.constants.DATABALLPY_SET_PIECES
        outcome_str (str): whether the shot is a goal or not on target or not.
            Should be in databallpy.utils.constants.DATABALLPY_SHOT_OUTCOMES
        y_target (float, optional): y location of the goal. Defaults to np.nan.
        z_target (float, optional): z location of the goal. Defaults to np.nan.
        first_touch (bool, optional): whether the shot is taken with the first touch.
            Defaults to False.
        ball_goal_distance (float, optional): distance between the ball and the goal.
            Defaults to np.nan.
        shot_angle (float, optional): angle of the shot. Defaults to np.nan.


    Properties:
        xt (float): expected threat of the event. This is calculated using a model
            that is trained on the distance and angle to the goal, and the distance
            times theangle to the goal. See the notebook in the notebooks folder for
            more information on the model.
        base_df_attributes (list[str]): list of attributes that are used to create a
            DataFrame
        xg (float): expected goals of the shot. This is calculated using a model that is
            trained on the distance and angle to the goal, and the distance times the
            angle to the goal. See the notebook in the notebooks folder for more
            information on the model.

    Returns:
        ShotEvent: instance of the ShotEvent class

    Note:
        For opta event data, the related event id is not the "event_id" of the
        event data, but the "opta_id" in the event data, since opta uses different ids.
    """

    outcome_str: str
    y_target: float = np.nan
    z_target: float = np.nan
    first_touch: bool = False
    ball_goal_distance: float = np.nan
    shot_angle: float = np.nan
    xg: float = np.nan

    def __post_init__(self):
        super().__post_init__()
        self._validate_inputs_on_ball_event()
        _ = self.xt
        if pd.isnull(self.ball_goal_distance):
            self._update_ball_goal_distance()
        if pd.isnull(self.shot_angle):
            self._update_shot_angle()
        self.xg = float(self.get_xg())

    @property
    def df_attributes(self) -> list[str]:
        base_attributes = super().base_df_attributes
        return base_attributes + [
            "outcome_str",
            "y_target",
            "z_target",
            "first_touch",
            "xg",
        ]

    def _update_ball_goal_distance(self, ball_xy: np.ndarray | None = None):
        """Function to update the ball goal distance. Uses ball_xy input
        if provided, else uses the start_x and start_y of the event data.

        Args:
            ball_xy (np.ndarray | None, optional): The start location of the event.
                Defaults to None.
        """
        goal_xy = (
            [self.pitch_size[0] / 2.0, 0]
            if self.team_side == "home"
            else [-self.pitch_size[0] / 2.0, 0]
        )

        if ball_xy is None:
            # use event data
            ball_xy = np.array([self.start_x, self.start_y])
        self.ball_goal_distance = math.dist(ball_xy, goal_xy)

    def _update_shot_angle(self, ball_xy: np.ndarray | None = None):
        """Function to update the shot angle. Uses ball_xy input
        if provided, else uses the start_x and start_y of the event data.

        Args:
            ball_xy (np.ndarray | None, optional): The start location of the event.
                Defaults to None.
        """

        left_post_xy = (
            [self.pitch_size[0] / 2.0, (7.32 / 2)]
            if self.team_side == "home"
            else [-self.pitch_size[0] / 2.0, -(7.32 / 2)]
        )
        right_post_xy = (
            [self.pitch_size[0] / 2.0, -(7.32 / 2)]
            if self.team_side == "home"
            else [-self.pitch_size[0] / 2.0, (7.32 / 2)]
        )

        if ball_xy is None:
            # use event data
            ball_xy = np.array([self.start_x, self.start_y])
        # define vectors
        ball_left_post_vector = np.array(left_post_xy) - np.array(ball_xy)
        ball_right_post_vector = np.array(right_post_xy) - np.array(ball_xy)

        self.shot_angle = get_smallest_angle(
            ball_left_post_vector, ball_right_post_vector, angle_format="degree"
        )

    def get_xg(self):
        """Get expected goals of the shot event. This function calculates the xG of
        the shot. A notebook on how th xG models were created can be found in
        the documentation in features.
        """

        if self.outcome_str == "own_goal":
            return 0.0
        path = os.path.join(os.path.dirname(__file__), "..", "models")
        if pd.isnull(self.ball_goal_distance) or pd.isnull(self.shot_angle):
            return np.nan

        with open(f"{path}/xg_params.json", "r") as f:
            xg_params = json.load(f)

        if self.set_piece == "penalty":
            return 0.79

        elif self.set_piece == "free_kick":
            return scale_and_predict_logreg(
                np.array([[self.ball_goal_distance, self.shot_angle]]),
                xg_params["xg_by_free_kick"],
            )[0]

        elif "foot" not in self.body_part:
            return scale_and_predict_logreg(
                np.array([[self.ball_goal_distance, self.shot_angle]]),
                xg_params["xg_by_head"],
            )[0]

        else:  # take most general model, shot by foot
            return scale_and_predict_logreg(
                np.array([[self.ball_goal_distance, self.shot_angle]]),
                xg_params["xg_by_foot"],
            )[0]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShotEvent):
            return False
        for field in fields(self):
            if not _values_are_equal_(
                getattr(self, field.name), getattr(other, field.name)
            ):
                return False

        return True

    def copy(self):
        copied_kwargs = {
            f.name: _copy_value_(getattr(self, f.name)) for f in fields(self)
        }
        return ShotEvent(**copied_kwargs)

    def _validate_inputs_on_ball_event(self):
        if not isinstance(self.outcome_str, str):
            raise TypeError(f"shot_outcome should be str, got {type(self.outcome_str)}")
        if self.outcome_str not in DATABALLPY_SHOT_OUTCOMES:
            raise ValueError(
                f"outcome_str should be {DATABALLPY_SHOT_OUTCOMES}"
                f", got '{self.outcome_str}'"
            )
        if not isinstance(self.y_target, (float, np.floating)):
            raise TypeError(
                f"y_target should be float or int, got {type(self.y_target)}"
            )
        if not isinstance(self.z_target, (float, np.floating)):
            raise TypeError(
                f"z_target should be float or int, got {type(self.z_target)}"
            )
        if not isinstance(self.first_touch, (bool, type(None))):
            raise TypeError(f"first_touch should be bool, got {type(self.first_touch)}")

        for name, td_var in zip(
            [
                "ball_goal_distance",
                "shot_angle",
            ],
            [
                self.ball_goal_distance,
                self.shot_angle,
            ],
        ):
            if not isinstance(td_var, (float, np.floating)):
                raise TypeError(f"{name} should be float, got {type(td_var)}")
