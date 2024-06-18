import json
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from databallpy.events.base_event import BaseOnBallEvent
from databallpy.features.angle import get_smallest_angle
from databallpy.features.pressure import get_pressure_on_player
from databallpy.models.utils import scale_and_predict_logreg
from databallpy.utils.constants import MISSING_INT


@dataclass
class ShotEvent(BaseOnBallEvent):
    """Class for shot events, inherits from BaseEvent. Saves all information from a
    shot from the event data, and adds information about the shot using the tracking
    data if available.

    Args:
        event_id (int): distinct id of the event
        period_id (int): id of the period
        minutes (int): minute in which the event occurs
        seconds (int): seconds within the aforementioned minute where the event occurs
        datetime (pd.Timestamp): datetime at which the event occured
        start_x (float): x location of the event
        start_y (float): y location of the event
        pitch_size (list): dimensions of the pitch
        team_side (str): side of the team that takes the shot, either "home" or "away"
        team_id (int): id of the team that takes the shot
        player_id (int): id of the player who takes the shot
        shot_outcome (str): whether the shot is a goal or not on target or not.
            Possible values: "goal", "miss_off_target", "miss_on_target",
            "blocked", "miss_hit_post" "miss"
        y_target (float, optional): y location of the goal. Defaults to np.nan.
        z_target (float, optional): z location of the goal. Defaults to np.nan.
        body_part (str, optional): body part with which the shot is taken. Defaults to
            None.
        type_of_play (str, optional): open play or type of set piece. Defaults to
            "regular_play".
        first_touch (bool, optional): whether the shot is taken with the first touch.
            Defaults to False.
        created_oppertunity (str, optional): how the chance was created, assisted or
            individual play. Defaults to None.
        related_event_id (int, optional): id of the event that led to the shot. Defaults
            to MISSING_INT.
        ball_goal_distance (float, optional): distance between the ball and the goal in
            meters.
        ball_gk_distance (float, optional): distance between the ball and the goalkeeper
            in meters.
        shot_angle (float, optional): angle between the shooting line and the goal in
            radian. At 0*pi radians, the ball is positined directly in front of the
            goal.
        gk_optimal_loc_distance (float, optional): Shortest distance between the gk and
            the line between the ball and the goal. The optimal location of the gk is
            when the gk is directly on the line between the ball and middle of the goal.
        pressure_on_ball (float, optional): pressure on the player who takes the shot.
            See Adrienko et al (2016), or the source code, for more information.
        n_obstructive_players (int, optional): number of obstructive players (both
            teammates as opponents) within the triangle from the ball and the two posts
            of the goal.
        n_obstructive_defenders (int, optional): number of obstructive opponents within
            the triangle from the ball and the two posts of the goal.
        goal_gk_distance (float, optional): distance between the goal and the
            goalkeeper in meters.
        set_piece (str, optional): type of set piece. Defaults to "no_set_piece".
            Choices: "no_set_piece", "free_kick", "penalty"

    Attributes:
        xG (float): expected goals of the shot. This is calculated using a model that is
            trained on the distance and angle to the goal, and the distance times the
            angle to the goal. See the notebook in the notebooks folder for more
            information on the model.
        xT (float): expected threat of the event. This is calculated using a model that
            is trained on the distance and angle to the goal, and the distance times
            the angle to the goal. See the notebook in the notebooks folder for more
            information on the model.
        df_attributes (list[str]): list of attributes that are used to create a
            DataFrame.

    Returns:
        ShotEvent: instance of the ShotEvent class

    Note:
        For opta event data, the related event id is not the "event_id" of the
        event data, but the "opta_id" in the event data, since opta uses different ids.
    """

    player_id: int
    shot_outcome: str
    y_target: float = np.nan
    z_target: float = np.nan
    body_part: str = None
    type_of_play: str = "regular_play"
    first_touch: bool = False
    created_oppertunity: str = None

    # id of the event that led to the shot. Note: opta id for opta event data
    related_event_id: int = MISSING_INT

    # tracking data variables
    ball_goal_distance: float = np.nan
    ball_gk_distance: float = np.nan
    shot_angle: float = np.nan
    gk_optimal_loc_distance: float = np.nan
    pressure_on_ball: float = np.nan
    n_obstructive_players: int = MISSING_INT
    n_obstructive_defenders: int = MISSING_INT
    goal_gk_distance: float = np.nan
    xG: float = np.nan
    set_piece: str = "no_set_piece"

    def __post_init__(self):
        super().__post_init__()
        self._check_datatypes()
        if self.type_of_play in ["penalty", "free_kick"]:
            self.set_piece = self.type_of_play
        _ = self._xt
        if pd.isnull(self.ball_goal_distance):
            self._update_ball_goal_distance()
        if pd.isnull(self.shot_angle):
            self._update_shot_angle()
        self.xG = float(self.get_xG())

    @property
    def df_attributes(self) -> list[str]:
        base_attributes = super().base_df_attributes
        return base_attributes + [
            "player_id",
            "shot_outcome",
            "y_target",
            "z_target",
            "body_part",
            "type_of_play",
            "first_touch",
            "created_oppertunity",
            "related_event_id",
            "ball_goal_distance",
            "ball_gk_distance",
            "shot_angle",
            "gk_optimal_loc_distance",
            "pressure_on_ball",
            "n_obstructive_players",
            "n_obstructive_defenders",
            "goal_gk_distance",
            "xG",
            "set_piece",
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

    def add_tracking_data_features(
        self,
        tracking_data_frame: pd.Series,
        column_id: str,
        gk_column_id: str,
    ):
        """Add tracking data features to the shot event. This function calculates the
        distance between the ball and the goal, the distance between the ball and the
        goalkeeper, the angle between the ball and the goal, the distance between the
        gk and the optimal position of the gk, the pressure on the ball, the
        number of obstructive players, and the number of obstructive defenders.


        Args:
            tracking_data_frame (pd.Series): tracking data frame of the event
            team_side (str): side of the team that takes the shot, either
                "home" or "away"
            pitch_dimensions (list): dimensions of the pitch
            column_id (str): column id of the player who takes the shot
            gk_column_id (str): column id of the goalkeeper
        """
        # define positions
        goal_xy = (
            [self.pitch_size[0] / 2.0, 0]
            if self.team_side == "home"
            else [-self.pitch_size[0] / 2.0, 0]
        )
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
        ball_xy = tracking_data_frame[["ball_x", "ball_y"]].values
        gk_xy = tracking_data_frame[[f"{gk_column_id}_x", f"{gk_column_id}_y"]].values

        # calculate obstructive players
        triangle = Delaunay([right_post_xy, left_post_xy, ball_xy])
        players_column_ids = [
            x[:-2]
            for x in tracking_data_frame.index
            if ("_x" in x and "ball" not in x and x != f"{column_id}_x")
        ]
        x_vals = tracking_data_frame[[f"{x}_x" for x in players_column_ids]].values
        y_vals = tracking_data_frame[[f"{x}_y" for x in players_column_ids]].values
        players_xy = np.array([x_vals, y_vals]).T
        n_obstructive_players = (triangle.find_simplex(players_xy) >= 0).sum()

        opponent_column_ids = [x for x in players_column_ids if self.team_side not in x]
        x_vals = tracking_data_frame[[f"{x}_x" for x in opponent_column_ids]].values
        y_vals = tracking_data_frame[[f"{x}_y" for x in opponent_column_ids]].values
        opponent_xy = np.array([x_vals, y_vals]).T
        n_obstructive_defenders = (triangle.find_simplex(opponent_xy) >= 0).sum()

        # add variables
        self._update_ball_goal_distance(ball_xy)
        self.ball_gk_distance = math.dist(ball_xy, gk_xy)
        self._update_shot_angle(ball_xy)
        self.gk_optimal_loc_distance = float(
            np.linalg.norm(np.cross(goal_xy - ball_xy, ball_xy - gk_xy))
            / np.linalg.norm(goal_xy - ball_xy)
        )
        self.pressure_on_ball = float(
            get_pressure_on_player(
                tracking_data_frame,
                column_id,
                pitch_size=self.pitch_size,
                d_front="variable",
                d_back=3.0,
                q=1.75,
            )
        )
        self.n_obstructive_players = int(n_obstructive_players)
        self.n_obstructive_defenders = int(n_obstructive_defenders)
        self.goal_gk_distance = float(np.linalg.norm(goal_xy - gk_xy))

        self.xG = float(self.get_xG())

    def get_xG(self):
        """Get xG of the shot event. This function calculates the xG of the shot.
        A notebook on how th xG models were created can be found in the notebooks
        folder.
        """

        path = os.path.join(os.path.dirname(__file__), "..", "models")
        if pd.isnull(self.ball_goal_distance) or pd.isnull(self.shot_angle):
            return np.nan

        with open(f"{path}/xg_params.json", "r") as f:
            xg_params = json.load(f)

        if self.type_of_play == "penalty":
            return 0.79

        elif self.type_of_play == "free_kick":
            return scale_and_predict_logreg(
                np.array([[self.ball_goal_distance, self.shot_angle]]),
                xg_params["xG_by_free_kick"],
            )[0]

        elif (
            self.type_of_play
            in [
                "regular_play",
                "corner_kick",
                "crossed_free_kick",
                "counter_attack",
            ]
            and "foot" not in self.body_part
        ):
            return scale_and_predict_logreg(
                np.array([[self.ball_goal_distance, self.shot_angle]]),
                xg_params["xG_by_head"],
            )[0]

        else:  # take most general model, shot by foot
            return scale_and_predict_logreg(
                np.array([[self.ball_goal_distance, self.shot_angle]]),
                xg_params["xG_by_foot"],
            )[0]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShotEvent):
            return False
        result = [
            super().__eq__(other),
            self.player_id == other.player_id,
            self.shot_outcome == other.shot_outcome,
            round(self.y_target, 4) == round(other.y_target, 4)
            if not pd.isnull(self.y_target)
            else pd.isnull(other.y_target),
            round(self.z_target, 4) == round(other.z_target, 4)
            if not pd.isnull(self.z_target)
            else pd.isnull(other.z_target),
            self.body_part == other.body_part
            if not pd.isnull(self.body_part)
            else pd.isnull(other.body_part),
            self.type_of_play == other.type_of_play
            if not pd.isnull(self.type_of_play)
            else pd.isnull(other.type_of_play),
            self.first_touch == other.first_touch
            if not pd.isnull(self.first_touch)
            else pd.isnull(other.first_touch),
            self.created_oppertunity == other.created_oppertunity
            if not pd.isnull(self.created_oppertunity)
            else pd.isnull(other.created_oppertunity),
            self.related_event_id == other.related_event_id,
            self.ball_gk_distance == other.ball_gk_distance
            if not pd.isnull(self.ball_gk_distance)
            else pd.isnull(other.ball_gk_distance),
            math.isclose(
                self.ball_goal_distance, other.ball_goal_distance, abs_tol=1e-5
            )
            if not pd.isnull(self.ball_goal_distance)
            else pd.isnull(other.ball_goal_distance),
            math.isclose(self.shot_angle, other.shot_angle, abs_tol=1e-5)
            if not pd.isnull(self.shot_angle)
            else pd.isnull(other.shot_angle),
            self.gk_optimal_loc_distance == other.gk_optimal_loc_distance
            if not pd.isnull(self.gk_optimal_loc_distance)
            else pd.isnull(other.gk_optimal_loc_distance),
            self.pressure_on_ball == other.pressure_on_ball
            if not pd.isnull(self.pressure_on_ball)
            else pd.isnull(other.pressure_on_ball),
            self.n_obstructive_players == other.n_obstructive_players,
            self.n_obstructive_defenders == other.n_obstructive_defenders,
            self.goal_gk_distance == other.goal_gk_distance
            if not pd.isnull(self.goal_gk_distance)
            else pd.isnull(other.goal_gk_distance),
            math.isclose(self.xG, other.xG, abs_tol=1e-5)
            if not pd.isnull(self.xG)
            else pd.isnull(other.xG),
        ]
        return all(result)

    def copy(self):
        return ShotEvent(
            event_id=self.event_id,
            period_id=self.period_id,
            minutes=self.minutes,
            seconds=self.seconds,
            datetime=self.datetime,
            start_x=self.start_x,
            start_y=self.start_y,
            pitch_size=self.pitch_size,
            team_side=self.team_side,
            _xt=self._xt,
            team_id=self.team_id,
            player_id=self.player_id,
            shot_outcome=self.shot_outcome,
            y_target=self.y_target,
            z_target=self.z_target,
            body_part=self.body_part,
            type_of_play=self.type_of_play,
            first_touch=self.first_touch,
            created_oppertunity=self.created_oppertunity,
            related_event_id=self.related_event_id,
            ball_gk_distance=self.ball_gk_distance,
            ball_goal_distance=self.ball_goal_distance,
            shot_angle=self.shot_angle,
            gk_optimal_loc_distance=self.gk_optimal_loc_distance,
            pressure_on_ball=self.pressure_on_ball,
            n_obstructive_players=self.n_obstructive_players,
            n_obstructive_defenders=self.n_obstructive_defenders,
            goal_gk_distance=self.goal_gk_distance,
            xG=self.xG,
        )

    def _check_datatypes(self):
        if not isinstance(self.player_id, (int, np.integer, str)):
            raise TypeError(f"player_id should be int, got {type(self.player_id)}")
        if not isinstance(self.shot_outcome, str):
            raise TypeError(
                f"shot_outcome should be str, got {type(self.shot_outcome)}"
            )
        if self.shot_outcome not in [
            "goal",
            "miss_off_target",
            "miss_hit_post",
            "miss_on_target",
            "blocked",
            "own_goal",
            "miss",
            "not_specified",
        ]:
            raise ValueError(
                "shot_outcome should be goal, miss_off_target, miss_hit_post, "
                f"miss_on_target, blocked or own_goal, got '{self.shot_outcome}'"
            )
        if not isinstance(self.y_target, (float, np.floating)):
            raise TypeError(
                f"y_target should be float or int, got {type(self.y_target)}"
            )
        if not isinstance(self.z_target, (float, np.floating)):
            raise TypeError(
                f"z_target should be float or int, got {type(self.z_target)}"
            )
        if not isinstance(self.body_part, (str, type(None))):
            raise TypeError(
                f"body_part should be str or None, got {type(self.body_part)}"
            )
        if self.body_part not in ["left_foot", "right_foot", "head", "other", None]:
            raise ValueError(
                "body_part should be left_foot, right_foot, head or other, "
                f"got {self.body_part}"
            )
        if not isinstance(self.type_of_play, (str, type(None))):
            raise TypeError(
                f"type_of_play should be str, got {type(self.type_of_play)}"
            )
        if self.type_of_play not in [
            "penalty",
            "regular_play",
            "counter_attack",
            "crossed_free_kick",
            "corner_kick",
            "free_kick",
            None,
        ]:
            raise ValueError(
                "type_of_play should be penalty, regular_play, counter_attack, "
                f"crossed_free_kick, corner_kick or free_kick, got {self.type_of_play}"
            )
        if not isinstance(self.first_touch, (bool, type(None))):
            raise TypeError(f"first_touch should be bool, got {type(self.first_touch)}")
        if not isinstance(self.created_oppertunity, (str, type(None))):
            raise TypeError(
                "created_oppertunity should be str or None, got "
                f"{type(self.created_oppertunity)}"
            )
        if self.created_oppertunity not in [
            "assisted",
            "individual_play",
            "regular_play",
            None,
        ]:
            raise ValueError(
                "created_oppertunity should be assisted, regular_play, or "
                f"individual_play, got {self.created_oppertunity}"
            )
        if not isinstance(self.related_event_id, (int, np.integer)):
            raise TypeError(
                f"related_event_id should be int, got {type(self.related_event_id)}"
            )

        for name, td_var in zip(
            [
                "ball_goal_distance",
                "ball_gk_distance",
                "shot_angle",
                "gk_optimal_loc_distance",
                "pressure_on_ball",
                "goal_gk_distance",
            ],
            [
                self.ball_goal_distance,
                self.ball_gk_distance,
                self.shot_angle,
                self.gk_optimal_loc_distance,
                self.pressure_on_ball,
                self.goal_gk_distance,
            ],
        ):
            if not isinstance(td_var, (float, np.floating)):
                raise TypeError(f"{name} should be float, got {type(td_var)}")

        for name, td_var in zip(
            ["n_obstructive_players", "n_obstructive_defenders"],
            [self.n_obstructive_players, self.n_obstructive_defenders],
        ):
            if not isinstance(td_var, (int, np.integer)):
                raise TypeError(f"{name} should be int, got {type(td_var)}")
