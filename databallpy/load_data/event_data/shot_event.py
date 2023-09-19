import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from databallpy.features.angle import get_smallest_angle
from databallpy.features.pressure import get_pressure_on_player
from databallpy.load_data.event_data.base_event import BaseEvent
from databallpy.utils.utils import MISSING_INT


@dataclass
class ShotEvent(BaseEvent):
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
        team_id (int): id of the team that takes the shot
        player_id (int): id of the player who takes the shot
        shot_outcome (str): whether the shot is a goal or not on target or not.
            Possible values: "goal", "own_goal", "miss_off_target", "miss_on_target",
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
        gk_angle (float, optional): angle between the shooting line and the goalkeeper
            in radian. At 0*pi radians, the gk is positined directly on the shooting
            line.
        pressure_on_ball (float, optional): pressure on the player who takes the shot.
            See Adrienko et al (2016), or the source code, for more information.
        n_obstructive_players (int, optional): number of obstructive players (both
            teammates as opponents) within the triangle from the ball and the two posts
            of the goal.


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
    gk_angle: float = np.nan
    pressure_on_ball: float = np.nan
    n_obstructive_players: int = MISSING_INT

    def __post_init__(self):
        super().__post_init__()
        self._check_datatypes()

    def add_tracking_data_features(
        self,
        tracking_data_frame: pd.Series,
        team_side: str,
        pitch_dimensions: list,
        column_id: str,
        gk_column_id: str,
    ):
        """Add tracking data features to the shot event. This function calculates the
        distance between the ball and the goal, the distance between the ball and the
        goalkeeper, the angle between the ball and the goal, the angle between the
        ball and the goalkeeper, the pressure on the ball, and the number of
        obstructive defenders.


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
            [pitch_dimensions[0] / 2.0, 0]
            if team_side == "home"
            else [-pitch_dimensions[0] / 2.0, 0]
        )
        left_post_xy = (
            [pitch_dimensions[0] / 2.0, (7.32 / 2)]
            if team_side == "home"
            else [-pitch_dimensions[0] / 2.0, -(7.32 / 2)]
        )
        right_post_xy = (
            [pitch_dimensions[0] / 2.0, -(7.32 / 2)]
            if team_side == "home"
            else [-pitch_dimensions[0] / 2.0, (7.32 / 2)]
        )
        ball_xy = tracking_data_frame[["ball_x", "ball_y"]].values
        gk_xy = tracking_data_frame[[f"{gk_column_id}_x", f"{gk_column_id}_y"]].values
        middle_xy = [0, 0]

        # define vectors
        ball_goal_vector = np.array(goal_xy) - np.array(ball_xy)
        goal_gk_vector = np.array(goal_xy) - np.array(gk_xy)
        goal_middle_vector = np.array(goal_xy) - np.array(middle_xy)

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
        n_obstructive_players = 0
        for player_xy in players_xy:
            if triangle.find_simplex(player_xy) >= 0:
                n_obstructive_players += 1

        # add variables
        self.ball_goal_distance = math.dist(ball_xy, goal_xy)
        self.ball_gk_distance = math.dist(ball_xy, gk_xy)
        self.shot_angle = get_smallest_angle(goal_middle_vector, ball_goal_vector)
        self.gk_angle = get_smallest_angle(ball_goal_vector, goal_gk_vector)
        self.pressure_on_ball = get_pressure_on_player(
            tracking_data_frame,
            column_id,
            pitch_size=pitch_dimensions,
            d_front="variable",
            d_back=3.0,
            q=1.75,
        )
        self.n_obstructive_players = n_obstructive_players

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
            self.ball_goal_distance == other.ball_goal_distance
            if not pd.isnull(self.ball_goal_distance)
            else pd.isnull(other.ball_goal_distance),
            self.shot_angle == other.shot_angle
            if not pd.isnull(self.shot_angle)
            else pd.isnull(other.shot_angle),
            self.gk_angle == other.gk_angle
            if not pd.isnull(self.gk_angle)
            else pd.isnull(other.gk_angle),
            self.pressure_on_ball == other.pressure_on_ball
            if not pd.isnull(self.pressure_on_ball)
            else pd.isnull(other.pressure_on_ball),
            self.n_obstructive_players == other.n_obstructive_players,
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
            gk_angle=self.gk_angle,
            pressure_on_ball=self.pressure_on_ball,
            n_obstructive_players=self.n_obstructive_players,
        )

    def _check_datatypes(self):
        if not isinstance(self.player_id, int):
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
        if not isinstance(self.y_target, float):
            raise TypeError(
                f"y_target should be float or int, got {type(self.y_target)}"
            )
        if not isinstance(self.z_target, float):
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
        if not isinstance(self.related_event_id, int):
            raise TypeError(
                f"related_event_id should be int, got {type(self.related_event_id)}"
            )
