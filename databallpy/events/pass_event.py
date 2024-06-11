from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from databallpy.events.base_event import BaseOnBallEvent
from databallpy.features.angle import get_smallest_angle
from databallpy.features.pressure import get_pressure_on_player
from databallpy.utils.constants import MISSING_INT


@dataclass
class PassEvent(BaseOnBallEvent):
    """This is the pass event class. It contains all the information that is available
    for a pass event.

    Args:
        event_id (int): distinct id of the pass event
        period_id (int): id of the period
        minutes (int): minute in which the pass occurs
        seconds (int): seconds within the aforementioned minute where the pass occurs
        datetime (pd.Timestamp): datetime at which the pass occured
        start_x (float): x coordinate of the start location of the pass
        start_y (float): y coordinate of the start location of the pass
        pitch_size (tuple): size of the pitch in meters.
        team_side (str): side of the team that performed the pass, either
            ["home", "away"]
        team_id (int): id of the team that performed the pass
        outcome (str): outcome of the pass, options are:
            ['successful', 'unsuccessful', 'offside', 'results_in_shot',
            'assist', 'fair_play', 'not_specified']
        player_id (int): id of the player that performed the pass
        end_x (float): x coordinate of the end location of the pass
        end_y (float): y coordinate of the end location of the pass
        pass_type (str): type of the pass, options are:
            ['long_ball', 'cross', 'through_ball', 'chipped', 'lay-off', 'lounge',
            'flick_on', 'pull_back', 'switch_off_play', 'not_specified']
        set_piece (str): type of set piece, options are:
            ['goal_kick', 'free_kick', 'throw_in', 'corner_kick', 'kick_off',
            'penalty', 'no_set_piece', unspecified_set_piece]
        pass_length (float): length of the pass
        forward_distance (float): distance the pass is made forward.
            Default is np.nan.
        passer_goal_distance (float): distance of the passer to the goal.
            Default is np.nan.
        pass_end_loc_goal_distance (float): distance of the end location of the pass
            to the goal. Default is np.nan.
        opponents_in_passing_lane (int): number of opponents in the passing lane.
            Default is np.nan.
        pressure_on_passer (float): pressure on the passer. Default is np.nan.
        pressure_on_receiver (float): pressure on the receiver. Default is np.nan.
        pass_goal_angle (float): angle between the passer, the goal and the end
            location of the pass. Default is np.nan.

    Attributes:
        xT (float): xT (float): expected threat of the event. This is calculated using a
            model that is trained on the distance and angle to the goal, and the
            distance times the angle to the goal. See the notebook in the notebooks
            folder for more information on the model.
        df_attributes (list[str]): list of attributes that are used to create a
            DataFrame.

    Raises:
        TypeError: If any of the inputtypes is not correct
    """

    outcome: str
    player_id: int
    end_x: float
    end_y: float
    pass_type: str
    set_piece: str
    receiver_id: int = MISSING_INT
    pass_length: float = np.nan
    forward_distance: float = np.nan
    passer_goal_distance: float = np.nan
    pass_end_loc_goal_distance: float = np.nan
    opponents_in_passing_lane: int = np.nan
    pressure_on_passer: float = np.nan
    pressure_on_receiver: float = np.nan
    pass_goal_angle: float = np.nan

    def add_tracking_data_features(
        self,
        tracking_data_frame: pd.Series,
        passer_column_id: str,
        receiver_column_id: str,
        pass_end_location: np.ndarray,
        pitch_dimensions: list,
        opponent_column_ids: list,
    ):
        frame = tracking_data_frame.copy()
        side = passer_column_id[:4]

        self.pressure_on_passer = get_pressure_on_player(
            frame, passer_column_id, pitch_size=pitch_dimensions
        )
        self.pressure_on_receiver = get_pressure_on_player(
            frame, receiver_column_id, pitch_size=pitch_dimensions
        )

        # make sure the team that is passing is always represented as
        # playing from left to right
        if side == "away":
            to_flip_cols = [x for x in frame.index if x[-1] == "x" or x[-1] == "y"]
            frame[to_flip_cols] *= -1
            pass_end_location *= -1

        goal_loc = [pitch_dimensions[0] / 2.0, 0]
        passer_loc = frame[[f"{passer_column_id}_x", f"{passer_column_id}_y"]].values
        self.pass_length = float(np.linalg.norm(passer_loc - pass_end_location))
        self.forward_distance = float(pass_end_location[0] - passer_loc[0])
        self.passer_goal_distance = float(np.linalg.norm(passer_loc - goal_loc))
        self.pass_end_loc_goal_distance = float(
            np.linalg.norm(pass_end_location - goal_loc)
        )
        self.opponents_in_passing_lane = int(
            get_opponents_in_passing_lane(
                frame, passer_loc, pass_end_location, opponent_column_ids
            )
        )

        passer_goal_vec = goal_loc - passer_loc
        passer_receiver_vec = pass_end_location - passer_loc
        self.pass_goal_angle = float(
            get_smallest_angle(
                passer_goal_vec, passer_receiver_vec, angle_format="degree"
            )
        )

    def copy(self):
        return PassEvent(
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
            outcome=self.outcome,
            player_id=self.player_id,
            end_x=self.end_x,
            end_y=self.end_y,
            pass_length=self.pass_length,
            pass_type=self.pass_type,
            set_piece=self.set_piece,
            forward_distance=self.forward_distance,
            passer_goal_distance=self.passer_goal_distance,
            pass_end_loc_goal_distance=self.pass_end_loc_goal_distance,
            opponents_in_passing_lane=self.opponents_in_passing_lane,
            pressure_on_passer=self.pressure_on_passer,
            pressure_on_receiver=self.pressure_on_receiver,
            pass_goal_angle=self.pass_goal_angle,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PassEvent):
            return False
        result = [
            super().__eq__(other),
            self.team_id == other.team_id,
            self.outcome == other.outcome,
            self.player_id == other.player_id,
            round(self.end_x, 4) == round(other.end_x, 4)
            if not pd.isnull(self.end_x)
            else pd.isnull(other.end_x),
            round(self.end_y, 4) == round(other.end_y, 4)
            if not pd.isnull(self.end_y)
            else pd.isnull(other.end_y),
            self.pass_length == other.pass_length
            if not pd.isnull(self.pass_length)
            else pd.isnull(other.pass_length),
            self.pass_type == other.pass_type,
            self.set_piece == other.set_piece,
            round(self.forward_distance, 4) == round(other.forward_distance, 4)
            if not pd.isnull(self.forward_distance)
            else pd.isnull(other.forward_distance),
            round(self.passer_goal_distance, 4) == round(other.passer_goal_distance, 4)
            if not pd.isnull(self.passer_goal_distance)
            else pd.isnull(other.passer_goal_distance),
            round(self.pass_end_loc_goal_distance, 4)
            == round(other.pass_end_loc_goal_distance, 4)
            if not pd.isnull(self.pass_end_loc_goal_distance)
            else pd.isnull(other.pass_end_loc_goal_distance),
            self.opponents_in_passing_lane == other.opponents_in_passing_lane
            if not pd.isnull(self.opponents_in_passing_lane)
            else pd.isnull(other.opponents_in_passing_lane),
            round(self.pressure_on_passer, 4) == round(other.pressure_on_passer, 4)
            if not pd.isnull(self.pressure_on_passer)
            else pd.isnull(other.pressure_on_passer),
            round(self.pressure_on_receiver, 4) == round(other.pressure_on_receiver, 4)
            if not pd.isnull(self.pressure_on_receiver)
            else pd.isnull(other.pressure_on_receiver),
            round(self.pass_goal_angle, 4) == round(other.pass_goal_angle, 4)
            if not pd.isnull(self.pass_goal_angle)
            else pd.isnull(other.pass_goal_angle),
        ]
        return all(result)

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.outcome, (str, type(None))):
            raise TypeError(f"outcome should be str, not {type(self.outcome)}")

        valid_outcomes = [
            "successful",
            "unsuccessful",
            "offside",
            "results_in_shot",
            "assist",
            "fair_play",
            "not_specified",
            None,
        ]
        if self.outcome not in valid_outcomes:
            raise ValueError(
                f"outcome should be one of {valid_outcomes}, not {self.outcome}"
            )

        if not isinstance(self.player_id, (int, np.integer, str)):
            raise TypeError(f"player_id should be int, not {type(self.player_id)}")

        values = [
            self.end_x,
            self.end_y,
            self.pass_length,
            self.forward_distance,
            self.passer_goal_distance,
            self.pass_end_loc_goal_distance,
            self.pressure_on_passer,
            self.pressure_on_receiver,
            self.pass_goal_angle,
        ]
        names = [
            "end_x",
            "end_y",
            "pass_length",
            "forward_distance",
            "passer_goal_distance",
            "pass_end_loc_goal_distance",
            "pressure_on_passer",
            "pressure_on_receiver",
            "pass_goal_angle",
        ]
        for value, name in zip(values, names):
            if not isinstance(value, (float, np.floating)):
                raise TypeError(f"{name} should be float, not {type(value)}")

        if not isinstance(
            self.opponents_in_passing_lane, (np.integer, int, float, np.floating)
        ):
            raise TypeError(
                "opponents_in_passing_lane should be int, not "
                f"{type(self.opponents_in_passing_lane)}"
            )

        if not isinstance(self.pass_type, str):
            raise TypeError(f"pass_type should be str, not {type(self.pass_type)}")
        valid_pass_types = [
            "long_ball",
            "cross",
            "through_ball",
            "chipped",
            "lay-off",
            "lounge",
            "flick_on",
            "pull_back",
            "switch_off_play",
            "not_specified",
            "assist",
        ]
        if self.pass_type not in valid_pass_types:
            raise ValueError(
                f"pass_type should be one of {valid_pass_types}, not {self.pass_type}"
            )

        if not isinstance(self.set_piece, str):
            raise TypeError(f"set_piece should be str, not {type(self.set_piece)}")

        valid_set_pieces = [
            "goal_kick",
            "free_kick",
            "throw_in",
            "corner_kick",
            "kick_off",
            "penalty",
            "no_set_piece",
            "unspecified_set_piece",
        ]
        if self.set_piece not in valid_set_pieces:
            raise ValueError(
                f"set_piece should be one of {valid_set_pieces}, not {self.set_piece}"
            )
        _ = self._xt

    @property
    def df_attributes(self) -> list[str]:
        base_attributes = super().base_df_attributes
        return base_attributes + [
            "outcome",
            "player_id",
            "end_x",
            "end_y",
            "pass_type",
            "set_piece",
            "receiver_id",
            "pass_length",
            "forward_distance",
            "passer_goal_distance",
            "pass_end_loc_goal_distance",
            "opponents_in_passing_lane",
            "pressure_on_passer",
            "pressure_on_receiver",
            "pass_goal_angle",
        ]


def get_opponents_in_passing_lane(
    frame: pd.Series,
    start_loc: list,
    end_loc: list,
    opponent_column_ids: str,
    lane_size: float = 0.8,
    angle: float = 10.0,
) -> int:
    """Function to calculate the number of opponents in the passing lane. The
    passing lane is defined as the area between the two lines that are parallel
    to the passing direction and are at a distance of 0.8 (lane_size) meters from
    the passing direction. However, when the angle parameter is set, the passing
    lane is between the lines that start the lane_size distance from the passing
    direction and end with an angle of the angle parameter with the passing direction.

    Args:
        frame (pd.Series): Frame of tracking data when the pass is performed
        start_loc (list): Start location of pass
        end_loc (list): End location of pass
        opponent_column (list): All column ids of the opponents.
        lane_size (float, optional): The distance on either side of the passing
            direction that makes the area of the passing lane. Defaults to 0.8 meters.
        angle (float, optional): The angle in degrees that the passing lane makes with
            the passing direction. Defaults to 10 degrees.

    Returns:
        int: Number of opponents in the passing lane
    """
    start_end_vec = np.array(end_loc) - np.array(start_loc)

    magnitude = np.linalg.norm(start_end_vec)
    unit_vec = start_end_vec / magnitude
    perpendicular_vec = np.array([-unit_vec[1], unit_vec[0]])

    angle_rad = np.deg2rad(angle)
    to_add = np.tan(angle_rad) * magnitude

    # create a vector parallel to the passing lane
    plus_point1 = (start_loc + perpendicular_vec * lane_size).tolist()
    plus_point2 = (
        (end_loc + perpendicular_vec * lane_size) + (to_add * perpendicular_vec)
    ).tolist()
    minus_point1 = (start_loc - perpendicular_vec * lane_size).tolist()
    minus_point2 = (
        (end_loc - perpendicular_vec * lane_size) - (to_add * perpendicular_vec)
    ).tolist()

    # create an area and count the number of opponents in that area
    if pd.isnull([plus_point1, plus_point2, minus_point1, minus_point2]).any():
        return np.nan
    area = Delaunay(np.array([plus_point1, plus_point2, minus_point1, minus_point2]))
    selected_columns = [f"{x}_x" for x in opponent_column_ids] + [
        f"{x}_y" for x in opponent_column_ids
    ]
    opponent_locs = frame[selected_columns].values
    opponent_locs = opponent_locs.reshape(2, -1).T

    tot_defs = (area.find_simplex(opponent_locs) >= 0).sum()

    return tot_defs
