from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from databallpy.events.base_event import IndividualOnBallEvent
from databallpy.features.angle import get_smallest_angle
from databallpy.features.pressure import get_pressure_on_player
from databallpy.utils.constants import (
    DATABALLPY_PASS_OUTCOMES,
    DATABALLPY_PASS_TYPES,
    MISSING_INT,
)
from databallpy.utils.utils import _copy_value_, _values_are_equal_


@dataclass
class PassEvent(IndividualOnBallEvent):
    """This is the pass event class. It contains all the information that is available
    for a pass event.

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
        outcome_str (str): outcome of the pass, should be in
            databallpy.utils.constants.DATABALLPY_PASS_OUTCOMES
        end_x (float): x coordinate of the end location of the pass
        end_y (float): y coordinate of the end location of the pass
        pass_type (str): type of the pass, should be in
            databallpy.utils.constants.DATABALLPY_PASS_TYPES
        receiver_id (int): id of the player that receives the pass. Default is
            databallpy.utils.constants.MISSING_INT
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

    outcome_str: str
    end_x: float
    end_y: float
    pass_type: str
    receiver_player_id: int = MISSING_INT
    pass_length: float = np.nan
    forward_distance: float = np.nan
    passer_goal_distance: float = np.nan
    pass_end_loc_goal_distance: float = np.nan
    opponents_in_passing_lane: int = MISSING_INT
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
        copied_kwargs = {
            f.name: _copy_value_(getattr(self, f.name)) for f in fields(self)
        }
        return PassEvent(**copied_kwargs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PassEvent):
            return False
        for field in fields(self):
            if not _values_are_equal_(
                getattr(self, field.name), getattr(other, field.name)
            ):
                return False

        return True

    def __post_init__(self):
        super().__post_init__()
        self._validate_inputs_on_ball_event()

    @property
    def df_attributes(self) -> list[str]:
        base_attributes = super().base_df_attributes
        return base_attributes + [
            "outcome_str",
            "end_x",
            "end_y",
            "pass_type",
            "receiver_player_id",
            "pass_length",
            "forward_distance",
            "passer_goal_distance",
            "pass_end_loc_goal_distance",
            "opponents_in_passing_lane",
            "pressure_on_passer",
            "pressure_on_receiver",
            "pass_goal_angle",
        ]

    def _validate_inputs_on_ball_event(self):
        if not isinstance(self.outcome_str, (str, type(None))):
            raise TypeError(f"outcome should be str, not {type(self.outcome_str)}")

        if self.outcome_str not in DATABALLPY_PASS_OUTCOMES:
            raise ValueError(
                f"outcome_str should be one of {DATABALLPY_PASS_OUTCOMES},"
                f" not {self.outcome_str}"
            )

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
        for name in names:
            value = getattr(self, name)
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

        if self.pass_type not in DATABALLPY_PASS_TYPES:
            raise ValueError(
                f"pass_type should be one of {DATABALLPY_PASS_TYPES}, "
                f"not {self.pass_type}"
            )


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
