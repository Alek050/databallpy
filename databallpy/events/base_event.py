import math
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from databallpy.features.angle import get_smallest_angle


@dataclass
class BaseOnBallEvent:
    """This is the base on ball event class from which the specific event classes are
    inherited. It containts all the basic information that is available for every event.

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
        pitch_size (tuple): size of the pitch in meters, default is (106, 68)

    Attributes:
        xT (float): expected threat of the event. This is calculated using a model
            that is trained on the distance and angle to the goal, and the distance
            times theangle to the goal. See the notebook in the notebooks folder for
            more information on the model.
    """

    event_id: int
    period_id: int
    minutes: int
    seconds: int
    datetime: pd.Timestamp
    start_x: float
    start_y: float
    team_id: int
    team_side: str
    pitch_size: tuple[float, float]
    _xt: float

    @property
    def xT(self) -> float:
        """This property returns the expected threat of the event. It is calculated
        using a model that is trained on the distance and angle to the goal, and the
        distance times the angle to the goal. See the notebook in the notebooks folder
        for more information on the model.

        Raises:
            ValueError: when the set_piece is not one of ['penalty', 'corner_kick',
                'goal_kick', 'kick_off', 'throw_in', 'free_kick', 'no_set_piece',
                'unspecified_set_piece']

        Returns:
            float: expected threat of the event
        """
        if self._xt < 0.0:
            if hasattr(self, "set_piece"):
                set_piece = self.set_piece
            else:
                set_piece = "no_set_piece"

            path = os.path.join(os.path.dirname(__file__), "..", "models")

            if set_piece in [
                "no_set_piece",
                "free_kick",
                "throw_in",
                "unspecified_set_piece",
            ]:
                goal_loc = (
                    np.array([self.pitch_size[0] / 2, 0.0])
                    if self.team_side == "home"
                    else np.array([-self.pitch_size[0] / 2, 0.0])
                )
                left_post = (
                    np.array([self.pitch_size[0] / 2, 3.66])
                    if self.team_side == "home"
                    else np.array([-self.pitch_size[0] / 2, -3.66])
                )
                right_post = (
                    np.array([self.pitch_size[0] / 2, -3.66])
                    if self.team_side == "home"
                    else np.array([-self.pitch_size[0] / 2, 3.66])
                )
                event_loc = np.array([self.start_x, self.start_y])
                angle = get_smallest_angle(
                    event_loc - left_post, event_loc - right_post, angle_format="degree"
                )
                distance = math.dist([self.start_x, self.start_y], goal_loc)
                angle_distance = angle * distance

            if set_piece == "penalty":
                self._xt = 0.797
            elif set_piece == "corner_kick":
                self._xt = 0.049
            elif set_piece == "goal_kick":
                self._xt = 0.0
            elif set_piece == "kick_off":
                self._xt = 0.001
            elif set_piece == "throw_in":
                model = joblib.load(f"{path}/xT_throw_ins.pkl")
                self._xt = np.clip(
                    model.predict([[distance / 107.313]])[0], a_min=0, a_max=1
                )
            elif set_piece == "free_kick":
                model = joblib.load(f"{path}/xT_free_kicks.pkl")
                self._xt = np.clip(
                    model.predict(
                        [
                            [
                                angle / 24.265,
                                distance / 107.313,
                                angle_distance / 419.069,
                            ]
                        ]
                    )[0],
                    a_min=0,
                    a_max=1,
                )
            elif set_piece in ["no_set_piece", "unspecified_set_piece"]:
                model = joblib.load(f"{path}/xT_open_play.pkl")
                self._xt = np.clip(
                    model.predict(
                        [
                            [
                                angle / 125.493,
                                distance / 109.313,
                                angle_distance / 419.195,
                            ]
                        ]
                    )[0],
                    a_min=0,
                    a_max=1,
                )
            else:
                raise ValueError(
                    "set_piece should be one of ['penalty', 'corner_kick', 'goal_kick',"
                    " 'kick_off', 'throw_in', 'free_kick', 'no_set_piece', "
                    f"'unspecified_set_piece'], not {set_piece}"
                )

        return self._xt

    def __post_init__(self):
        if not isinstance(self.event_id, (np.integer, int)):
            raise TypeError(f"event_id should be int, not {type(self.event_id)}")

        if not isinstance(self.period_id, (np.integer, int)):
            raise TypeError(f"period_id should be int, not {type(self.period_id)}")

        if not isinstance(self.minutes, (np.integer, int, float, np.floating)):
            raise TypeError(f"minutes should be int, not {type(self.minutes)}")

        if not isinstance(self.seconds, (np.integer, int, float, np.floating)):
            raise TypeError(f"seconds should be int, not {type(self.seconds)}")

        if not pd.isnull(self.datetime) and not isinstance(self.datetime, pd.Timestamp):
            raise TypeError(
                f"datetime should be pd.Timestamp, not {type(self.datetime)}"
            )

        if not isinstance(self.start_x, (float, np.floating)):
            raise TypeError(f"x_start should be a float, not {type(self.start_x)}")

        if not isinstance(self.start_y, (float, np.floating)):
            raise TypeError(f"y_start should be a float, not {type(self.start_y)}")

        if not isinstance(self.team_id, (int, np.integer, str)):
            raise TypeError(f"team_id should be int, not {type(self.team_id)}")

        if not isinstance(self.team_side, str):
            raise TypeError(f"team_side should be str, not {type(self.team_side)}")
        if self.team_side not in ["home", "away"]:
            raise ValueError(
                f"team_side should be either 'home' or 'away', not {self.team_side}"
            )

        if self.pitch_size is not None:
            if not isinstance(self.pitch_size, (list, tuple, np.ndarray)):
                raise TypeError(
                    "pitch_size should be list, tuple or np.ndarray, "
                    f"not {type(self.pitch_size)}"
                )
            if len(self.pitch_size) != 2:
                raise ValueError(
                    f"pitch_size should have length 2, not {len(self.pitch_size)}"
                )
            if not all(
                [
                    isinstance(x, (int, np.integer, float, np.floating))
                    for x in self.pitch_size
                ]
            ):
                raise TypeError(
                    "pitch_size should contain only numbers, "
                    f"not {type(self.pitch_size[0])}"
                )

        if not isinstance(self._xt, (float, np.floating, int, np.integer)):
            raise TypeError(f"xT should be float, not {type(self._xt)}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseOnBallEvent):
            return False
        result = [
            self.event_id == other.event_id,
            self.period_id == other.period_id,
            self.minutes == other.minutes,
            round(self.seconds, 4) == round(other.seconds, 4),
            self.datetime == other.datetime
            if not pd.isnull(self.datetime)
            else pd.isnull(other.datetime),
            round(self.start_x, 4) == round(other.start_x, 4)
            if not pd.isnull(self.start_x)
            else pd.isnull(other.start_x),
            round(self.start_y, 4) == round(other.start_y, 4)
            if not pd.isnull(self.start_y)
            else pd.isnull(other.start_y),
            self.team_id == other.team_id,
            self.team_side == other.team_side,
            self._xt == other._xt,
            all([x == y for x, y in zip(self.pitch_size, other.pitch_size)])
            if self.pitch_size is not None
            else other.pitch_size is None,
        ]
        return all(result)
