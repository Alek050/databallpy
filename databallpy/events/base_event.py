import os
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from databallpy.models.utils import get_xt_prediction
from databallpy.utils.constants import (
    DATABALLPY_POSSESSION_TYPES,
    DATABALLPY_SET_PIECES,
    DATBALLPY_BODY_PARTS,
)
from databallpy.utils.utils import _copy_value_, _values_are_equal_

path = os.path.join(os.path.dirname(__file__), "..", "models")
FREE_KICK_XT = np.load(f"{path}/free_kick_xT.npy")
OPEN_PLAY_XT = np.load(f"{path}/open_play_xT.npy")
THROW_IN_XT = np.load(f"{path}/throw_in_xT.npy")


@dataclass
class IndividualCloseToBallEvent:
    """This is the base close to ball event class from which the specific event classes
    are inherited. It containts all the basic information that is available for every
    event.

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
        related_event_id (int | str | list | None): id of the event that the event is related
            to the current event.

    Properties:
        base_df_attributes (list[str]): list of attributes that are used to create a
            DataFrame
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
    player_id: int | str
    jersey: int
    outcome: bool
    related_event_id: int | str | list | None

    @property
    def base_df_attributes(self) -> list[str]:
        return [
            "event_id",
            "period_id",
            "minutes",
            "seconds",
            "datetime",
            "start_x",
            "start_y",
            "team_id",
            "team_side",
            "player_id",
            "jersey",
            "outcome",
            "related_event_id",
        ]

    def __post_init__(self):
        self._validate_inputs_close_to_ball_event()

    def copy(self):
        copied_kwargs = {
            f.name: _copy_value_(getattr(self, f.name)) for f in fields(self)
        }
        return IndividualCloseToBallEvent(**copied_kwargs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IndividualCloseToBallEvent):
            return False
        for field in fields(self):
            if not _values_are_equal_(
                getattr(self, field.name), getattr(other, field.name)
            ):
                return False

        return True

    def _validate_inputs_close_to_ball_event(self):
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
                f"not {type(self.pitch_size[0])}, {type(self.pitch_size[1])}"
            )

        self.pitch_size = [float(x) for x in self.pitch_size]

        if not isinstance(self.player_id, (int, np.integer, str)):
            raise TypeError(f"player_id should be int, not {type(self.player_id)}")

        if not isinstance(self.jersey, (int, np.integer)):
            raise TypeError(f"jersey should be int, not {type(self.jersey)}")

        if not isinstance(self.outcome, bool):
            raise TypeError(f"outcome should be bool, not {type(self.outcome)}")

        if not isinstance(
            self.related_event_id, (int, np.integer, str, list, type(None))
        ):
            raise TypeError(
                f"related_event_id should be int, not {type(self.related_event_id)}"
            )


@dataclass
class IndividualOnBallEvent(IndividualCloseToBallEvent):
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

    Properties:
        xt (float): expected threat of the event. This is calculated using a model
            that is trained on the distance and angle to the goal, and the distance
            times theangle to the goal. See the notebook in the notebooks folder for
            more information on the model.
        base_df_attributes (list[str]): list of attributes that are used to create a
            DataFrame
    """

    body_part: str
    possession_type: str
    set_piece: str
    _xt: float

    def __post_init__(self):
        self._validate_inputs_close_to_ball_event()
        self._validate_inputs_on_ball_event()
        self._xt = self.xt

    @property
    def base_df_attributes(self) -> list[str]:
        return super().base_df_attributes + [
            "xt",
            "body_part",
            "possession_type",
            "set_piece",
        ]

    @property
    def xt(self) -> float:
        """This property returns the expected threat of the event. It is calculated
        using a model that is trained on the distance and angle to the goal, and the
        distance times the angle to the goal. See the notebook in the notebooks folder
        for more information on the model.

        Returns:
            float: expected threat of the event
        """
        if self._xt < 0.0 or np.isnan(self._xt):
            x = self.start_x if self.team_side == "home" else -self.start_x
            y = self.start_y if self.team_side == "home" else -self.start_y

            if self.set_piece == "penalty":
                self._xt = 0.797
            elif self.set_piece == "corner_kick":
                self._xt = 0.049
            elif self.set_piece == "goal_kick":
                self._xt = 0.0
            elif self.set_piece == "kick_off":
                self._xt = 0.001
            elif self.set_piece == "throw_in":
                self._xt = get_xt_prediction(x, y, THROW_IN_XT)
            elif self.set_piece == "free_kick":
                self._xt = get_xt_prediction(x, y, FREE_KICK_XT)
            elif self.set_piece in ["no_set_piece", "unspecified"]:
                self._xt = get_xt_prediction(x, y, OPEN_PLAY_XT)
            else:
                raise ValueError(
                    f"set_piece should be one of {DATABALLPY_SET_PIECES}, "
                    f"not {self.set_piece}"
                )

        return self._xt

    def copy(self):
        copied_kwargs = {
            f.name: _copy_value_(getattr(self, f.name)) for f in fields(self)
        }
        return IndividualOnBallEvent(**copied_kwargs)

    def __eq__(self, other):
        if not isinstance(other, IndividualOnBallEvent):
            return False
        for field in fields(self):
            if not _values_are_equal_(
                getattr(self, field.name), getattr(other, field.name)
            ):
                return False

        return True

    def _validate_inputs_on_ball_event(self):
        if not isinstance(self.body_part, str):
            raise TypeError(f"body_part should be str, not {type(self.body_part)}")

        if self.body_part not in DATBALLPY_BODY_PARTS:
            raise ValueError(
                f"body_part should be one of {DATBALLPY_BODY_PARTS},"
                f" not {self.body_part}"
            )

        if not isinstance(self.possession_type, str):
            raise TypeError(
                f"possession_type should be str, not {type(self.possession_type)}"
            )

        if self.possession_type not in DATABALLPY_POSSESSION_TYPES:
            raise ValueError(
                f"possession_type should be one of {DATABALLPY_POSSESSION_TYPES},"
                f" not {self.possession_type}"
            )

        if not isinstance(self.set_piece, str):
            raise TypeError(f"set_piece should be str, not {type(self.set_piece)}")

        if self.set_piece not in DATABALLPY_SET_PIECES:
            raise ValueError(
                f"set_piece should be one of {DATABALLPY_SET_PIECES},"
                f" not {self.set_piece}"
            )

        if not isinstance(self._xt, (float, np.floating, int, np.integer)):
            raise TypeError(f"xt should be float, not {type(self._xt)}")
