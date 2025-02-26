from dataclasses import dataclass, fields

import numpy as np

from databallpy.events.base_event import IndividualOnBallEvent
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

    Attributes:
        xt (float): expected threat of the event. This is calculated using a
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
        ]
        for name in names:
            value = getattr(self, name)
            if not isinstance(value, (float, np.floating)):
                raise TypeError(f"{name} should be float, not {type(value)}")

        if not isinstance(self.pass_type, str):
            raise TypeError(f"pass_type should be str, not {type(self.pass_type)}")

        if self.pass_type not in DATABALLPY_PASS_TYPES:
            raise ValueError(
                f"pass_type should be one of {DATABALLPY_PASS_TYPES}, "
                f"not {self.pass_type}"
            )
