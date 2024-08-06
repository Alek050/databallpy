from dataclasses import dataclass, fields

from databallpy.events.base_event import IndividualOnBallEvent
from databallpy.utils.utils import _values_are_equal_


@dataclass
class DribbleEvent(IndividualOnBallEvent):
    """Class for dribble events

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
        duel_type (str): type of duel that the event is related to. Should be in
            ["offensive", "defensive", "unspecified"].
        with_opponent (bool): whether the event was performed with an opponent or not

    Properties:
        xT (float): expected threat of the event. This is calculated using a model
            that is trained on the distance and angle to the goal, and the distance
            times theangle to the goal. See the notebook in the notebooks folder for
            more information on the model.
        base_df_attributes (list[str]): list of attributes that are used to create a
            DataFrame
    """

    duel_type: str
    with_opponent: bool

    def __post_init__(self):
        super().__post_init__()
        self._validate_inputs_dribble_event()

    def __eq__(self, other):
        if not isinstance(other, DribbleEvent):
            return False
        for field in fields(self):
            if not _values_are_equal_(
                getattr(self, field.name), getattr(other, field.name)
            ):
                return False

        return True

    @property
    def df_attributes(self) -> list[str]:
        base_attributes = super().base_df_attributes
        return base_attributes + ["duel_type", "with_opponent"]

    def copy(self):
        super_copy = super().copy()
        return DribbleEvent(
            event_id=super_copy.event_id,
            period_id=super_copy.period_id,
            minutes=super_copy.minutes,
            seconds=super_copy.seconds,
            datetime=super_copy.datetime,
            start_x=super_copy.start_x,
            start_y=super_copy.start_y,
            team_id=super_copy.team_id,
            team_side=super_copy.team_side,
            pitch_size=super_copy.pitch_size,
            player_id=super_copy.player_id,
            jersey=super_copy.jersey,
            outcome=super_copy.outcome,
            related_event_id=super_copy.related_event_id,
            body_part=super_copy.body_part,
            possession_type=super_copy.possession_type,
            set_piece=super_copy.set_piece,
            _xt=super_copy._xt,
            duel_type=self.duel_type,
            with_opponent=self.with_opponent,
        )

    def _validate_inputs_dribble_event(self):
        if not isinstance(self.duel_type, (str, type(None))):
            raise TypeError(
                f"duel_type should be str, got {type(self.duel_type)} instead"
            )
        if self.duel_type not in ["offensive", "defensive", "unspecified"]:
            raise ValueError(
                "duel_type should be in ['offensive', 'defensive', 'unspecified'],"
                f" got {self.duel_type} instead"
            )
        if not isinstance(self.with_opponent, bool):
            raise TypeError(
                f"with_opponent should be bool, got {type(self.with_opponent)} instead"
            )
