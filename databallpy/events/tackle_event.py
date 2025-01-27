from dataclasses import dataclass, fields

from databallpy.events import IndividualCloseToBallEvent
from databallpy.utils.utils import _copy_value_, _values_are_equal_


@dataclass
class TackleEvent(IndividualCloseToBallEvent):
    """
    Class to represent a tackle event in a soccer game.

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

    Properties:
        base_df_attributes (list[str]): list of attributes that are used to create a
            DataFrame
    """

    def __eq__(self, other):
        if not isinstance(other, TackleEvent):
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
        return TackleEvent(**copied_kwargs)

    @property
    def df_attributes(self):
        return super().base_df_attributes
