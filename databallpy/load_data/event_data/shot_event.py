from dataclasses import dataclass

import numpy as np
import pandas as pd

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
        player_id (int): id of the player who takes the shot
        shot_outcome (str): whether the shot is a goal or not on target or not
        y_target (float, optional): y location of the goal. Defaults to np.nan.
        z_target (float, optional): z location of the goal. Defaults to np.nan.
        body_part (str, optional): body part with which the shot is taken. Defaults to
            None.
        type_of_play (str, optional): open play or type of set piece. Defaults to
            "open_play".
        first_touch (bool, optional): whether the shot is taken with the first touch.
            Defaults to False.
        created_oppertunity (bool, optional): how the chance was created, assisted or
            individual play. Defaults to False.

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
    type_of_play: str = "open_play"
    first_touch: bool = False
    created_oppertunity: bool = False

    # id of the event that led to the shot. Note: opta id for opta event data
    related_event: int = MISSING_INT

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
            self.body_part == other.body_part,
            self.type_of_play == other.type_of_play,
            self.first_touch == other.first_touch,
            self.created_oppertunity == other.created_oppertunity,
            self.related_event == other.related_event,
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
            player_id=self.player_id,
            shot_outcome=self.shot_outcome,
            y_target=self.y_target,
            z_target=self.z_target,
            body_part=self.body_part,
            type_of_play=self.type_of_play,
            first_touch=self.first_touch,
            created_oppertunity=self.created_oppertunity,
            related_event=self.related_event,
        )
