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
        shot_outcome (str): whether the shot is a goal or not on target or not.
            Possible values: "goal", "own_goal", "miss_off_target", "miss_on_target",
            "blocked", "miss_hit_post"
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

    def __post_init__(self):
        super().__post_init__()
        self._check_datatypes()

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
            self.related_event_id == other.related_event_id,
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
            related_event_id=self.related_event_id,
        )

    def _check_datatypes(self):
        if not isinstance(self.player_id, int):
            raise TypeError(f"player_id should be int, got {type(self.player_id)}")
        if not isinstance(self.shot_outcome, str):
            raise TypeError(f"shot_outcome should be str, got {type(self.shot_outcome)}")
        if not self.shot_outcome in ["goal", "miss_off_target", "miss_hit_post", "miss_on_target", "blocked", "own_goal"]:
            raise ValueError(f"shot_outcome should be goal, miss_off_target, miss_hit_post, miss_on_target, blocked or own_goal, got {self.shot_outcome}")
        if not isinstance(self.y_target, float):
            raise TypeError(f"y_target should be float or int, got {type(self.y_target)}")
        if not isinstance(self.z_target, float):
            raise TypeError(f"z_target should be float or int, got {type(self.z_target)}")
        if not isinstance(self.body_part, (str, type(None))):
            raise TypeError(f"body_part should be str or None, got {type(self.body_part)}")
        if not self.body_part in ["left_foot", "right_foot", "head", "other"]:
            raise ValueError(f"body_part should be left_foot, right_foot, head or other, got {self.body_part}")
        if not isinstance(self.type_of_play, str):
            raise TypeError(f"type_of_play should be str, got {type(self.type_of_play)}")
        if not self.type_of_play in ["penalty", "regular_play", "counter_attack", "crossed_free_kick", "corner_kick", "free_kick"]:
            raise ValueError(f"type_of_play should be penalty, regular_play, counter_attack, crossed_free_kick, corner_kick or free_kick, got {self.type_of_play}")
        if not isinstance(self.first_touch, bool):
            raise TypeError(f"first_touch should be bool, got {type(self.first_touch)}")
        if not isinstance(self.created_oppertunity, (str, type(None))):
            raise TypeError(
                f"created_oppertunity should be str or None, got {type(self.created_oppertunity)}"
            )
        if not self.created_oppertunity in ["assisted", "individual_play", "regular_play"]:
            raise ValueError(f"created_oppertunity should be assisted, regular_play, or individual_play, got {self.created_oppertunity}")
        if not isinstance(self.related_event_id, int):
            raise TypeError(f"related_event_id should be int, got {type(self.related_event_id)}")
        
