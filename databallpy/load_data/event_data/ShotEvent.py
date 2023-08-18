from databallpy.load_data.event_data.base_event_class import Event
from dataclasses import dataclass
import numpy as np

@dataclass
class ShotEvent(Event):
    shot_outcome: str # whether the shot is a goal or not on target or not
    y_target: float = np.nan  # y location of the goal
    z_target: float = np.nan  # z location of the goal
    body_part: str = None  # body part with which the shot is taken
    type_of_play: str = "open_play"  # open play or type of set piece
