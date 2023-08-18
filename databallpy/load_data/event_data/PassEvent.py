from dataclasses import dataclass
from databallpy.load_data.event_data.base_event_class import Event

@dataclass
class PassEvent(Event):
    outcome: int
    team_id: int
    player_id: int
    x_end: float
    y_end: float