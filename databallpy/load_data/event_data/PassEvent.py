from dataclasses import dataclass
from databallpy.load_data.event_data.base_event_class import Event

@dataclass
class PassEvent(Event):
    outcome: str
    team_id: int
    player_id: int
    x_end: float
    y_end: float
    length: float
    angle: float
    pass_type: str
    set_piece: str

    def __post_init__(self):
        if not isinstance(self.outcome, str):
            raise TypeError(f"outcome should be str, not {type(self.outcome)}")
        
        if not isinstance(self.team_id, int):
            raise TypeError(f"team_id should be int, not {type(self.team_id)}")
        
        if not isinstance(self.player_id, int):
            raise TypeError(f"player_id should be int, not {type(self.player_id)}")
        
        if not isinstance(self.x_end, float):
            raise TypeError(f"x_end should be float, not {type(self.x_end)}")
        
        if not isinstance(self.y_end, float):
            raise TypeError(f"y_end should be float, not {type(self.y_end)}")
        
        if not isinstance(self.length, float):
            raise TypeError(f"length should be float, not {type(self.length)}")
        
        if not isinstance(self.angle, float):
            raise TypeError(f"angle should be float, not {type(self.angle)}")
        
        if not isinstance(self.pass_type, str):
            raise TypeError(f"pass_type should be str, not {type(self.pass_type)}")
        
        if not isinstance(self.set_piece, str):
            raise TypeError(f"set_piece should be str, not {type(self.set_piece)}")