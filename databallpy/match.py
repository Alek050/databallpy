from dataclasses import dataclass
import pandas as pd
from typing import Tuple
from databallpy.load_data.tracking_data.tracab import load_tracab_tracking_data
from databallpy.load_data.event_data.opta import load_opta_event_data

@dataclass
class Match:
    tracking_data:pd.DataFrame
    event_data:pd.DataFrame
    pitch_dimensions:Tuple[float, float]
    periods: pd.DataFrame
    frame_rate:int
    home_team_id: int
    home_team_name: str
    home_players: pd.DataFrame
    home_score: int
    home_formation: str
    away_team_id: int
    away_team_name: str
    away_players: pd.DataFrame
    away_score: int
    away_formation: str

    @property
    def name(self):
        return f"{self.home_team_name} {self.home_score} - {self.away_score} {self.away_team_name}"

    @property
    def home_players_column_ids(self):
        return [id for id in self.tracking_data.columns if "home" in id]

    @property
    def away_players_column_ids(self):
        return [id for id in self.tracking_data.columns if "away" in id]


def get_match(
    *, 
    tracking_data_loc:str, 
    tracking_metadata_loc:str, 
    event_data_loc:str, 
    event_metadata_loc:str, 
    tracking_data_provider:str, 
    event_data_provider:str
    ):

    assert tracking_data_provider in ["tracab"], f"We do not support {tracking_data_provider} as tracking data provider yet, please open an issue in our Github repository."
    assert event_data_provider in ["opta"], f"We do not supper {event_data_provider} as event data provider yet, please open an issue in our Github repository."  

    # Get tracking data and tracking metadata
    if tracking_data_provider == "tracab":
        tracking_data, tracking_metadata = load_tracab_tracking_data(tracking_data_loc, tracking_metadata_loc)
    
    # Get event data and event metadata
    if event_data_provider == "opta":
        event_data, event_metadata = load_opta_event_data(f7_loc=event_metadata_loc, f24_loc=event_data_loc)
    
    

