from dataclasses import dataclass
import pandas as pd
from typing import List
from databallpy.load_data.tracking_data.tracab import load_tracab_tracking_data
from databallpy.load_data.event_data.opta import load_opta_event_data

@dataclass
class Match:
    """This is the match class. It contains all information of the match and has some simple functions to easily obtain information about the match.

    Args:
        tracking_data (pd.DataFrame): Tracking data of the match.
        tracking_data_provider (str): Provider of the tracking data.
        event_data (pd.DataFrame): Event data of the match.
        event_data_provider (str): Provider of the event data.
        pitch_dimensions (Tuple): The size of the pitch in meters in x and y direction.
        periods (pd.DataFrame): The start and end indicatiors of all periods.
        frame_rate (int): The frequency of the tracking data.
        home_team_id (int): The id of the home team.
        home_team_name (str): The name of the home team.
        home_players (pd.DataFrame): Information about the home players.
        home_score (int): Number of goals scored over the match by the home team.
        home_formation (str): Indication of the formation of the home team.
        away_team_id (int): The id of the away team.
        away_team_name (str): The name of the away team.
        away_players (pd.DataFrame): Information about the away players.
        away_score (int): Number of goals scored over the match by the away team.
        away_formation (str): Indication of the formation of the away team.
        name (str): The home and away team name and score.
        home_players_column_ids (list): All column ids of the tracking data that refer 
                                        to information about the home team players.
        away_players_column_ids (list): All column ids of the tracking data that refer 
                                        to information about the away team players.

    Funcs
        player_column_id_to_full_name: Simple function to get the full name of a player 
                                       from the column id
    """
    tracking_data:pd.DataFrame
    tracking_data_provider:str
    event_data:pd.DataFrame
    event_data_provider:str
    pitch_dimensions:List[float, float]
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
    def name(self) -> str:
        return f"{self.home_team_name} {self.home_score} - {self.away_score} {self.away_team_name}"

    @property
    def home_players_column_ids(self) -> List[str]:
        return [id for id in self.tracking_data.columns if "home" in id]

    @property
    def away_players_column_ids(self) -> List[str]:
        return [id for id in self.tracking_data.columns if "away" in id]
    
    def player_column_id_to_full_name(self, column_id:str) -> str:
        """Simple function to get the full name of a player from the column id

        Args:
            column_id (str): the column id of a player, for instance "home_1"

        Returns:
            str: full name of the player
        """
        shirt_num = int(column_id.split("_")[1])
        if column_id[:4] == "home":
            return self.home_players.loc[self.home_players["shirt_num"]==shirt_num, "full_name"].iloc[0]
        else:
            return self.away_players.loc[self.away_players["shirt_num"]==shirt_num, "full_name"].iloc[0]


def get_match(
    *, 
    tracking_data_loc:str, 
    tracking_metadata_loc:str, 
    event_data_loc:str, 
    event_metadata_loc:str, 
    tracking_data_provider:str, 
    event_data_provider:str
    ):

    assert tracking_data_provider in ["tracab"], f"We do not support '{tracking_data_provider}' as tracking data provider yet, please open an issue in our Github repository."
    assert event_data_provider in ["opta"], f"We do not supper '{event_data_provider}' as event data provider yet, please open an issue in our Github repository."  

    # Get event data and event metadata
    if event_data_provider == "opta":
        event_data, event_metadata = load_opta_event_data(f7_loc=event_metadata_loc, f24_loc=event_data_loc)

    # Get tracking data and tracking metadata
    if tracking_data_provider == "tracab":
        tracking_data, tracking_metadata = load_tracab_tracking_data(tracking_data_loc, tracking_metadata_loc)
    
    # Check if the event data is scaled the right way
    if not tracking_metadata.pitch_dimensions == event_metadata.pitch_dimensions:
        x_correction = tracking_metadata.pitch_dimensions[0] / event_metadata.pitch_dimensions[0]
        y_correction = tracking_metadata.pitch_dimensions[1] / event_metadata.pitch_dimensions[1]
        event_data["start_x"] *= x_correction
        event_data["start_y"] *= y_correction

    # Merge periods
    merged_periods = pd.concat(
        (
            tracking_metadata.periods_frames, 
            event_metadata.periods_frames.drop("period", axis=1)
            ), 
            axis=1
        )

    # Merged player info
    home_players = tracking_metadata.home_players.merge(event_metadata.home_players)
    away_players = tracking_metadata.away_players.merge(event_metadata.away_players)

    match = Match(
        tracking_data=tracking_data,
        tracking_data_provider=tracking_data_provider,
        event_data=event_data,
        event_data_provider=event_data_provider,
        pitch_dimensions=tracking_metadata.pitch_dimensions,
        periods=merged_periods,
        frame_rate=tracking_metadata.frame_rate,
        home_team_id=event_metadata.home_team_id,
        home_formation=event_metadata.home_formation,
        home_score=event_metadata.home_score,
        home_team_name=event_metadata.home_team_name,
        home_players=home_players,
        away_team_id=event_metadata.away_team_id,
        away_formation=event_metadata.away_formation,
        away_score=event_metadata.away_score,
        away_team_name=event_metadata.away_team_name,
        away_players=away_players,
    )
    
    
    return match
    


