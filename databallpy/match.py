from dataclasses import dataclass
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from databallpy.load_data.event_data.opta import load_opta_event_data
from databallpy.load_data.tracking_data.tracab import load_tracab_tracking_data


@dataclass
class Match:
    """This is the match class. It contains all information of the match and has
    some simple functions to easily obtain information about the match.

    Args:
        tracking_data (pd.DataFrame): Tracking data of the match.
        tracking_data_provider (str): Provider of the tracking data.
        event_data (pd.DataFrame): Event data of the match.
        event_data_provider (str): Provider of the event data.
        pitch_dimensions (Tuple): The size of the pitch in meters in x and y direction.
        periods (pd.DataFrame): The start and end idicators of all periods.
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
        name (str): The home and away team name and score (e.g "nameH 3 - 1 nameA").
        home_players_column_ids (list): All column ids of the tracking data that refer
                                        to information about the home team players.
        away_players_column_ids (list): All column ids of the tracking data that refer
                                        to information about the away team players.

    Funcs
        player_column_id_to_full_name: Simple function to get the full name of a player
                                       from the column id
    """

    tracking_data: pd.DataFrame
    tracking_data_provider: str
    event_data: pd.DataFrame
    event_data_provider: str
    pitch_dimensions: List[float]
    periods: pd.DataFrame
    frame_rate: int
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
        home_text = f"{self.home_team_name} {self.home_score}"
        away_text = f"{self.away_score} {self.away_team_name}"
        return f"{home_text} - {away_text}"

    @property
    def home_players_column_ids(self) -> List[str]:
        return [id for id in self.tracking_data.columns if "home" in id]

    @property
    def away_players_column_ids(self) -> List[str]:
        return [id for id in self.tracking_data.columns if "away" in id]

    def player_column_id_to_full_name(self, column_id: str) -> str:
        """Simple function to get the full name of a player from the column id

        Args:
            column_id (str): the column id of a player, for instance "home_1"

        Returns:
            str: full name of the player
        """
        shirt_num = int(column_id.split("_")[1])
        if column_id[:4] == "home":
            return self.home_players.loc[
                self.home_players["shirt_num"] == shirt_num, "full_name"
            ].iloc[0]
        else:
            return self.away_players.loc[
                self.away_players["shirt_num"] == shirt_num, "full_name"
            ].iloc[0]

    def full_name_to_player_column_id(self, full_name: str) -> str:
        """Simple function to get the column id based on full name

        Args:
            full_name (str): full name of the player

        Returns:
            str: column id of the player, for instance "home_1"
        """
        if (self.home_players["full_name"].eq(full_name)).any():
            num = self.home_players[self.home_players["full_name"] == full_name]["shirt_num"].iloc[0]
            return f"home_{num}"
        elif (self.away_players["full_name"].eq(full_name)).any():
            num = self.away_players[self.away_players["full_name"] == full_name]["shirt_num"].iloc[0]
            return f"away_{num}"
        else:
            return ""

    def __eq__(self, other):
        if isinstance(other, Match):
            res = [
                self.tracking_data.equals(other.tracking_data),
                self.tracking_data_provider == other.tracking_data_provider,
                self.event_data.equals(other.event_data),
                self.pitch_dimensions == other.pitch_dimensions,
                self.periods.equals(other.periods),
                self.frame_rate == other.frame_rate,
                self.home_team_id == other.home_team_id,
                self.home_team_name == other.home_team_name,
                self.home_formation == other.home_formation,
                self.home_players.equals(other.home_players),
                self.home_score == other.home_score,
                self.away_team_id == other.away_team_id,
                self.away_team_name == other.away_team_name,
                self.away_formation == other.away_formation,
                self.away_players.equals(other.away_players),
                self.away_score == other.away_score,
            ]
            return all(res)
        else:
            return False


def get_match(
    *,
    tracking_data_loc: str,
    tracking_metadata_loc: str,
    event_data_loc: str,
    event_metadata_loc: str,
    tracking_data_provider: str,
    event_data_provider: str,
):

    assert tracking_data_provider in [
        "tracab"
    ], f"We do not support '{tracking_data_provider}' as tracking data provider yet, "
    "please open an issue in our Github repository."
    assert event_data_provider in [
        "opta"
    ], f"We do not supper '{event_data_provider}' as event data provider yet, "
    "please open an issue in our Github repository."

    # Get event data and event metadata
    if event_data_provider == "opta":
        event_data, event_metadata = load_opta_event_data(
            f7_loc=event_metadata_loc, f24_loc=event_data_loc
        )

    # Get tracking data and tracking metadata
    if tracking_data_provider == "tracab":
        tracking_data, tracking_metadata = load_tracab_tracking_data(
            tracking_data_loc, tracking_metadata_loc
        )

    # Check if the event data is scaled the right way
    if not tracking_metadata.pitch_dimensions == event_metadata.pitch_dimensions:
        x_correction = (
            tracking_metadata.pitch_dimensions[0] / event_metadata.pitch_dimensions[0]
        )
        y_correction = (
            tracking_metadata.pitch_dimensions[1] / event_metadata.pitch_dimensions[1]
        )
        event_data["start_x"] *= x_correction
        event_data["start_y"] *= y_correction

    # Merge periods
    merged_periods = pd.concat(
        (
            tracking_metadata.periods_frames,
            event_metadata.periods_frames.drop("period", axis=1),
        ),
        axis=1,
    )

    # Merged player info
    home_players = tracking_metadata.home_players.merge(
        event_metadata.home_players[["id", "formation_place", "position", "starter"]],
        on="id",
    )
    away_players = tracking_metadata.away_players.merge(
        event_metadata.away_players[["id", "formation_place", "position", "starter"]],
        on="id",
    )

    home_players_td = tracking_metadata.home_players
    home_players_ed = event_metadata.home_players
    away_players_td = tracking_metadata.away_players
    away_players_ed = event_metadata.away_players

    full_names_dict = {}
    for num in home_players_td["shirt_num"].unique():
        name_td = home_players_td[home_players_td["shirt_num"] == num]["full_name"].iloc[0]
        name_ed = home_players_ed[home_players_ed["shirt_num"] == num]["full_name"].iloc[0]
        if name_ed != name_td:
            full_names_dict[name_ed] = name_td

    for num in away_players_td["shirt_num"].unique():
        name_td = away_players_td[away_players_td["shirt_num"] == num]["full_name"].iloc[0]
        name_ed = away_players_ed[away_players_ed["shirt_num"] == num]["full_name"].iloc[0]  
        if name_ed != name_td:
            full_names_dict[name_ed] = name_td

    event_data["player_name"] = event_data["player_name"].map(full_names_dict).fillna(event_data["player_name"])

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


def synchronise_event_and_tracking_data(match):
    tracking_data = match.tracking_data
    event_data = match.event_data

    date = np.datetime64(str(match.periods.iloc[1,3])[:10])
    tracking_data["datetime"] = [date + np.timedelta64(int(x/25*1000), "ms") for x in tracking_data["timestamp"]]

    first_events = event_data.iloc[4:10, :].reset_index(drop=False)
    first_events = first_events[first_events["event"] == "pass"].reset_index()
    first_tracking = tracking_data.iloc[:380, :].reset_index(drop=False)
    dist_mat = np.zeros((len(first_tracking), len(first_events)))

    for i_outer, tracking_row in first_tracking.iterrows():
        for i_inner, event_row in first_events.iterrows():
            if type(event_row["player_name"]) == str:
                column_id_player = match.full_name_to_player_column_id(full_name=event_row["player_name"])
            else:
                column_id_player = np.nan
            dist_mat[i_outer, i_inner] = _calculate_distance(tracking_row, event_row, column_id_player)
    
    import pdb;pdb.set_trace()
    den = max(dist_mat.min(axis=0))*1.2
    dist_mat = np.minimum(np.exp(-dist_mat/den), np.ones(np.shape(dist_mat)))
    event_frame_dict = nw(dist_mat)
    tracking_data["event"] = np.nan
    tracking_data["event_id"] = np.nan
    for event, frame in event_frame_dict.items():
        event_id = first_events.iloc[event]["event_id"]
        event_type = first_events.iloc[event]["event"]
        tracking_data["event"].iloc[frame] = event_type
        tracking_data["event_id"].iloc[frame] = int(event_id)

    match.tracking_data = tracking_data
    
    return match

def _calculate_distance(tracking_frame, event, column_id_player):
    time_diff = (tracking_frame["datetime"] - event["datetime"])/np.timedelta64(1, "s")
    ball_loc_diff_x = tracking_frame["ball_x"] - event["start_x"]
    ball_loc_diff_y = tracking_frame["ball_y"] - event["start_y"]
    ball_loc_diff = np.hypot(ball_loc_diff_x, ball_loc_diff_y)
    if type(column_id_player) == str:
        player_ball_diff_x = tracking_frame["ball_x"] - tracking_frame[f"{column_id_player}_x"]
        player_ball_diff_y = tracking_frame["ball_y"] - tracking_frame[f"{column_id_player}_y"]
        player_ball_diff = np.hypot(player_ball_diff_x, player_ball_diff_y)
    else:
        player_ball_diff = 0
    
    return np.abs(time_diff) + ball_loc_diff/5 + player_ball_diff

def nw(dist_mat, gap_event = -1, gap_frame = 1):
    """Based on: https://gist.github.com/slowkow/06c6dba9180d013dfd82bec217d22eb5

    Args:
        dist_mat (_type_): _description_
        gap_event (int, optional): _description_. Defaults to -1.
        gap_frame (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    n_frames = np.shape(dist_mat)[0]
    n_events = np.shape(dist_mat)[1]
    
    F = np.zeros((n_frames + 1, n_events + 1))
    F[:,0] = np.linspace(0, n_frames * gap_frame, n_frames + 1)
    F[0,:] = np.linspace(0, n_events * gap_event, n_events + 1)
    
    # Pointer matrix
    P = np.zeros((n_frames + 1, n_events + 1))
    P[:,0] = 3
    P[0,:] = 4
   
    t = np.zeros(3)
    for i in range(n_frames):
        for j in range(n_events):
            t[0] = F[i,j] + dist_mat[i,j]
            t[1] = F[i,j+1] + gap_frame #top + gap frame
            t[2] = F[i+1,j] + gap_event #left + gap event
            tmax = np.max(t)
            F[i+1,j+1] = tmax
            
            if t[0] == tmax: #match
                P[i+1,j+1] += 2
            if t[1] == tmax: #frame unassigned
                P[i+1,j+1] += 3
            if t[2] == tmax: #event unassigned
                P[i+1,j+1] += 4

    # Trace through an optimal alignment.
    i = n_frames 
    j = n_events 
    frames = [] #frames
    events = [] #event
    while i > 0 or j > 0:
        if P[i,j] in [2, 5, 6, 9]: #2 was added, match
            frames.append(i)
            events.append(j)
            i -= 1
            j -= 1
        elif P[i,j] in [3, 5, 7, 9]: #3 was added, frame unassigned
            frames.append(i)
            events.append(0)
            i -= 1
        elif P[i,j] in [4, 6, 7, 9]: #4 was added, event unassigned
            frames.append(0)
            events.append(j)
            j -= 1
    
    frames = frames[::-1]
    events = events[::-1]
    
    idx_events = [idx for idx,i in enumerate(events) if i > 0]
    event_frame_dict = {}
    for i in idx_events:
        event_frame_dict[events[i]-1] = frames[i]-1
    
    import pdb;pdb.set_trace()
    return event_frame_dict

