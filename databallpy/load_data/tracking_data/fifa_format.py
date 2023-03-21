from databallpy.load_data.metadata import Metadata
from databallpy.utils import _to_int
from databallpy.load_data.tracking_data._add_player_tracking_data_to_dict import (
    _add_player_tracking_data_to_dict,
)
from databallpy.load_data.tracking_data._add_ball_data_to_dict import (
    _add_ball_data_to_dict,
)
from databallpy.load_data.tracking_data._insert_missing_rows import _insert_missing_rows
from databallpy.load_data.tracking_data._get_matchtime import _get_matchtime
from databallpy.load_data.tracking_data._normalize_playing_direction_tracking import (
    _normalize_playing_direction_tracking,
)
from databallpy.load_data.tracking_data._add_periods_to_tracking_data import (
    _add_periods_to_tracking_data,
)

from typing import Tuple
from bs4 import BeautifulSoup
from tqdm import tqdm

import numpy as np
import pandas as pd
import bs4

def load_fifa_format_data(fifa_loc:str, metadata_loc, verbose:bool=True) -> Tuple[pd.DataFrame, Metadata]:
    """

    Args:
        fifa_loc (str): _description_
        verbose (bool): _description_

    Returns:
        Tuple[pd.DataFrame, Metadata]: _description_
    """

    metadata = _get_metadata(metadata_loc, verbose)
    td_channels = _get_td_channels(metadata_loc, metadata)
    tracking_data = _get_tracking_data(fifa_loc, td_channels, metadata.pitch_dimensions, verbose)
    first_frame = metadata.periods_frames[metadata.periods_frames["start_frame"]>0]["start_frame"].min()
    last_frame = metadata.periods_frames["end_frame"].max()
    tracking_data = tracking_data[(tracking_data["frame"] >= first_frame) & (tracking_data["frame"] <= last_frame)]
    tracking_data = _normalize_playing_direction_tracking(
        tracking_data, metadata.periods_frames
    )
    tracking_data["period"] = _add_periods_to_tracking_data(
        tracking_data["frame"], metadata.periods_frames
    )
    tracking_data["matchtime_td"] = _get_matchtime(
        tracking_data["frame"], tracking_data["period"], metadata
    )

    return tracking_data, metadata

def _get_tracking_data(fifa_loc:str, td_channels:list, pitch_dimensions:list, verbose:bool) -> pd.DataFrame:
    """_summary_

    Args:
        fifa_loc (str): _description_
        verbose (bool): _description_

    Returns:
        pd.DataFrame: _description_
    """
    if verbose:
        print(f"Reading in {fifa_loc}", end="")
    file = open(fifa_loc, "r")
    lines = file.readlines()
    if verbose:
        print(" - Completed")
    file.close()
    
    size_lines = len(lines)
    data = {
        "frame": [np.nan] * size_lines,
        "ball_x": [np.nan] * size_lines,
        "ball_y": [np.nan] * size_lines,
        "ball_z": [np.nan] * size_lines,
        "ball_status": [None] * size_lines,
        "ball_posession": [None] * size_lines,
    }

    if verbose:
        print("Writing lines to dataframe:")
        lines = tqdm(lines)

    for idx, line in enumerate(lines):

        frame, players_home, players_away, ball_info = line.split(":")
        frame = _to_int(frame)
        data["frame"][idx] = frame
        players_info= players_home + ";" + players_away
        players = players_info.split(";")
        for i, player in enumerate(players):
            if player != ",":
                y, x = player.split(",")
            else:
                y, x = np.nan, np.nan
            team = td_channels[i].split("_")[0]
            shirt_num = td_channels[i].split("_")[1]
            data = _add_player_tracking_data_to_dict(team, shirt_num, x, y, data, idx)
        
        y, x, z, _, ball_status = ball_info.split(",")
        data = _add_ball_data_to_dict(x, y, z, None, ball_status, data, idx)

    df = pd.DataFrame(data)
    df["ball_status"] = np.where(df["ball_status"], "dead", "alive")
    for col in [x for x in df.columns if "_x" in x]:
        df[col] = df[col] - (pitch_dimensions[0] / 2)

    for col in [x for x in df.columns if "_y" in x]:
        df[col] = df[col] - (pitch_dimensions[1] / 2)

    df = _insert_missing_rows(df, "frame")

    return df


def _get_td_channels(metadata_loc:str, metadata:Metadata)->list: 
    
    file = open(metadata_loc, "r", encoding="UTF-8").read()
    soup = BeautifulSoup(file, "xml")
    

    res = []
    for channel in soup.find_all("PlayerChannel"):
        player_id = channel.attrs["id"].split("_")[0]
        value = channel.attrs["id"].split("_")[1]
        if "y" in value:
            continue
        home_mask = metadata.home_players["id"] == player_id
        away_mask = metadata.away_players["id"] == player_id
        if home_mask.any():
            team = "home"
            shirt_num = metadata.home_players.loc[home_mask, "shirt_num"].values[0]
        else:
            team = "away"
            shirt_num = metadata.away_players.loc[away_mask, "shirt_num"].values[0]
        res.append(f"{team}_{shirt_num}")

    return res
        

def _get_metadata(metadata_loc:str, verbose:bool) -> Metadata:
    """_summary_

    Args:
        meta_data_loc (str): _description_
        verbose (bool): _description_

    Returns:
        Metadata: _description_
    """
    
    file = open(metadata_loc, "r", encoding="UTF-8").read()
    soup = BeautifulSoup(file, "xml")
    
    periods_dict = {"period": [1,2,3,4,5],
                    "start_frame": [-999]*5,
                    "end_frame": [-999]*5,
                    "start_datetime_td": [pd.to_datetime("NaT")]*5,
                    "end_datetime_td": [pd.to_datetime("NaT")]*5
                    }
    periods = soup.find_all("Session")
    
    i=0
    for period in periods:
        if period.SessionType.text == "Period":
            values = [int(x.text) for x in period.find_all("Value")]
            periods_dict["start_frame"][i] = _to_int(values[0])
            periods_dict["end_frame"][i] = _to_int(values[1])
            periods_dict["start_datetime_td"][i] = pd.to_datetime(period.find("Start").text)
            periods_dict["end_datetime_td"][i] = pd.to_datetime(period.find("End").text)
            i += 1
    periods_frames = pd.DataFrame(periods_dict)

    home_team = soup.Teams.find_all("Team")[0]
    home_team_id = home_team.attrs["id"]
    home_team_name = home_team.Name.text
    home_team_player_data = soup.find_all("Player", {"teamId": home_team_id})
    home_players = _get_player_data(home_team_player_data)
    home_score = int(soup.LocalTeamScore.text)

    away_team = soup.Teams.find_all("Team")[1]
    away_team_id = away_team.attrs["id"]
    away_team_name = away_team.Name.text
    away_team_player_data = soup.find_all("Player", {"teamId": away_team_id})
    away_players = _get_player_data(away_team_player_data)
    away_score = int(soup.VisitingTeamScore.text)
    
    metadata = Metadata(
        match_id = int(soup.Session.attrs["id"]),
        pitch_dimensions = [float(soup.MatchParameters.FieldSize.Length.text),
                            float(soup.MatchParameters.FieldSize.Width.text)],
        periods_frames = periods_frames,

        frame_rate = int(soup.FrameRate.text),

        home_team_id = home_team_id,
        home_team_name = home_team_name,
        home_players = home_players,
        home_score = home_score, 
        home_formation = None,

        away_team_id = away_team_id,
        away_team_name = away_team_name,
        away_players = away_players,
        away_score = away_score,
        away_formation = None,
    )
    return metadata

def _get_player_data(team: bs4.element.Tag) -> pd.DataFrame:
    """Function that creates a df containing info on all players for a team
    Args:
        team (bs4.element.Tag): containing info no all players of a team
    Returns:
        pd.DataFrame: contains all player information for a team
    """

    player_dict = {
        "id": [],
        "full_name": [],
        "shirt_num": [],
        "player_type": [],
        "start_frame": [],
        "end_frame": []
    }
    for player in team:
        player_dict["id"].append(player["id"])
        player_dict["full_name"].append(player.Name.text)
        player_dict["shirt_num"].append(int(player.ShirtNumber.text))
        values = [x.text for x in player.find_all("Value")]
        player_dict["player_type"].append(values[0])
        player_dict["start_frame"].append(int(values[1]))
        player_dict["end_frame"].append(int(values[2]))
    df = pd.DataFrame(player_dict)

    return df

    