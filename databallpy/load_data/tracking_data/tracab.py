from databallpy.load_data.metadata import Metadata

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


def _get_lines_from_dat(tracab_loc: str, verbose: bool) -> list:
    """Function that reads .dat file and returns a list with all lines

    Args:
        tracab_loc (str): location of the tracking_data.dat file
        verbose (bool): whether to print info in terminal

    Returns:
        list: list containing all lines from .dat file
    """

    if verbose:
        match = tracab_loc.split("\\")[-1]
        print(f"Reading in {match}:")

    file = open(tracab_loc, "r")
    lines = file.readlines()

    return lines

def _add_player_data_to_dict(player: str, data: dict, idx: int) -> dict:
    """Function that adds the data of one player to the data dict for one frame

    Args:
        player (str): containing data from player
        data (dict): data dictionary to write results to
        idx (int): indicates position in data dictionary

    Returns:
        dict: contains all tracking data
    """
    
    team_id, _, shirt_num, x, y, speed = player.split(",")
        
    team_ids = {0: "home", 1: "away"}
    team = team_ids.get(int(team_id))
    if team is None: #player is unknown or referee
        return data

    if f"{team}_{shirt_num}_x" not in data.keys(): #create keys for new player
        data[f"{team}_{shirt_num}_x"] = [np.nan] * len(data["timestamp"])
        data[f"{team}_{shirt_num}_y"] = [np.nan] * len(data["timestamp"])
        data[f"{team}_{shirt_num}_speed"] = [np.nan] * len(data["timestamp"])

    data[f"{team}_{shirt_num}_x"][idx] = int(x)
    data[f"{team}_{shirt_num}_y"][idx] = int(y)
    data[f"{team}_{shirt_num}_speed"][idx] = float(speed)

    return data
        
def _add_ball_data_to_dict(ball_info:str, data:dict, idx:str) -> dict:
    """Function that adds the data of the ball to the data dict for one frame

    Args:
        ball_info (str): containing data from the ball
        data (dict): data dictionary to write results to
        idx (int): indicates position in data dictionary

    Returns:
        dict: contains all tracking data
    """

    x, y, z, speed, posession, status = ball_info.split(";")[0].split(",")[:6]
    data["ball_x"][idx] = int(x)
    data["ball_y"][idx] = int(y)
    data["ball_z"][idx] = int(z)
    data["ball_speed"][idx] = float(speed)
    data["ball_posession"][idx] = posession
    data["ball_status"][idx] = status

    return data

def _insert_missing_rows(df:pd.DataFrame) -> pd.DataFrame:
    """Functions that inserts missing rows based on gaps in timestamp

    Args:
        df (pd.DataFrame): containing tracking data

    Returns:
        pd.DataFrame: contains tracking data with inserted missing rows
    """
    assert "timestamp" in df.columns, "Calculations are based on timestamp column, which is not in the df"

    missing = np.where(df["timestamp"].diff() > 1)[0]
    for start_missing in missing:
        n_missing = int(df["timestamp"].diff()[start_missing] - 1)
        start_timestamp = df.loc[start_missing, "timestamp"] - n_missing
        to_add_data = {"timestamp": list(np.arange(start_timestamp, start_timestamp+n_missing))}
        to_add_df = pd.DataFrame(to_add_data)
        df = pd.concat((df, to_add_df)).sort_values(by="timestamp")

    df.reset_index(drop=True, inplace=True)
    
    return df

def _get_tracking_data(tracab_loc:str, verbose:bool) -> pd.DataFrame:
    """Function that reads tracking data from .dat file and stores it in a pd.DataFrame

    Args:
        tracab_loc (str): location of the tracking_data.dat file
        verbose (bool): whether to print info in terminal

    Returns:
        pd.DataFrame: contains tracking data
    """
    
    lines = _get_lines_from_dat(tracab_loc, verbose)
    size_lines = len(lines)

    data = {
        "timestamp": [np.nan] * size_lines,
        "ball_x": [np.nan] * size_lines,
        "ball_y": [np.nan] * size_lines,
        "ball_z": [np.nan] * size_lines,
        "ball_speed": [np.nan] * size_lines,
        "ball_status": [None] * size_lines,
        "ball_posession": [None] * size_lines,
    }

    for idx, line in enumerate(lines):
        timestamp, players_info, ball_info, _ = line.split(":")
        data["timestamp"][idx] = int(timestamp)

        players = players_info.split(";")[:-1]
        for player in players: 
            data = _add_player_data_to_dict(player, data, idx)

        data = _add_ball_data_to_dict(ball_info, data, idx)

    df=pd.DataFrame(data)

    df = _insert_missing_rows(df)
    
    return df

def _get_player_data(team) -> pd.DataFrame:
    
    player_dict = {"id":[], "full_name":[], "shirt_num":[], "start_frame":[], "end_frame":[]}
    for player in team.find("Players").find_all("Player"):
        player_dict["id"].append(int(player.find("PlayerId").text))
        player_dict["full_name"].append(player.find("FirstName").text + " " + player.find("LastName").text)
        player_dict["shirt_num"].append(int(player.find("JerseyNo").text))
        player_dict["start_frame"].append(int(player.find("StartFrameCount").text))
        player_dict["end_frame"].append(int(player.find("EndFrameCount").text))
    df = pd.DataFrame(player_dict)
    return df

def _get_meta_data(meta_data_loc:str) -> Metadata:
    """Function that reads metadata.xml file and stores it in Metadata class

    Args:
        meta_data_loc (str): location of the metadata.xml file

    Returns:
        Metadata: class that contains metadata
    """
    file = open(meta_data_loc, "r").read()
    file = file.replace("ï»¿", "")
    soup = BeautifulSoup(file, "xml")
    
    match_id = int(soup.find("match")["iId"])
    pitch_size_x = float(soup.find("match")["fPitchXSizeMeters"])
    pitch_size_y = float(soup.find("match")["fPitchYSizeMeters"])
    frame_rate = int(soup.find("match")["iFrameRateFps"])
    datetime_string = soup.find("match")["dtDate"]
    match_start_datetime = np.datetime64(datetime_string)

    frames_dict = {"period":[], "start_frame":[], "end_frame":[]}
    for i, period in enumerate(soup.find_all("period")):    
        frames_dict["period"].append(period["iId"])
        frames_dict["start_frame"].append(period["iStartFrame"])
        frames_dict["end_frame"].append(period["iEndFrame"])
    df_frames = pd.DataFrame(frames_dict)
        
    home_team = soup.find("HomeTeam")
    home_team_name = home_team.find("LongName").text
    home_team_id = int(home_team.find("TeamId").text)
    df_home_players = _get_player_data(home_team)

    away_team = soup.find("AwayTeam")
    away_team_name = away_team.find("LongName").text
    away_team_id = int(away_team.find("TeamId").text)
    df_away_players = _get_player_data(away_team)

    meta_data = Metadata(match_id=match_id,
                         pitch_dimensions=[pitch_size_x, pitch_size_y],
                         match_start_datetime=match_start_datetime,
                         periods_frames=df_frames,
                         frame_rate=frame_rate,
                         home_team_id=home_team_id,
                         home_team_name=home_team_name,
                         home_players=df_home_players,
                         home_score=None,
                         home_formation=None,
                         away_team_id=away_team_id,
                         away_team_name=away_team_name,
                         away_players=df_away_players,
                         away_score=None,
                         away_formation=None
                         )
    
    return meta_data


def load_tracking_data_tracab(tracab_loc, meta_data_loc, verbose=True):
    tracking_data = _get_tracking_data(tracab_loc, verbose)
    meta_data = _get_meta_data(meta_data_loc)

    return tracking_data, meta_data
