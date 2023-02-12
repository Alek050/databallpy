from typing import Tuple

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from databallpy.load_data.metadata import Metadata
from databallpy.load_data.tracking_data._add_ball_data_to_dict import (
    _add_ball_data_to_dict,
)
from databallpy.load_data.tracking_data._add_player_tracking_data_to_dict import (
    _add_player_tracking_data_to_dict,
)
from databallpy.load_data.tracking_data._get_matchtime import _get_matchtime
from databallpy.load_data.tracking_data._insert_missing_rows import _insert_missing_rows


def load_tracab_tracking_data(
    tracab_loc: str, metadata_loc: str, verbose: bool = True
) -> Tuple[pd.DataFrame, Metadata]:
    """Function to load tracking data and metadata from the tracab format

    Args:
        tracab_loc (str): location of the tracking_data.dat file
        metadata_loc (str): location of the meta_data.xml file
        verbose (bool): whether to print on progress of loading in the terminal,
        defaults to True

    Returns:
        Tuple[pd.DataFrame, Metadata], the tracking data and metadata class
    """

    tracking_data = _get_tracking_data(tracab_loc, verbose)
    metadata = _get_metadata(metadata_loc)

    tracking_data["matchtime_td"] = _get_matchtime(tracking_data["timestamp"], metadata)

    return tracking_data, metadata


def _get_tracking_data(tracab_loc: str, verbose: bool) -> pd.DataFrame:
    """Function that reads tracking data from .dat file and stores it in a pd.DataFrame

    Args:
        tracab_loc (str): location of the tracking_data.dat file
        verbose (bool): whether to print info in terminal

    Returns:
        pd.DataFrame: contains tracking data
    """

    if verbose:
        print(f"Reading in {tracab_loc}", end="")

    file = open(tracab_loc, "r")
    lines = file.readlines()
    if verbose:
        print(" - Completed")

    file.close()
    size_lines = len(lines)

    data = {
        "timestamp": [np.nan] * size_lines,
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

        timestamp, players_info, ball_info, _ = line.split(":")
        data["timestamp"][idx] = int(timestamp)

        players = players_info.split(";")[:-1]
        for player in players:
            team_id, _, shirt_num, x, y, _ = player.split(",")
            team_ids = {0: "away", 1: "home"}
            team = team_ids.get(int(team_id))
            if team is None:  # player is unknown or referee
                continue
            data = _add_player_tracking_data_to_dict(team, shirt_num, x, y, data, idx)

        ball_x, ball_y, ball_z, _, posession, status = ball_info.split(";")[0].split(
            ","
        )[:6]
        home_away_map = {"H": "home", "A": "away"}
        posession = home_away_map[posession]
        data = _add_ball_data_to_dict(
            ball_x, ball_y, ball_z, posession, status.lower(), data, idx
        )

    df = pd.DataFrame(data)

    for col in df.columns:
        if "_x" in col or "_y" in col or "_z" in col:
            df[col] = np.round(df[col] / 100, 3)  # change cm to m

    df = _insert_missing_rows(df, "timestamp")

    return df


def _get_metadata(metadata_loc: str) -> Metadata:
    """Function that reads metadata.xml file and stores it in Metadata class

    Args:
        meta_data_loc (str): location of the metadata.xml file

    Returns:
        Metadata: class that contains metadata
    """

    file = open(metadata_loc, "r")
    lines = file.read()
    lines = lines.replace("ï»¿", "")
    soup = BeautifulSoup(lines, "xml")

    match_id = int(soup.find("match")["iId"])
    pitch_size_x = float(soup.find("match")["fPitchXSizeMeters"])
    pitch_size_y = float(soup.find("match")["fPitchYSizeMeters"])
    frame_rate = int(soup.find("match")["iFrameRateFps"])
    datetime_string = soup.find("match")["dtDate"]
    date = np.datetime64(datetime_string[:10])

    frames_dict = {
        "period": [],
        "start_frame": [],
        "end_frame": [],
        "start_time_td": [],
        "end_time_td": [],
    }
    for i, period in enumerate(soup.find_all("period")):
        frames_dict["period"].append(int(period["iId"]))
        start_frame = int(period["iStartFrame"])
        end_frame = int(period["iEndFrame"])

        frames_dict["start_frame"].append(start_frame)
        frames_dict["end_frame"].append(end_frame)
        if start_frame != 0:
            frames_dict["start_time_td"].append(
                date + np.timedelta64(int(start_frame / frame_rate), "s")
            )
            frames_dict["end_time_td"].append(
                date + np.timedelta64(int(end_frame / frame_rate), "s")
            )
        else:
            frames_dict["start_time_td"].append(np.nan)
            frames_dict["end_time_td"].append(np.nan)
    df_frames = pd.DataFrame(frames_dict)

    home_team = soup.find("HomeTeam")
    home_team_name = home_team.find("LongName").text
    home_team_id = int(home_team.find("TeamId").text)

    home_players_info = []
    for player in home_team.find_all("Player"):
        player_dict = {}
        for element in player.findChildren():
            player_dict[element.name] = element.text
        home_players_info.append(player_dict)
    df_home_players = _get_players_metadata(home_players_info)

    away_team = soup.find("AwayTeam")
    away_team_name = away_team.find("LongName").text
    away_team_id = int(away_team.find("TeamId").text)

    away_players_info = []
    for player in away_team.find_all("Player"):
        player_dict = {}
        for element in player.findChildren():
            player_dict[element.name] = element.text
        away_players_info.append(player_dict)
    df_away_players = _get_players_metadata(away_players_info)

    metadata = Metadata(
        match_id=match_id,
        pitch_dimensions=[pitch_size_x, pitch_size_y],
        periods_frames=df_frames,
        frame_rate=frame_rate,
        home_team_id=home_team_id,
        home_team_name=home_team_name,
        home_players=df_home_players,
        home_score=np.nan,
        home_formation=None,
        away_team_id=away_team_id,
        away_team_name=away_team_name,
        away_players=df_away_players,
        away_score=np.nan,
        away_formation=None,
    )

    file.close()

    return metadata


def _get_players_metadata(players_info: list) -> pd.DataFrame:
    """Function that creates a df containing info on all players for a team

    Args:
        team (list): contains an information dictionary for each player

    Returns:
        pd.DataFrame: contains all player information for a team
    """

    player_dict = {
        "id": [],
        "full_name": [],
        "shirt_num": [],
        "start_frame": [],
        "end_frame": [],
    }
    for player in players_info:
        player_dict["id"].append(int(player["PlayerId"]))
        full_name = player["FirstName"] + " " + player["LastName"]
        if player["FirstName"] == "":
            full_name = full_name.lstrip()
        player_dict["full_name"].append(full_name)
        player_dict["shirt_num"].append(int(player["JerseyNo"]))
        player_dict["start_frame"].append(int(player["StartFrameCount"]))
        player_dict["end_frame"].append(int(player["EndFrameCount"]))
    df = pd.DataFrame(player_dict)

    return df
