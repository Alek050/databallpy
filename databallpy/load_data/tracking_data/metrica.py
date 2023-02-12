import datetime as dt
import io
import os
from typing import Tuple, Union

import numpy as np
import pandas as pd
import requests
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


def load_metrica_tracking_data(
    tracking_data_loc: str, metadata_loc: str, verbose: bool = True
) -> Tuple[pd.DataFrame, Metadata]:
    """Function to load metrica tracking data.

    Args:
        tracking_data_loc (str): location of the tracking data .txt file
        metadata_loc (str): location of the metadata .xml file
        verbose (bool, optional): whether to print information about the progress in the terminall. Defaults to True.

    Raises:
        TypeError: if tracking_data_loc is not a string or io.StringIO

    Returns:
        Tuple[pd.DataFrame, Metadata]: tracking and metadata of the match
    """

    if isinstance(tracking_data_loc, str):
        assert os.path.exists(tracking_data_loc)
        assert os.path.exists(metadata_loc)
    elif isinstance(tracking_data_loc, io.StringIO):
        pass
    else:
        raise TypeError(
            f"tracking_data_loc must be either a str or a StringIO object, not a {type(tracking_data_loc)}"
        )

    metadata = _get_metadata(metadata_loc)
    td_channels = _get_td_channels(metadata_loc, metadata)
    tracking_data = _get_tracking_data(
        tracking_data_loc, td_channels, metadata.pitch_dimensions, verbose=verbose
    )
    tracking_data["matchtime_td"] = _get_matchtime(tracking_data["timestamp"], metadata)

    return tracking_data, metadata


def load_metrica_open_tracking_data() -> Tuple[pd.DataFrame, Metadata]:
    """Function to load open dataset of metrica

    Returns:
        Tuple[pd.DataFrame, Metadata]: tracking and metadata of the match
    """
    td_data_link = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_3/Sample_Game_3_tracking.txt"
    td_metadata_link = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_3/Sample_Game_3_metadata.xml"

    td_data = io.StringIO(requests.get(td_data_link).text)
    td_metadata = requests.get(td_metadata_link).text
    return load_metrica_tracking_data(
        tracking_data_loc=td_data, metadata_loc=td_metadata
    )


def _get_tracking_data(
    tracking_data_loc: Union[str, io.StringIO],
    channels: dict,
    pitch_dimensions: list,
    verbose: bool = True,
) -> pd.DataFrame:
    """Function to load the tracking data of metrica.

    Args:
        tracking_data_loc (Union[str, io.StringIO]): location of the tracking data .txt file
        channels (dict): dictionary with for all timestamps the order of which players are referred to in the raw tracking data
        pitch_dimensions (list): x and y dimensions of the pitch in meters
        verbose (bool, optional): whether to print information about the progress in the terminal. Defaults to True.

    Returns:
        pd.DataFrame: tracking data of the match in a pd dataframe
    """

    if isinstance(tracking_data_loc, str):
        file = open(tracking_data_loc)
        lines = file.readlines()
        file.close()
    else:
        lines = tracking_data_loc.readlines()

    channels = pd.DataFrame(channels)
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

        timestamp, players_info, ball_info = line.split(":")
        timestamp = int(timestamp)
        data["timestamp"][idx] = timestamp

        channel = channels.loc[
            (channels["start"] <= timestamp) & (channels["end"] >= timestamp),
            "ids",
        ].iloc[0]

        players = players_info.split(";")
        for i, player in enumerate(players):
            x, y = player.split(",")
            team = channel[i].split("_")[0]
            shirt_num = channel[i].split("_")[1]
            data = _add_player_tracking_data_to_dict(team, shirt_num, x, y, data, idx)

        x, y = ball_info.split(",")
        data = _add_ball_data_to_dict(x, y, np.nan, None, None, data, idx)

    df = pd.DataFrame(data)
    df["ball_status"] = np.where(pd.isnull(df["ball_x"]), "dead", "alive")

    for col in [x for x in df.columns if "_x" in x]:
        df[col] = df[col] * pitch_dimensions[0] - (pitch_dimensions[0] / 2)

    for col in [x for x in df.columns if "_y" in x]:
        df[col] = df[col] * pitch_dimensions[1] - (pitch_dimensions[1] / 2)

    df = _insert_missing_rows(df, "timestamp")

    return df


def _get_td_channels(metadata_loc: str, metadata: Metadata) -> dict:
    """Function to get the channels for every timeperiod with what players are referred to in the raw tracking data

    Args:
        metadata_loc (str): locatin of the metadata
        metadata (Metadata): the Metadata of the match

    Returns:
        dict: dict with for every timestamp what players are referred to in the raw tracking data
    """
    if os.path.exists(metadata_loc):
        file = open(metadata_loc, "r")
        lines = file.read()
        soup = BeautifulSoup(lines, "xml")
    else:
        soup = BeautifulSoup(metadata_loc.strip(), "xml")

    res = {"start": [], "end": [], "ids": []}
    for idx, data_format_specification in enumerate(
        soup.find_all("DataFormatSpecification")
    ):
        res["ids"].append([])
        res["start"].append(int(data_format_specification.attrs["startFrame"]))
        res["end"].append(int(data_format_specification.attrs["endFrame"]))
        for channel in data_format_specification.findChildren("PlayerChannelRef"):
            full_name = channel.attrs["playerChannelId"].split("_")[0]
            value = channel.attrs["playerChannelId"].split("_")[1]
            if "y" in value:
                continue
            home_mask = (
                metadata.home_players["full_name"]
                .str.lower()
                .str.replace(" ", "")
                .isin([full_name])
            )
            away_mask = (
                metadata.away_players["full_name"]
                .str.lower()
                .str.replace(" ", "")
                .isin([full_name])
            )
            if home_mask.any():
                team = "home"
                shirt_num = metadata.home_players.loc[home_mask, "shirt_num"].iloc[0]
            else:
                team = "away"
                shirt_num = metadata.away_players.loc[away_mask, "shirt_num"].iloc[0]
            res["ids"][idx].append(f"{team}_{shirt_num}")
        res["ids"][idx].append("ball")
    return res


def _get_metadata(metadata_loc: str) -> Metadata:
    """Function to get the metadata of the match

    Args:
        metadata_loc (str): Location of the metadata .xml file

    Returns:
        Metadata: all information of the match
    """
    if os.path.exists(metadata_loc):
        file = open(metadata_loc, "r")
        lines = file.read()
        soup = BeautifulSoup(lines, "xml")
    else:
        soup = BeautifulSoup(metadata_loc.strip(), "xml")

    match_id = _to_int(soup.find("Session").attrs["id"])
    pitch_size_x = _to_int(soup.find("FieldSize").find("Width").text)
    pitch_size_y = _to_int(soup.find("FieldSize").find("Height").text)
    frame_rate = _to_int(soup.find("FrameRate").text)
    datetime = pd.to_datetime(soup.find("Start").text)

    periods_dict = {
        "period": [],
        "start_frame": [],
        "end_frame": [],
        "start_time_td": [],
        "end_time_td": [],
    }

    periods_map = {
        "first_half_start": 1,
        "first_half_end": 1,
        "second_half_start": 2,
        "second_half_end": 2,
        "first_extra_half_start": 3,
        "first_extra_half_end": 3,
        "second_extra_half_start": 4,
        "second_extra_half_end": 4,
    }

    for period_soup in soup.find("ProviderGlobalParameters").find_all(
        "ProviderParameter"
    ):
        name = period_soup.find("Name").text
        period = periods_map[period_soup.find("Name").text]

        if "start" in name:
            periods_dict["period"].append(period)
            current_timestamp = _to_int(period_soup.find("Value").text)
            periods_dict["start_frame"].append(current_timestamp)
            first_timestamp = periods_dict["start_frame"][0]
            seconds = (current_timestamp - first_timestamp) / frame_rate
            if not pd.isnull(seconds):
                periods_dict["start_time_td"].append(
                    datetime + dt.timedelta(seconds=seconds)
                )
            else:
                periods_dict["start_time_td"].append(np.nan)
        elif "end" in name:
            current_timestamp = _to_int(period_soup.find("Value").text)
            periods_dict["end_frame"].append(current_timestamp)
            first_timestamp = periods_dict["start_frame"][0]
            seconds = (current_timestamp - first_timestamp) / frame_rate
            if not pd.isnull(seconds):
                periods_dict["end_time_td"].append(
                    datetime + dt.timedelta(seconds=seconds)
                )
            else:
                periods_dict["end_time_td"].append(np.nan)

    teams_info = {}
    for team in soup.find_all("Team"):
        team_info = {}
        team_info["team_name"] = team.find("Name").text
        team_info["team_id"] = team.attrs["id"]
        team_info["formation"] = ""
        score = soup.find("Score")
        if score.attrs["idLocalTeam"] == team_info["team_id"]:
            team_info["side"] = "home"
            team_info["score"] = _to_int(score.find("LocalTeamScore").text)
        else:
            team_info["side"] = "away"
            team_info["score"] = _to_int(score.find("VisitingTeamScore").text)

        teams_info[team_info["side"]] = team_info

    players = soup.find_all("Player")
    home_players = {
        "id": [],
        "full_name": [],
        "formation_place": [],
        "position": [],
        "starter": [],
        "shirt_num": [],
    }
    away_players = {
        "id": [],
        "full_name": [],
        "formation_place": [],
        "position": [],
        "starter": [],
        "shirt_num": [],
    }
    for player in players:
        if player.attrs["teamId"] == teams_info["home"]["team_id"]:
            res_dict = home_players
        else:
            res_dict = away_players

        res_dict["id"].append(_to_int(player.attrs["id"][1:]))
        res_dict["full_name"].append(player.find("Name").text)
        res_dict["shirt_num"].append(_to_int(player.find("ShirtNumber").text))
        res_dict["starter"].append(np.nan)
        for param in player.findChildren("ProviderParameter"):
            if param.find("Name").text == "position_type":
                res_dict["position"].append(param.find("Value").text)
            elif param.find("Name").text == "position_index":
                res_dict["formation_place"].append(_to_int(param.find("Value").text))

    metadata = Metadata(
        match_id=match_id,
        pitch_dimensions=[pitch_size_x, pitch_size_y],
        periods_frames=pd.DataFrame(periods_dict),
        frame_rate=frame_rate,
        home_team_id=teams_info["home"]["team_id"],
        home_team_name=teams_info["home"]["team_name"],
        home_players=pd.DataFrame(home_players),
        home_score=teams_info["home"]["score"],
        home_formation=teams_info["home"]["formation"],
        away_team_id=teams_info["away"]["team_id"],
        away_team_name=teams_info["away"]["team_name"],
        away_players=pd.DataFrame(away_players),
        away_score=teams_info["away"]["score"],
        away_formation=teams_info["away"]["formation"],
    )
    return metadata


def _to_int(value) -> Union[float, int]:
    """Function to make a integer of the value if possible, else np.nan

    Args:
        value (): a variable value

    Returns:
        Union[float, int]: integer if value can be changed to integer, else np.nan
    """
    try:
        return int(value)
    except ValueError:
        return np.nan
