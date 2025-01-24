import datetime as dt
import os

import chardet
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from databallpy.data_parsers import Metadata
from databallpy.utils.constants import DATABALLPY_POSITIONS, MISSING_INT
from databallpy.utils.utils import _to_float, _to_int


def _get_td_channels(metadata_loc: str, metadata: Metadata) -> pd.DataFrame:
    """Function to get the channels for every timeperiod with what players are
    referred to in the raw tracking data

    Args:
        metadata_loc (str): locatin of the metadata
        metadata (Metadata): the Metadata of the game

    Returns:
        pd.DataFrame: df with for every frame what players are referred to in
        the raw tracking data
    """
    if os.path.exists(metadata_loc):
        with open(metadata_loc, "rb") as file:
            encoding = chardet.detect(file.read())["encoding"]
        with open(metadata_loc, "r", encoding=encoding) as file:
            lines = file.read()
        soup = BeautifulSoup(lines, "xml")
    else:
        soup = BeautifulSoup(metadata_loc.strip(), "xml")

    channel_id_to_player_id_dict = {}
    for player in soup.find_all("PlayerChannel"):
        channel_id = player.attrs["id"].split("_")[0]
        value = player.attrs["id"].split("_")[1]
        if "y" in value:
            continue
        channel_id_to_player_id_dict[channel_id] = int(player.attrs["playerId"][1:])

    res = {"start": [], "end": [], "ids": []}
    for idx, data_format_specification in enumerate(
        soup.find_all("DataFormatSpecification")
    ):
        res["ids"].append([])
        res["start"].append(int(data_format_specification.attrs["startFrame"]))
        res["end"].append(int(data_format_specification.attrs["endFrame"]))
        for channel in data_format_specification.findChildren("PlayerChannelRef"):
            channel_id = channel.attrs["playerChannelId"].split("_")[0]
            value = channel.attrs["playerChannelId"].split("_")[1]
            if "y" in value:
                continue
            player_id = channel_id_to_player_id_dict[channel_id]
            home_mask = metadata.home_players["id"] == player_id
            away_mask = metadata.away_players["id"] == player_id
            if home_mask.any():
                team = "home"
                shirt_num = metadata.home_players.loc[home_mask, "shirt_num"].values[0]
            else:
                team = "away"
                shirt_num = metadata.away_players.loc[away_mask, "shirt_num"].values[0]
            res["ids"][idx].append(f"{team}_{shirt_num}")
    return pd.DataFrame(res)


def _get_metadata(
    metadata_loc: str, is_tracking_data: bool = True, is_event_data: bool = False
) -> Metadata:
    """Function to get the metadata of the game

    Args:
        metadata_loc (str): Location of the metadata .xml file
        is_tracking_data (bool, optional): Whether the metadata is from the tracking
        data. Defaults to True.
        is_event_data (bool, optional): Whether the metadata is from the event data.
        Defaults to False

    Returns:
        Metadata: all information of the game
    """
    if os.path.exists(metadata_loc):
        with open(metadata_loc, "rb") as file:
            encoding = chardet.detect(file.read())["encoding"]
        with open(metadata_loc, "r", encoding=encoding) as file:
            lines = file.read()
        soup = BeautifulSoup(lines, "xml")
    else:
        soup = BeautifulSoup(metadata_loc.strip(), "xml")

    game_id = _to_int(soup.find("Session").attrs["id"])
    pitch_size_x = _to_float(soup.find("FieldSize").find("Width").text)
    pitch_size_y = _to_float(soup.find("FieldSize").find("Height").text)
    frame_rate = _to_int(soup.find("FrameRate").text)

    # no idea about time zone, so just assume utc
    datetime = pd.to_datetime(soup.find("Start").text, utc=True)

    periods_dict = {
        "period_id": [],
        "start_frame": [],
        "end_frame": [],
    }
    if is_tracking_data:
        periods_dict["start_datetime_td"] = []
        periods_dict["end_datetime_td"] = []
    if is_event_data:
        periods_dict["start_datetime_ed"] = []
        periods_dict["end_datetime_ed"] = []

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
        period = periods_map[name]

        current_frame = _to_int(period_soup.find("Value").text)

        if "start" in name:
            periods_dict["period_id"].append(period)
            periods_dict["start_frame"].append(current_frame)
            first_frame = periods_dict["start_frame"][0]
            if is_tracking_data:
                seconds = (current_frame - first_frame) / frame_rate
                periods_dict["start_datetime_td"].append(
                    datetime + dt.timedelta(milliseconds=seconds * 1000)
                ) if not current_frame == MISSING_INT else periods_dict[
                    "start_datetime_td"
                ].append(pd.to_datetime("NaT"))
            if is_event_data:
                seconds = (current_frame - first_frame) / frame_rate
                periods_dict["start_datetime_ed"].append(
                    datetime + dt.timedelta(milliseconds=seconds * 1000)
                ) if not current_frame == MISSING_INT else periods_dict[
                    "start_datetime_ed"
                ].append(pd.to_datetime("NaT"))

        elif "end" in name:
            periods_dict["end_frame"].append(current_frame)
            first_frame = periods_dict["start_frame"][0]
            seconds = (current_frame - first_frame) / frame_rate
            if is_tracking_data:
                periods_dict["end_datetime_td"].append(
                    datetime + dt.timedelta(milliseconds=seconds * 1000)
                ) if not current_frame == MISSING_INT else periods_dict[
                    "end_datetime_td"
                ].append(pd.to_datetime("NaT"))
            if is_event_data:
                periods_dict["end_datetime_ed"].append(
                    datetime + dt.timedelta(milliseconds=seconds * 1000)
                ) if not current_frame == MISSING_INT else periods_dict[
                    "end_datetime_ed"
                ].append(pd.to_datetime("NaT"))

    # add fifth period
    periods_dict["period_id"].append(5)
    periods_dict["start_frame"].append(MISSING_INT)
    periods_dict["end_frame"].append(MISSING_INT)
    if is_tracking_data:
        periods_dict["start_datetime_td"].append(pd.to_datetime("NaT"))
        periods_dict["end_datetime_td"].append(pd.to_datetime("NaT"))
    if is_event_data:
        periods_dict["start_datetime_ed"].append(pd.to_datetime("NaT"))
        periods_dict["end_datetime_ed"].append(pd.to_datetime("NaT"))

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
    team_dicts = [
        {
            "team_id": teams_info["home"]["team_id"],
            "player_dict": {
                "id": [],
                "full_name": [],
                "formation_place": [],
                "position": [],
                "starter": [],
                "shirt_num": [],
                "start_frame": [],
                "end_frame": [],
            },
        },
        {
            "team_id": teams_info["away"]["team_id"],
            "player_dict": {
                "id": [],
                "full_name": [],
                "formation_place": [],
                "position": [],
                "starter": [],
                "shirt_num": [],
                "start_frame": [],
                "end_frame": [],
            },
        },
    ]

    for player in players:
        for team in team_dicts:
            if player.attrs["teamId"] == team["team_id"]:
                res_dict = team["player_dict"]

                res_dict["id"].append(_to_int(player.attrs["id"][1:]))
                res_dict["full_name"].append(player.find("Name").text)
                res_dict["shirt_num"].append(_to_int(player.find("ShirtNumber").text))
                res_dict["starter"].append(np.nan)
                res_dict["start_frame"].append(MISSING_INT)
                res_dict["end_frame"].append(MISSING_INT)
                for param in player.findChildren("ProviderParameter"):
                    if param.find("Name").text == "position_type":
                        res_dict["position"].append(
                            _metrica_position_to_databallpy_position(
                                param.find("Value").text.lower()
                            )
                        )
                    elif param.find("Name").text == "position_index":
                        res_dict["formation_place"].append(
                            _to_int(param.find("Value").text)
                        )

    home_players = team_dicts[0]["player_dict"]
    away_players = team_dicts[1]["player_dict"]

    metadata = Metadata(
        game_id=_to_int(game_id),
        pitch_dimensions=[float(pitch_size_x), float(pitch_size_y)],
        periods_frames=pd.DataFrame(periods_dict),
        frame_rate=_to_int(frame_rate),
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
        country="",
        periods_changed_playing_direction=None,
    )
    return metadata


def _metrica_position_to_databallpy_position(position: str) -> str:
    for pos in [pos for pos in DATABALLPY_POSITIONS if pos != "defender"]:
        if pos in position:
            return pos
    if "back" in position:
        return "defender"


def _update_metadata(td_channels: pd.DataFrame, metadata: Metadata) -> Metadata:
    """Function to add the starters and formation based on the metadata and tracking
    data channels to the metadata

    Args:
        td_channels (pd.DataFrame): tracking data channels, what indexes in the
        tracking data belong to which players, all players in the first line are
        starters
        metadata (Metadata): metadata of the game

    Returns:
        Metadata: updated metadata of the game
    """
    home_formation = [0, 0, 0, 0]
    away_formation = [0, 0, 0, 0]
    metadata.home_players["starter"] = False
    metadata.away_players["starter"] = False
    starting = td_channels.iloc[0]["ids"]

    for starter in starting:
        team, shirt_number = starter.split("_")
        shirt_number = _to_int(shirt_number)
        if team == "home":
            metadata.home_players.loc[
                metadata.home_players["shirt_num"] == shirt_number, "starter"
            ] = True
            formation_place = metadata.home_players.loc[
                metadata.home_players["shirt_num"] == shirt_number, "formation_place"
            ].iloc[0]
            home_formation[formation_place] += 1
        else:
            metadata.away_players.loc[
                metadata.away_players["shirt_num"] == shirt_number, "starter"
            ] = True
            formation_place = metadata.away_players.loc[
                metadata.away_players["shirt_num"] == shirt_number, "formation_place"
            ].iloc[0]
            away_formation[formation_place] += 1

    metadata.home_formation = "".join(str(i) for i in home_formation)
    metadata.away_formation = "".join(str(i) for i in away_formation)

    return metadata
