import os
from typing import Tuple

import bs4
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from databallpy.load_data.metadata import Metadata
from databallpy.load_data.tracking_data._add_ball_data_to_dict import (
    _add_ball_data_to_dict,
)
from databallpy.load_data.tracking_data._add_periods_to_tracking_data import (
    _add_periods_to_tracking_data,
)
from databallpy.load_data.tracking_data._add_player_tracking_data_to_dict import (
    _add_player_tracking_data_to_dict,
)
from databallpy.load_data.tracking_data._get_matchtime import _get_matchtime
from databallpy.load_data.tracking_data._insert_missing_rows import _insert_missing_rows
from databallpy.load_data.tracking_data._normalize_playing_direction_tracking import (
    _normalize_playing_direction_tracking,
)
from databallpy.utils.tz_modification import utc_to_local_datetime
from databallpy.utils.utils import MISSING_INT, _to_float, _to_int


def load_inmotio_tracking_data(
    tracking_data_loc: str, metadata_loc: str, verbose: bool = True
) -> Tuple[pd.DataFrame, Metadata]:
    """Function to load inmotio tracking data.

    Args:
        tracking_data_loc (str): location of the tracking data .txt file
        metadata_loc (str): location of the metadata .xml file
        verbose (bool, optional): whether to print information about the progress
        in the terminall. Defaults to True.

    Raises:
        TypeError: if tracking_data_loc is not a string

    Returns:
        Tuple[pd.DataFrame, Metadata]: tracking and metadata of the match
    """
    if isinstance(tracking_data_loc, str):
        assert os.path.exists(tracking_data_loc)
        assert os.path.exists(metadata_loc)
    else:
        raise TypeError(
            f"tracking_data_loc must be  a str, not a {type(tracking_data_loc)}"
        )

    metadata = _get_metadata(metadata_loc)
    td_channels = _get_td_channels(metadata_loc, metadata)
    tracking_data = _get_tracking_data(
        tracking_data_loc, td_channels, metadata.pitch_dimensions, verbose
    )
    first_frame = metadata.periods_frames[metadata.periods_frames["start_frame"] > 0][
        "start_frame"
    ].min()
    last_frame = metadata.periods_frames["end_frame"].max()
    tracking_data = tracking_data[
        (tracking_data["frame"] >= first_frame) & (tracking_data["frame"] <= last_frame)
    ].reset_index(drop=True)
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


def _get_tracking_data(
    tracking_data_loc: str,
    td_channels: list,
    pitch_dimensions: list,
    verbose: bool = True,
) -> pd.DataFrame:
    """Function to load in inmotio format tracking_data

    Args:
        tracking_data_loc (str): location of the tracking .txt file
        td_channels (list): the order of which players are referred
        to in the raw tracking data
        pitch_dimensions (list): x and y dimensions of the pitch in meters
        verbose (bool, optional): whether to print information about the progress in the
        terminal. Defaults to True.

    Returns:
        pd.DataFrame: tracking data of the match in a pd dataframe
    """
    if verbose:
        print(f"Reading in {tracking_data_loc}", end="")
    file = open(tracking_data_loc, "r")
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
        players_info = players_home + ";" + players_away
        players = players_info.split(";")
        for i, player in enumerate(players):
            y, x = player.split(",")
            team = td_channels[i].split("_")[0]
            shirt_num = td_channels[i].split("_")[1]
            data = _add_player_tracking_data_to_dict(
                team, shirt_num, _to_float(x), _to_float(y), data, idx
            )

        y, x, z, _, ball_status = ball_info.replace("\n", "").split(",")
        data = _add_ball_data_to_dict(
            _to_float(x), _to_float(y), _to_float(z), None, ball_status, data, idx
        )

    df = pd.DataFrame(data)
    df["ball_status"] = ["alive" if x == "1" else "dead" for x in df["ball_status"]]
    for col in [x for x in df.columns if "_x" in x]:
        df[col] = df[col] - (pitch_dimensions[0] / 2)
        df[col] *= -1

    for col in [x for x in df.columns if "_y" in x]:
        df[col] = df[col] - (pitch_dimensions[1] / 2)

    df = _insert_missing_rows(df, "frame")

    return df


def _get_td_channels(metadata_loc: str, metadata: Metadata) -> list:
    """Function to get the channels the order of which players
    are referred to in the raw tracking data

    Args:
        metadata_loc (str): location of the metadata
        metadata (Metadata): the Metadata of the match

    Returns:
        list: List with the order of which players are referred to
        in the raw tracking data
    """
    file = open(metadata_loc, "r", encoding="UTF-8")
    lines = file.read()
    soup = BeautifulSoup(lines, "xml")
    file.close()

    res = []
    for channel in soup.find_all("PlayerChannel"):
        player_id = int(channel.attrs["id"].split("_")[0][2:])
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


def _get_metadata(metadata_loc: str) -> Metadata:
    """Function to get the metadata of the match

    Args:
        metadata_loc (str): Location of the metadata .xml file

    Returns:
        Metadata: all information of the match
    """

    file = open(metadata_loc, "r", encoding="UTF-8")
    lines = file.read()
    soup = BeautifulSoup(lines, "xml")
    file.close()

    periods_dict = {
        "period": [1, 2, 3, 4, 5],
        "start_frame": [MISSING_INT] * 5,
        "end_frame": [MISSING_INT] * 5,
        "start_datetime_td": [pd.to_datetime("NaT")] * 5,
        "end_datetime_td": [pd.to_datetime("NaT")] * 5,
    }
    periods = soup.find_all("Session")

    i = 0
    for period in periods:
        if period.SessionType.text == "Period":
            values = [int(x.text) for x in period.find_all("Value")]
            periods_dict["start_frame"][i] = _to_int(values[0])
            periods_dict["end_frame"][i] = _to_int(values[1])
            periods_dict["start_datetime_td"][i] = pd.to_datetime(
                period.find("Start").text, utc=True
            )
            periods_dict["end_datetime_td"][i] = pd.to_datetime(
                period.find("End").text, utc=True
            )
            i += 1
    periods_frames = pd.DataFrame(periods_dict)

    competition = soup.find("Competition").text.split(",")[0]

    # set to the right timezone
    periods_frames["start_datetime_td"] = utc_to_local_datetime(
        periods_frames["start_datetime_td"], competition
    )
    periods_frames["end_datetime_td"] = utc_to_local_datetime(
        periods_frames["end_datetime_td"], competition
    )

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
        match_id=int(soup.Session.attrs["id"]),
        pitch_dimensions=[
            float(soup.MatchParameters.FieldSize.Length.text),
            float(soup.MatchParameters.FieldSize.Width.text),
        ],
        periods_frames=periods_frames,
        frame_rate=int(soup.FrameRate.text),
        home_team_id=home_team_id,
        home_team_name=home_team_name,
        home_players=home_players,
        home_score=home_score,
        home_formation="",
        away_team_id=away_team_id,
        away_team_name=away_team_name,
        away_players=away_players,
        away_score=away_score,
        away_formation="",
        country="",
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
        "end_frame": [],
    }
    for player in team:
        player_dict["id"].append(int(player["id"][2:]))
        player_dict["full_name"].append(player.Name.text)
        player_dict["shirt_num"].append(int(player.ShirtNumber.text))
        values = [x.text for x in player.find_all("Value")]
        player_dict["player_type"].append(values[0])
        player_dict["start_frame"].append(int(values[1]))
        player_dict["end_frame"].append(int(values[2]))
    df = pd.DataFrame(player_dict)

    return df
