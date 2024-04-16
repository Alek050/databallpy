import datetime as dt
import os

import chardet
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from databallpy.data_parsers import Metadata
from databallpy.data_parsers.tracking_data_parsers.utils import (
    _add_ball_data_to_dict,
    _add_datetime,
    _add_periods_to_tracking_data,
    _add_player_tracking_data_to_dict,
    _adjust_start_end_frames,
    _get_matchtime,
    _insert_missing_rows,
    _normalize_playing_direction_tracking,
)
from databallpy.utils.constants import MISSING_INT
from databallpy.utils.logging import create_logger
from databallpy.utils.tz_modification import localize_datetime

LOGGER = create_logger(__name__)


def load_tracab_tracking_data(
    tracab_loc: str, metadata_loc: str, verbose: bool = True
) -> tuple[pd.DataFrame, Metadata]:
    """Function to load tracking data and metadata from the tracab format

    Args:
        tracab_loc (str): location of the tracking_data.dat file
        metadata_loc (str): location of the meta_data.xml file
        verbose (bool): whether to print on progress of loading in the terminal,
        defaults to True

    Returns:
        Tuple[pd.DataFrame, Metadata], the tracking data and metadata class
    """
    LOGGER.info("Trying to load Tracab tracking data")

    if not os.path.exists(tracab_loc):
        message = f"Could not find {tracab_loc}."
        LOGGER.error(message)
        raise FileNotFoundError(message)
    if not os.path.exists(metadata_loc):
        message = f"Could not find {metadata_loc}."
        LOGGER.error(message)
        raise FileNotFoundError(message)

    tracking_data = _get_tracking_data(tracab_loc, verbose)
    LOGGER.info("Successfully loaded the Tracab tracking data.")
    metadata = _get_metadata(metadata_loc)
    LOGGER.info("Successfully loaded the Tracab metdata.")

    tracking_data["period_id"] = _add_periods_to_tracking_data(
        tracking_data["frame"], metadata.periods_frames
    )

    tracking_data["datetime"] = _add_datetime(
        tracking_data["frame"],
        metadata.frame_rate,
        metadata.periods_frames["start_datetime_td"].iloc[0],
    )
    tracking_data, metadata = _adjust_start_end_frames(tracking_data, metadata)

    tracking_data["matchtime_td"] = _get_matchtime(
        tracking_data["frame"], tracking_data["period_id"], metadata
    )

    tracking_data, changed_periods = _normalize_playing_direction_tracking(
        tracking_data, metadata.periods_frames
    )
    metadata.periods_changed_playing_direction = changed_periods

    LOGGER.info("Successfully post-processed the Tracab data.")
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

    with open(tracab_loc, "r") as file:
        lines = file.readlines()
    if verbose:
        print(" - Completed")

    size_lines = len(lines)

    data = {
        "frame": [np.nan] * size_lines,
        "ball_x": [np.nan] * size_lines,
        "ball_y": [np.nan] * size_lines,
        "ball_z": [np.nan] * size_lines,
        "ball_status": [None] * size_lines,
        "ball_possession": [None] * size_lines,
    }
    team_ids = {0: "away", 1: "home"}
    home_away_map = {"H": "home", "A": "away"}

    if verbose:
        lines = tqdm(
            lines, desc="Writing lines to dataframe", unit=" lines", leave=False
        )

    for idx, (frame, players_info, ball_info, _) in enumerate(
        (line.split(":") for line in lines)
    ):
        data["frame"][idx] = int(frame)

        players = players_info.split(";")[:-1]
        for team_id, _, shirt_num, x, y, _ in (player.split(",") for player in players):
            team = team_ids.get(int(team_id))
            if team is None:  # player is unknown or referee
                continue
            data = _add_player_tracking_data_to_dict(team, shirt_num, x, y, data, idx)

        ball_x, ball_y, ball_z, _, possession, status = ball_info.split(";")[0].split(
            ","
        )[:6]

        possession = home_away_map[possession]
        data = _add_ball_data_to_dict(
            ball_x, ball_y, ball_z, possession, status.lower(), data, idx
        )

    df = pd.DataFrame(data)

    mask = df.columns.str.contains("_x|_y|_z")
    df.loc[:, mask] = np.round(df.loc[:, mask] / 100, 3)  # change cm to m
    df = _insert_missing_rows(df, "frame")
    return df


def _get_metadata(metadata_loc: str) -> Metadata:
    """Function that reads metadata.xml file and stores it in Metadata class

    Args:
        meta_data_loc (str): location of the metadata.xml file

    Returns:
        Metadata: class that contains metadata
    """

    with open(metadata_loc, "rb") as file:
        encoding = chardet.detect(file.read())["encoding"]
    with open(metadata_loc, "r", encoding=encoding) as file:
        lines = file.read()

    lines = lines.replace("ï»¿", "")
    soup = BeautifulSoup(lines, "xml")

    match_id = int(soup.find("match")["iId"])
    pitch_size_x = float(soup.find("match")["fPitchXSizeMeters"])
    pitch_size_y = float(soup.find("match")["fPitchYSizeMeters"])
    frame_rate = int(soup.find("match")["iFrameRateFps"])
    datetime_string = soup.find("match")["dtDate"]
    date = pd.to_datetime(datetime_string[:10])

    frames_dict = {
        "period_id": [],
        "start_frame": [],
        "end_frame": [],
        "start_datetime_td": [],
        "end_datetime_td": [],
    }
    for _, period in enumerate(soup.find_all("period")):
        frames_dict["period_id"].append(int(period["iId"]))
        start_frame = int(period["iStartFrame"])
        end_frame = int(period["iEndFrame"])

        if start_frame != 0:
            frames_dict["start_frame"].append(start_frame)
            frames_dict["end_frame"].append(end_frame)
            start_frame_corrected = start_frame % (frame_rate * 60 * 60 * 24)
            end_frame_corrected = end_frame % (frame_rate * 60 * 60 * 24)
            frames_dict["start_datetime_td"].append(
                date
                + dt.timedelta(
                    milliseconds=int((start_frame_corrected / frame_rate) * 1000)
                )
            )
            frames_dict["end_datetime_td"].append(
                date
                + dt.timedelta(
                    milliseconds=int((end_frame_corrected / frame_rate) * 1000)
                )
            )
        else:
            frames_dict["start_frame"].append(MISSING_INT)
            frames_dict["end_frame"].append(MISSING_INT)
            frames_dict["start_datetime_td"].append(pd.to_datetime("NaT"))
            frames_dict["end_datetime_td"].append(pd.to_datetime("NaT"))
    df_frames = pd.DataFrame(frames_dict)

    # set to right timezone, tracab has no location/competition info
    # in metadata, so we have to guess
    df_frames["start_datetime_td"] = localize_datetime(
        df_frames["start_datetime_td"], "Netherlands"
    )
    df_frames["end_datetime_td"] = localize_datetime(
        df_frames["end_datetime_td"], "Netherlands"
    )
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
        home_score=MISSING_INT,
        home_formation="",
        away_team_id=away_team_id,
        away_team_name=away_team_name,
        away_players=df_away_players,
        away_score=MISSING_INT,
        away_formation="",
        country="",
    )

    return metadata


def _get_players_metadata(players_info: list[dict[str, int | float]]) -> pd.DataFrame:
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
