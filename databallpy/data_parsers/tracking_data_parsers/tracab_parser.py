import datetime as dt
import os

import chardet
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from lxml import etree
from tqdm import tqdm

from databallpy.data_parsers import Metadata
from databallpy.data_parsers.sportec_metadata_parser import (
    _get_sportec_metadata,
    _get_sportec_open_data_url,
)
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
        Tuple[pd.DataFrame, Metadata]: the tracking data and metadata class
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

    metadata = _get_metadata(metadata_loc)
    LOGGER.info("Successfully loaded the Tracab metdata.")
    if tracab_loc.endswith(".dat") or tracab_loc.endswith(".txt"):
        tracking_data = _get_tracking_data_txt(tracab_loc, verbose)
        tracking_data["datetime"] = _add_datetime(
            tracking_data["frame"],
            metadata.frame_rate,
            metadata.periods_frames["start_datetime_td"].iloc[0],
        )
    elif tracab_loc.endswith(".xml"):
        tracking_data, periods_frames, frame_rate = _get_tracking_data_xml(
            tracab_loc, metadata.home_players, metadata.away_players, verbose
        )
        metadata.periods_frames = periods_frames
        metadata.frame_rate = int(frame_rate)
        tracking_data = _insert_missing_rows(
            tracking_data.reset_index(drop=True), "frame"
        )
    else:
        message = "Tracab tracking data should be either .txt, .dat, or .xml format."
        LOGGER.error(message)
        raise ValueError(message)

    tracking_data.insert(
        len(tracking_data.columns) - 1,
        "period_id",
        _add_periods_to_tracking_data(tracking_data["frame"], metadata.periods_frames),
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


def load_sportec_open_tracking_data(
    match_id: str, verbose: bool
) -> tuple[pd.DataFrame, Metadata]:
    """Load the tracking data from the sportec open data platform

    Args:
        match_id (str): The id of the match
        verbose (bool): Whether to print info about the loading of the data.

    Returns:
        tuple[pd.DataFrame, Metadata]: the tracking data and metadata class
    """
    metadata_url = _get_sportec_open_data_url(match_id, "metadata")
    save_path = os.path.join(os.getcwd(), "datasets", "IDSSE", match_id)
    os.makedirs(save_path, exist_ok=True)
    if not os.path.exists(os.path.join(save_path, "metadata.xml")):
        metadata = requests.get(metadata_url)
        with open(os.path.join(save_path, "metadata.xml"), "wb") as f:
            f.write(metadata.content)

    if not os.path.exists(os.path.join(save_path, "tracking_data.xml")):
        if verbose:
            print("Downloading open tracking data...", end="\r")
        session = requests.Session()
        response = session.get(
            _get_sportec_open_data_url(match_id, "tracking_data"), stream=True
        )
        total_size = int(response.headers.get("content-length", 0))

        with open(
            os.path.join(save_path, "tracking_data_temp.xml"), "wb"
        ) as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            disable=not verbose,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))

        # rename temp to non temp:
        os.rename(
            os.path.join(save_path, "tracking_data_temp.xml"),
            os.path.join(save_path, "tracking_data.xml"),
        )
        print("Done!", end="\r")

    return load_tracab_tracking_data(
        os.path.join(save_path, "tracking_data.xml"),
        os.path.join(save_path, "metadata.xml"),
        verbose=verbose,
    )


def _get_tracking_data_xml(
    tracab_loc: str,
    home_players: pd.DataFrame,
    away_players: pd.DataFrame,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if verbose:
        print(f"Reading in {tracab_loc}", end="")

    frames = [x for x in range(10000, 200000)]
    size_lines = len(frames)

    data = {
        "frame": frames,
        "ball_x": [np.nan] * size_lines,
        "ball_y": [np.nan] * size_lines,
        "ball_z": [np.nan] * size_lines,
        "ball_status": [None] * size_lines,
        "ball_possession": [None] * size_lines,
        "datetime": ["NaT"] * size_lines,
    }

    frames_df = pd.DataFrame(
        {
            "period_id": [1, 2, 3, 4, 5],
            "start_frame": [MISSING_INT] * 5,
            "end_frame": [MISSING_INT] * 5,
            "start_datetime_td": ["NaT"] * 5,
            "end_datetime_td": ["NaT"] * 5,
        }
    )
    frames_df["start_datetime_td"] = pd.to_datetime(frames_df["start_datetime_td"])
    frames_df["end_datetime_td"] = pd.to_datetime(frames_df["end_datetime_td"])
    fill_first_period = True
    fill_second_period = True

    context = etree.iterparse(tracab_loc, events=("start", "end"))
    event, _ = next(context)

    if verbose:
        context = tqdm(context, total=3400000)

    for event, elem in context:
        if not (event == "end" and elem.tag == "FrameSet"):
            continue
        frames = elem.findall("Frame")
        if fill_first_period and elem.get("GameSection") == "firstHalf":
            frames_df.loc[0, "start_frame"] = int(frames[0].get("N"))
            frames_df.loc[0, "start_datetime_td"] = pd.to_datetime(
                frames[0].get("T")
            ).tz_convert(None)
            frames_df.loc[0, "end_frame"] = int(frames[-1].get("N"))
            frames_df.loc[0, "end_datetime_td"] = pd.to_datetime(
                frames[-1].get("T")
            ).tz_convert(None)
            frame_rate = (
                1
                / (
                    pd.to_datetime(frames[1].get("T")).tz_convert(None)
                    - pd.to_datetime(frames[0].get("T")).tz_convert(None)
                ).total_seconds()
            )
            fill_first_period = False
        if fill_second_period and elem.get("GameSection") == "secondHalf":
            frames_df.loc[1, "start_frame"] = int(frames[0].get("N"))
            frames_df.loc[1, "start_datetime_td"] = pd.to_datetime(
                frames[0].get("T")
            ).tz_convert(None)
            frames_df.loc[1, "end_frame"] = int(frames[-1].get("N"))
            frames_df.loc[1, "end_datetime_td"] = pd.to_datetime(
                frames[-1].get("T")
            ).tz_convert(None)
            fill_second_period = False

        player_id = elem.get("PersonId")
        if player_id in home_players["id"].to_list():
            column_id = "home_" + str(
                home_players.loc[home_players["id"] == player_id, "shirt_num"].iloc[0]
            )
        elif player_id in away_players["id"].to_list():
            column_id = "away_" + str(
                away_players.loc[away_players["id"] == player_id, "shirt_num"].iloc[0]
            )
        else:
            column_id = "ball"

        if column_id + "_x" not in data.keys():
            data[f"{column_id}_x"] = [np.nan] * size_lines
            data[f"{column_id}_y"] = [np.nan] * size_lines

        start_frame = 0 if elem.get("GameSection") == "firstHalf" else 90000
        for i, frame in enumerate(frames):
            data[f"{column_id}_x"][start_frame + i] = float(frame.get("X"))
            data[f"{column_id}_y"][start_frame + i] = float(frame.get("Y"))
            if frame.get("Z") is not None:  # ball
                data[f"{column_id}_z"][start_frame + i] = float(frame.get("Z"))
                data[f"{column_id}_status"][start_frame + i] = (
                    "alive" if frame.get("BallStatus") == "1" else "dead"
                )
                data[f"{column_id}_possession"][start_frame + i] = (
                    "home" if int(frame.get("BallPossession")) == 1 else "away"
                )
                data["datetime"][start_frame + i] = frame.get("T")

    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(
        "Europe/Berlin"
    )
    df = df.dropna(axis=0, how="all", subset=[x for x in data.keys() if x != "frame"])

    frames_df["start_datetime_td"] = (
        frames_df["start_datetime_td"]
        .dt.tz_localize("UTC")
        .dt.tz_convert("Europe/Berlin")
    )
    frames_df["end_datetime_td"] = (
        frames_df["end_datetime_td"]
        .dt.tz_localize("UTC")
        .dt.tz_convert("Europe/Berlin")
    )

    return df, frames_df, frame_rate


def _get_tracking_data_txt(tracab_loc: str, verbose: bool) -> pd.DataFrame:
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

    if soup.find("match") is not None:
        return _get_tracab_metadata(soup)
    elif soup.find("General") is not None:
        return _get_sportec_metadata(metadata_loc)
    else:
        message = "Unknown type of tracab metadata, please open an issue on GitHub."
        LOGGER.error(message)
        raise ValueError(message)


def _get_tracab_metadata(soup: BeautifulSoup) -> Metadata:
    """This version is used in the Netherlands"""

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
    df_home_players = _get_players_metadata_v1(home_players_info)

    away_team = soup.find("AwayTeam")
    away_team_name = away_team.find("LongName").text
    away_team_id = int(away_team.find("TeamId").text)

    away_players_info = []
    for player in away_team.find_all("Player"):
        player_dict = {}
        for element in player.findChildren():
            player_dict[element.name] = element.text
        away_players_info.append(player_dict)
    df_away_players = _get_players_metadata_v1(away_players_info)

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


def _get_players_metadata_v1(
    players_info: list[dict[str, int | float]]
) -> pd.DataFrame:
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
