import datetime as dt
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import chardet
import numpy as np
import pandas as pd
import requests
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
    _downcast_tracking_data,
    _get_frame_selection,
    _get_gametime,
    _get_period_start_frames,
    _insert_missing_rows,
    _normalize_playing_direction_tracking,
)
from databallpy.utils.constants import MISSING_INT
from databallpy.utils.logging import logging_wrapper
from databallpy.utils.tz_modification import localize_datetime


@logging_wrapper(__file__)
def load_tracab_tracking_data(
    tracab_loc: str,
    metadata_loc: str,
    verbose: bool = True,
    *,
    period_id: int | list[int] | None = None,
    frames: tuple[int, int] | None = None,
) -> tuple[pd.DataFrame, Metadata]:
    """Function to load tracking data and metadata from the tracab format

    Args:
        tracab_loc (str): location of the tracking_data.dat file
        metadata_loc (str): location of the meta_data.xml file
        verbose (bool): whether to print on progress of loading in the terminal,
        defaults to True
        period_id (int | list[int], optional): only load the frames of this period,
            or of these periods. Defaults to None (load all periods).
        frames (tuple[int, int], optional): only load the frames between the first
            and the last frame, inclusive. Defaults to None (load all frames).

    Returns:
        Tuple[pd.DataFrame, Metadata]: the tracking data and metadata class

    Note:
        The metadata always describes the full game, also when only part of the
        tracking data is loaded. When `period_id` or `frames` is specified, the
        start and end frames of the periods are taken from the metadata as is,
        while they are estimated based on the tracking data for a full load.
    """

    metadata = _get_metadata(metadata_loc)
    if tracab_loc.endswith(".dat") or tracab_loc.endswith(".txt"):
        selection = _get_frame_selection(metadata.periods_frames, period_id, frames)
        tracking_data = _get_tracking_data_txt(
            tracab_loc,
            verbose,
            selection=selection,
            extra_frames=_get_period_start_frames(metadata.periods_frames),
        )
        tracking_data["datetime"] = _add_datetime(
            tracking_data["frame"],
            metadata.frame_rate,
            metadata.periods_frames["start_datetime_td"].iloc[0],
        )
    elif tracab_loc.endswith(".xml"):
        tracking_data, periods_frames, frame_rate = _get_tracking_data_xml(
            tracab_loc,
            metadata.home_players,
            metadata.away_players,
            verbose,
            period_id=period_id,
            frames=frames,
        )
        metadata.periods_frames = periods_frames
        metadata.frame_rate = int(frame_rate)
        selection = _get_frame_selection(periods_frames, period_id, frames)
        tracking_data = _insert_missing_rows(
            tracking_data.reset_index(drop=True), "frame", selection=selection
        )
    else:
        message = "Tracab tracking data should be either .txt, .dat, or .xml format."
        raise ValueError(message)

    tracking_data.insert(
        len(tracking_data.columns) - 1,
        "period_id",
        _add_periods_to_tracking_data(tracking_data["frame"], metadata.periods_frames),
    )
    if selection is None:
        tracking_data, metadata = _adjust_start_end_frames(tracking_data, metadata)

    tracking_data["gametime_td"] = _get_gametime(
        tracking_data["frame"], tracking_data["period_id"], metadata
    )
    tracking_data, changed_periods = _normalize_playing_direction_tracking(
        tracking_data, metadata.periods_frames
    )
    metadata.periods_changed_playing_direction = changed_periods

    if selection is not None:
        tracking_data = tracking_data[
            tracking_data["frame"].between(*selection)
        ].reset_index(drop=True)

    return _downcast_tracking_data(tracking_data), metadata


@logging_wrapper(__file__)
def load_sportec_open_tracking_data(
    game_id: str, verbose: bool, cache_path: Path
) -> tuple[pd.DataFrame, Metadata]:
    """Load the tracking data from the sportec open data platform

    Args:
        game_id (str): The id of the game
        verbose (bool): Whether to print info about the loading of the data.
        cache_path (Path): path to cache files.

    Returns:
        tuple[pd.DataFrame, Metadata]: the tracking data and metadata class

    Reference:
        Bassek, M., Weber, H., Rein, R., & Memmert,D. (2024). An integrated
        dataset of synchronized spatiotemporal and event data in elite soccer.
    """
    metadata_url = _get_sportec_open_data_url(game_id, "metadata")
    os.makedirs(cache_path, exist_ok=True)

    metadata = requests.get(metadata_url)
    with open(cache_path / "metadata_temp.xml", "wb") as f:
        f.write(metadata.content)

    if verbose:
        print("Downloading open tracking data...", end="\r")
    session = requests.Session()
    response = session.get(
        _get_sportec_open_data_url(game_id, "tracking_data"), stream=True
    )
    total_size = int(response.headers.get("content-length", 0))

    with (
        open(cache_path / "tracking_data_temp.xml", "wb") as file,
        tqdm(
            desc="Downloading",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            disable=not verbose,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

    print("Done!", end="\r")
    return load_tracab_tracking_data(
        str(cache_path / "tracking_data_temp.xml"),
        str(cache_path / "metadata_temp.xml"),
        verbose=verbose,
    )


@logging_wrapper(__file__)
def _get_tracking_data_xml(
    tracab_loc: str,
    home_players: pd.DataFrame,
    away_players: pd.DataFrame,
    verbose: bool,
    *,
    period_id: int | list[int] | None = None,
    frames: tuple[int, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if verbose:
        print(f"Reading in {tracab_loc}", end="")

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

    context = ET.iterparse(tracab_loc, events=("start", "end"))
    event, _ = next(context)

    frame_values = []
    n_elements = 0
    n_frames_first_half = None
    frame_rate = None
    # first find the two frame sets of the ball to initialize the frames
    for event, elem in context:
        n_elements += 1
        if event == "end" and elem.tag == "FrameSet" and elem.get("TeamId") == "BALL":
            frame_elements = elem.findall("Frame")
            frame_values.extend([int(x.get("N")) for x in frame_elements])
            game_section = elem.get("GameSection")
            frame_rate, n_frames_first_half = process_game_section(
                frame_elements, game_section, frames_df, frame_rate, n_frames_first_half
            )

    selection = _get_frame_selection(frames_df, period_id, frames)
    if selection is not None:
        first_frame, last_frame = selection
        extra_frames = _get_period_start_frames(frames_df)
        frame_values = [
            x
            for x in frame_values
            if first_frame <= x <= last_frame or x in extra_frames
        ]
    frame_to_idx = {frame: idx for idx, frame in enumerate(frame_values)}

    size_lines = len(frame_values)
    data = {
        "frame": frame_values,
        "ball_x": [np.nan] * size_lines,
        "ball_y": [np.nan] * size_lines,
        "ball_z": [np.nan] * size_lines,
        "ball_status": [None] * size_lines,
        "team_possession": [None] * size_lines,
        "datetime": ["NaT"] * size_lines,
    }

    context = ET.iterparse(tracab_loc, events=("start", "end"))
    event, _ = next(context)

    if verbose:
        context = tqdm(context, total=n_elements)

    for event, elem in context:
        if not (event == "end" and elem.tag == "FrameSet"):
            continue
        frame_elements = elem.findall("Frame")
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

        for frame in frame_elements:
            i = frame_to_idx.get(int(frame.get("N")))
            if i is None:  # frame is not part of the selection
                continue

            data[f"{column_id}_x"][i] = float(frame.get("X"))
            data[f"{column_id}_y"][i] = float(frame.get("Y"))
            if frame.get("Z") is not None:  # ball
                data[f"{column_id}_z"][i] = float(frame.get("Z"))
                data[f"{column_id}_status"][i] = (
                    "alive" if frame.get("BallStatus") == "1" else "dead"
                )
                data["team_possession"][i] = (
                    "home" if int(frame.get("BallPossession")) == 1 else "away"
                )
                data["datetime"][i] = frame.get("T")

    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(
        "Europe/Berlin"
    )

    frames_df["start_datetime_td"] = (
        frames_df["start_datetime_td"]
        .dt.tz_localize("UTC")
        .dt.tz_convert("Europe/Berlin")
    )
    frames_df["end_datetime_td"] = (
        frames_df["end_datetime_td"].dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
    )
    return _downcast_tracking_data(df), frames_df, frame_rate


def process_game_section(
    frames, game_section, frames_df, frame_rate=None, n_frames_first_half=None
):
    if game_section == "firstHalf":
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
        n_frames_first_half = len(frames)
    else:  # second half
        frames_df.loc[1, "start_frame"] = int(frames[0].get("N"))
        frames_df.loc[1, "start_datetime_td"] = pd.to_datetime(
            frames[0].get("T")
        ).tz_convert(None)
        frames_df.loc[1, "end_frame"] = int(frames[-1].get("N"))
        frames_df.loc[1, "end_datetime_td"] = pd.to_datetime(
            frames[-1].get("T")
        ).tz_convert(None)

    return frame_rate, n_frames_first_half


@logging_wrapper(__file__)
def _get_tracking_data_txt(
    tracab_loc: str,
    verbose: bool,
    *,
    selection: tuple[int, int] | None = None,
    extra_frames: set[int] | None = None,
) -> pd.DataFrame:
    """Function that reads tracking data from .dat file and stores it in a pd.DataFrame

    Args:
        tracab_loc (str): location of the tracking_data.dat file
        verbose (bool): whether to print info in terminal
        selection (tuple[int, int], optional): the first and last frame to load,
            inclusive. Defaults to None (load all frames).
        extra_frames (set[int], optional): frames outside of the selection that
            should be loaded as well. Defaults to None.

    Returns:
        pd.DataFrame: contains tracking data
    """

    if verbose:
        print(f"Reading in {tracab_loc}", end="")

    with open(tracab_loc, "r") as file:
        if selection is None:
            lines = file.readlines()
        else:
            first_frame, last_frame = selection
            extra_frames = extra_frames if extra_frames is not None else set()
            lines = [
                line
                for line in file
                if first_frame <= (frame := int(line.split(":", 1)[0])) <= last_frame
                or frame in extra_frames
            ]
    if verbose:
        print(" - Completed")

    size_lines = len(lines)

    data = {
        "frame": [np.nan] * size_lines,
        "ball_x": [np.nan] * size_lines,
        "ball_y": [np.nan] * size_lines,
        "ball_z": [np.nan] * size_lines,
        "ball_status": [None] * size_lines,
        "team_possession": [None] * size_lines,
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
    df = _insert_missing_rows(df, "frame", selection=selection)
    return _downcast_tracking_data(df)


@logging_wrapper(__file__)
def _get_metadata(metadata_loc: str) -> Metadata:
    """Function that reads metadata (supports both .json and .xml) file and
    stores it in Metadata class

    Args:
        meta_data_loc (str): location of the metadata file

    Returns:
        Metadata: class that contains metadata
    """

    format = metadata_loc.split(".")[-1]

    if format == "json":
        with open(metadata_loc, "r") as file:
            content = json.load(file)

            return _get_tracab_metadata_json(content)
    else:
        with open(metadata_loc, "rb") as file:
            encoding = chardet.detect(file.read())["encoding"]
        with open(metadata_loc, "r", encoding=encoding) as file:
            lines = file.read()

        lines = lines.replace("ï»¿", "")
        root = ET.fromstring(lines)

        if root.find(".//match") is not None:
            return _get_tracab_metadata_xml(root)
        elif root.find(".//General") is not None:
            return _get_sportec_metadata(metadata_loc)
        else:
            message = "Unknown type of tracab metadata, please open an issue on GitHub."
            raise ValueError(message)


@logging_wrapper(__file__)
def _get_tracab_metadata_json(metadata: dict) -> Metadata:
    """
    Function that reads metadata from .json file and stores it in Metadata class.
    """
    game_id = int(metadata["GameID"])
    pitch_size_x = float(metadata["PitchLongSide"]) / 100
    pitch_size_y = float(metadata["PitchShortSide"]) / 100
    frame_rate = int(metadata["FrameRate"])
    datetime_string = metadata["Kickoff"]
    date = pd.to_datetime(datetime_string[:10])

    frames_dict = {
        "period_id": [],
        "start_frame": [],
        "end_frame": [],
        "start_datetime_td": [],
        "end_datetime_td": [],
    }

    for i in range(1, 6):
        frames_dict["period_id"].append(i)

        if metadata[f"Phase{i}StartFrame"] != 0:
            start_frame = int(metadata[f"Phase{i}StartFrame"])
            end_frame = int(metadata[f"Phase{i}EndFrame"])

            frames_dict["start_frame"].append(start_frame)
            frames_dict["end_frame"].append(end_frame)

            start_frame_corrected = start_frame % (frame_rate * 60 * 60 * 24)
            end_frame_corrected = end_frame % (frame_rate * 60 * 60 * 24)

            frames_dict["start_datetime_td"].append(
                date
                + dt.timedelta(
                    milliseconds=int((start_frame_corrected / frame_rate) * 1_000)
                )
            )
            frames_dict["end_datetime_td"].append(
                date
                + dt.timedelta(
                    milliseconds=int((end_frame_corrected / frame_rate) * 1_000)
                )
            )
        else:
            frames_dict["start_frame"].append(MISSING_INT)
            frames_dict["end_frame"].append(MISSING_INT)
            frames_dict["start_datetime_td"].append(pd.NaT)
            frames_dict["end_datetime_td"].append(pd.NaT)

    df_frames = pd.DataFrame(frames_dict)

    df_frames["start_datetime_td"] = localize_datetime(
        df_frames["start_datetime_td"], "Netherlands"
    )
    df_frames["end_datetime_td"] = localize_datetime(
        df_frames["end_datetime_td"], "Netherlands"
    )

    home_team_id = metadata["HomeTeam"]["TeamID"]
    home_team_name = metadata["HomeTeam"]["LongName"]
    home_players_info = []

    for player in metadata["HomeTeam"]["Players"]:
        home_players_info.append(
            {
                "PlayerId": player["PlayerID"],
                "FirstName": player["FirstName"],
                "LastName": player["LastName"],
                "JerseyNo": player["JerseyNo"],
                "StartFrameCount": player["StartFrameCount"],
                "EndFrameCount": player["EndFrameCount"],
            }
        )

    away_team_id = metadata["AwayTeam"]["TeamID"]
    away_team_name = metadata["AwayTeam"]["LongName"]
    away_players_info = []

    for player in metadata["AwayTeam"]["Players"]:
        away_players_info.append(
            {
                "PlayerId": player["PlayerID"],
                "FirstName": player["FirstName"],
                "LastName": player["LastName"],
                "JerseyNo": player["JerseyNo"],
                "StartFrameCount": player["StartFrameCount"],
                "EndFrameCount": player["EndFrameCount"],
            }
        )

    df_home_players = _get_players_metadata_v1(home_players_info)
    df_away_players = _get_players_metadata_v1(away_players_info)

    return Metadata(
        game_id=game_id,
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


@logging_wrapper(__file__)
def _get_tracab_metadata_xml(root: ET.Element) -> Metadata:
    """This version is used in the Netherlands"""

    match_elem = root.find(".//match")
    game_id = int(match_elem.get("iId"))
    pitch_size_x = float(match_elem.get("fPitchXSizeMeters"))
    pitch_size_y = float(match_elem.get("fPitchYSizeMeters"))
    frame_rate = int(match_elem.get("iFrameRateFps"))
    datetime_string = match_elem.get("dtDate")
    date = pd.to_datetime(datetime_string[:10])

    frames_dict = {
        "period_id": [],
        "start_frame": [],
        "end_frame": [],
        "start_datetime_td": [],
        "end_datetime_td": [],
    }
    for _, period in enumerate(root.findall(".//period")):
        frames_dict["period_id"].append(int(period.get("iId")))
        start_frame = int(period.get("iStartFrame"))
        end_frame = int(period.get("iEndFrame"))

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
    home_team = root.find(".//HomeTeam")
    home_team_name = home_team.find("LongName").text
    home_team_id = int(home_team.find("TeamId").text)
    home_players_info = []
    for player in home_team.findall(".//Player"):
        player_dict = {}
        for element in player:
            player_dict[element.tag] = element.text
        home_players_info.append(player_dict)
    df_home_players = _get_players_metadata_v1(home_players_info)

    away_team = root.find(".//AwayTeam")
    away_team_name = away_team.find("LongName").text
    away_team_id = int(away_team.find("TeamId").text)

    away_players_info = []
    for player in away_team.findall(".//Player"):
        player_dict = {}
        for element in player:
            player_dict[element.tag] = element.text
        away_players_info.append(player_dict)
    df_away_players = _get_players_metadata_v1(away_players_info)

    metadata = Metadata(
        game_id=game_id,
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


def _get_players_metadata_v1(players_info: list[dict[str, int | float]]) -> pd.DataFrame:
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
        first_name = player["FirstName"] or ""
        full_name = first_name + " " + player["LastName"]
        if not first_name:
            full_name = full_name.lstrip()
        player_dict["full_name"].append(full_name)
        player_dict["shirt_num"].append(int(player["JerseyNo"]))
        player_dict["start_frame"].append(int(player["StartFrameCount"]))
        player_dict["end_frame"].append(int(player["EndFrameCount"]))
    df = pd.DataFrame(player_dict)
    df["starter"] = df["start_frame"] == df["start_frame"].value_counts().index[0]
    df["position"] = "unspecified"

    return df
