import datetime as dt
import json

import chardet
import numpy as np
import pandas as pd

from databallpy.data_parsers import Metadata
from databallpy.utils.constants import DATABALLPY_POSITIONS, MISSING_INT
from databallpy.utils.logging import logging_wrapper
from databallpy.utils.tz_modification import utc_to_local_datetime

INSTAT_DATABALLPY_MAP = {
    "Attacking pass accurate": ["pass", True],
    "Attacking pass inaccurate": ["pass", False],
    "Unsuccessful dribbling": ["dribble", False],
    "Successful dribbling": ["dribble", True],
    "Dribbling": ["dribble", None],
    "Inaccurate key pass": ["pass", False],
    "Crosses inaccurate": ["pass", True],
    "Blocked shot": ["shot", False],
    "Shots blocked": ["shot", False],
    "Wide shot": ["shot", False],
    "Accurate crossing from set piece with a shot": ["pass", True],
    "Shot on target": ["shot", False],
    "Crosses accurate": ["pass", True],
    "Pass into offside": ["pass", False],
    "Accurate key pass": ["pass", True],
    "Shot blocked by field player": ["shot", False],
    "Inaccurate set-piece cross": ["pass", False],
    "Accurate crossing from set piece": ["pass", True],
    "Key assist": ["pass", True],
    "Goal": ["shot", True],
    "Accurate crossing from set piece with a goal": ["pass", 1],
}

logging_wrapper(__file__)


def load_instat_event_data(
    event_data_loc: str, metadata_loc: str
) -> tuple[pd.DataFrame, Metadata]:
    """This function retrieves the metadata and event data of a specific game. The x
    and y coordinates provided have been scaled to the dimensions of the pitch, with
    (0, 0) being the center. Additionally, the coordinates have been standardized so
    that the home team is represented as playing from left to right for the entire
    game, and the away team is represented as playing from right to left.

    Args:
        event_data_loc (str): location of the event_data.json file
        event_data_metadata_loc (str): location of the metadata.json file

    Returns:
        Tuple[pd.DataFrame, Metadata]: the event data of the game and the  metadata
    """
    if not isinstance(event_data_loc, str):
        raise TypeError(
            f"event_data_loc should be a string, not a {type(event_data_loc)}"
        )
    if not isinstance(metadata_loc, str):
        raise TypeError(f"metadata_loc should be a string, not a {type(metadata_loc)}")
    if not event_data_loc[-5:] == ".json":
        raise ValueError("instat event file should by of .json format")
    if not metadata_loc[-5:] == ".json":
        raise ValueError("instat event metadata file should be of .json format")

    metadata = _load_metadata(metadata_loc=metadata_loc)
    metadata = _update_metadata(metadata=metadata, event_data_loc=event_data_loc)
    event_data, pitch_dimensions = _load_event_data(
        event_data_loc=event_data_loc, metadata=metadata
    )
    metadata.pitch_dimensions = pitch_dimensions

    return event_data, metadata, {}


logging_wrapper(__file__)


def _load_metadata(metadata_loc: str) -> pd.DataFrame:
    """Function to load the data from the metadata.json file

    Args:
        metdata_loc (str): location of the metadata.json file

    Returns:
        pd.DataFrame: metadata of the game
    """
    with open(metadata_loc, "rb") as f:
        encoding = chardet.detect(f.read())["encoding"]
    with open(metadata_loc, "r", encoding=encoding) as f:
        data = f.read()
    metadata_json = json.loads(data)
    game_info = metadata_json["data"]["match_info"][0]
    country = game_info["tournament_name"].split(".")[0]
    periods = {
        "period_id": [1, 2, 3, 4, 5],
        "start_datetime_ed": [pd.to_datetime("NaT", utc=True)] * 5,
        "end_datetime_ed": [pd.to_datetime("NaT", utc=True)] * 5,
    }

    # No idea why the instat times need to be subtracted by 3 hours to get to utc time
    periods["start_datetime_ed"][0] = pd.to_datetime(
        game_info["match_date"], utc=True
    ) - dt.timedelta(hours=3)
    periods["end_datetime_ed"][0] = periods["start_datetime_ed"][0] + dt.timedelta(
        minutes=45
    )
    periods["start_datetime_ed"][1] = periods["start_datetime_ed"][0] + dt.timedelta(
        hours=1
    )
    periods["end_datetime_ed"][1] = periods["start_datetime_ed"][0] + dt.timedelta(
        minutes=45, hours=1
    )

    periods = pd.DataFrame(periods)

    # set time to local time
    periods["start_datetime_ed"] = utc_to_local_datetime(
        periods["start_datetime_ed"], country
    )
    periods["end_datetime_ed"] = utc_to_local_datetime(
        periods["end_datetime_ed"], country
    )

    metadata = Metadata(
        game_id=int(game_info["id"]),
        pitch_dimensions=[np.nan, np.nan],
        periods_frames=pd.DataFrame(periods),
        frame_rate=np.nan,
        home_team_id=int(game_info["team1_id"]),
        home_team_name=str(game_info["team1_name"]),
        home_players=pd.DataFrame(columns=["id", "full_name", "shirt_num"]),
        home_score=int(game_info["score"].split(":")[0]),
        home_formation="",
        away_team_id=int(game_info["team2_id"]),
        away_team_name=str(game_info["team2_name"]),
        away_players=pd.DataFrame(columns=["id", "full_name", "shirt_num"]),
        away_score=int(game_info["score"].split(":")[1]),
        away_formation="",
        country=country,
    )

    return metadata


def _update_metadata(metadata: Metadata, event_data_loc: str) -> pd.DataFrame:
    """This function updates the metadata with the information in the
    event_data.json file

    Args:
        metadata (Metadata): metadata loaded from the metadata.json file
        event_data_loc (str): location of the event_data.json file

    Returns:
        pd.DataFrame: updated metadata of the game
    """
    with open(event_data_loc, "rb") as f:
        encoding = chardet.detect(f.read())["encoding"]
    with open(event_data_loc, "r", encoding=encoding) as f:
        data = f.read()
    event_data_json = json.loads(data)
    events = event_data_json["data"]["row"]

    players_dict = {
        "id": [],
        "full_name": [],
        "position": [],
        "starter": [],
        "shirt_num": [],
        "team_id": [],
    }

    home_formation = ""
    away_formation = ""

    for event in events:
        if event["action_id"].startswith(
            "16"
        ):  # events starting with 16 contain player metadata
            if int(event["player_id"]) not in players_dict["id"]:
                players_dict["id"].append(int(event["player_id"]))
                players_dict["full_name"].append(str(event["player_name"]))
                players_dict["position"].append(
                    _parse_instat_position(str(event["position_name"]))
                )
                if str(event["position_name"]) == "Substitute player":
                    players_dict["starter"].append(False)
                else:
                    players_dict["starter"].append(True)
                players_dict["shirt_num"].append(int(event["number"]))
                players_dict["team_id"].append(int(event["team_id"]))
        if event["action_id"].startswith(
            "15"
        ):  # events starting with 15 contain formations
            if int(event["team_id"]) == metadata.home_team_id and home_formation == "":
                home_formation = str(event["action_name"]).replace("-", "")
            if int(event["team_id"]) == metadata.away_team_id and away_formation == "":
                away_formation = str(event["action_name"]).replace("-", "")

    df_players = pd.DataFrame(players_dict)
    home_players = df_players[df_players["team_id"] == metadata.home_team_id]
    home_players = (
        home_players.sort_values("starter", ascending=False)
        .drop("team_id", axis=1)
        .reset_index(drop=True)
    )
    away_players = df_players[df_players["team_id"] == metadata.away_team_id]
    away_players = (
        away_players.sort_values("starter", ascending=False)
        .drop("team_id", axis=1)
        .reset_index(drop=True)
    )

    metadata.home_formation = home_formation
    metadata.away_formation = away_formation
    metadata.home_players = home_players
    metadata.away_players = away_players
    return metadata


def _parse_instat_position(position: str) -> str:
    for pos in DATABALLPY_POSITIONS:
        if pos in position.lower():
            return pos
    return "unspecified"


logging_wrapper(__file__)


def _load_event_data(event_data_loc: str, metadata: Metadata) -> pd.DataFrame:
    """Function to load the event_data.json file, the events of the game.
    Note: this function does ignore qualifiers for now


    Args:
        event_data_loc (str): location of the event_data.json file
        metadata(Metadata): metadata of the game

    Returns:
        pd.DataFrame: event data of the game
    """
    with open(event_data_loc, "rb") as f:
        encoding = chardet.detect(f.read())["encoding"]
    with open(event_data_loc, "r", encoding=encoding) as f:
        data = f.read()
    event_data_json = json.loads(data)
    events = event_data_json["data"]["row"]

    result_dict = {
        "event_id": [],
        "databallpy_event": [],
        "period_id": [],
        "minutes": [],
        "seconds": [],
        "player_id": [],
        "team_id": [],
        "is_successful": [],
        "start_x": [],
        "start_y": [],
        "end_x": [],
        "end_y": [],
        "datetime": [],
        "original_event_id": [],
        "original_event": [],
        "event_type_id": [],
    }

    start_time_period = {
        1: metadata.periods_frames.loc[0, "start_datetime_ed"],
        2: metadata.periods_frames.loc[0, "start_datetime_ed"]
        + dt.timedelta(minutes=60),
        3: metadata.periods_frames.loc[0, "start_datetime_ed"]
        + dt.timedelta(minutes=110),
        4: metadata.periods_frames.loc[0, "start_datetime_ed"]
        + dt.timedelta(minutes=130),
        5: metadata.periods_frames.loc[0, "start_datetime_ed"]
        + dt.timedelta(minutes=150),
    }

    for i_event, event in enumerate(events):
        if not event["action_id"].startswith(("16", "15")):
            result_dict["event_id"].append(i_event)
            result_dict["original_event_id"].append(int(event["id"]))
            result_dict["event_type_id"].append(int(event["action_id"]))
            result_dict["original_event"].append(str(event["action_name"]))
            result_dict["period_id"].append(int(event["half"]))
            result_dict["minutes"].append(float(event["second"]) // 60)
            result_dict["seconds"].append(float(event["second"]) % 60)

            if "player_id" in event.keys():
                result_dict["player_id"].append(int(event["player_id"]))
            else:
                result_dict["player_id"].append(MISSING_INT)

            if "team_id" in event.keys():
                result_dict["team_id"].append(int(event["team_id"]))
            else:
                result_dict["team_id"].append(MISSING_INT)

            if "pos_x" in event.keys():
                result_dict["start_x"].append(float(event["pos_x"]))
                result_dict["start_y"].append(float(event["pos_y"]))
            else:
                result_dict["start_x"].append(np.nan)
                result_dict["start_y"].append(np.nan)

            if "pos_dest_x" in event.keys():
                result_dict["end_x"].append(float(event["pos_dest_x"]))
                result_dict["end_y"].append(float(event["pos_dest_y"]))
            else:
                result_dict["end_x"].append(np.nan)
                result_dict["end_y"].append(np.nan)

            result_dict["datetime"].append(
                start_time_period[int(event["half"])]
                + dt.timedelta(milliseconds=float(event["second"]) * 1000)
            )

    result_dict["is_successful"] = [None] * len(result_dict["original_event_id"])
    result_dict["databallpy_event"] = [None] * len(result_dict["original_event_id"])

    event_data = pd.DataFrame(result_dict)
    event_data["databallpy_event"] = event_data["original_event"].apply(
        lambda x: INSTAT_DATABALLPY_MAP.get(x, [None])[0]
    )
    event_data["is_successful"] = (
        event_data["original_event"]
        .apply(lambda x: INSTAT_DATABALLPY_MAP.get(x, [None, None])[1])
        .astype("boolean")
    )

    potential_kick_off_events = ["pass", "shot"]
    x_start, y_start = (
        event_data[event_data["databallpy_event"].isin(potential_kick_off_events)]
        .reset_index()
        .loc[0, ["start_x", "start_y"]]
    )
    event_data["start_x"] -= x_start
    event_data["start_y"] -= y_start
    event_data["end_x"] -= x_start
    event_data["end_y"] -= y_start

    pitch_dimensions = [2 * x_start, 2 * y_start]

    id_full_name_dict = dict(
        zip(metadata.home_players["id"], metadata.home_players["full_name"])
    )
    id_full_name_dict_away = dict(
        zip(metadata.away_players["id"], metadata.away_players["full_name"])
    )
    id_full_name_dict.update(id_full_name_dict_away)
    event_data["player_name"] = (
        event_data["player_id"].map(id_full_name_dict).replace({np.nan: None})
    )
    away_mask = event_data["team_id"] == metadata.away_team_id
    event_data.loc[away_mask, ["start_x", "start_y", "end_x", "end_y"]] *= -1
    event_data["minutes"] = event_data["minutes"].round().astype(int)

    return event_data, pitch_dimensions
