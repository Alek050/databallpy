import json
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

from databallpy.load_data.metadata import Metadata
from databallpy.utils.tz_modification import utc_to_local_datetime
from databallpy.utils.utils import MISSING_INT
from databallpy.warnings import DataBallPyWarning


def load_ortec_event_data(
    event_data_loc: str, metadata_loc: str
) -> Tuple[pd.DataFrame, Metadata]:
    """Function that loads the event data from Ortec

    Args:
        event_data_loc (str): location of the event data
        metadata_loc (str): location of the metadata

    Note: event data of Ortec is not yet supported since ortec event
    data uses identifiers which are unknown to us. Therefore, we cannot
    parse the event data. For now, None is returned as event data.

    Returns:
        Tuple[pd.DataFrame, Metadata]: event data and metadata of the match
    """
    metadata = load_metadata(metadata_loc)

    if event_data_loc is not None:
        warnings.warn(
            "Ortec event data is not yet supported, returning None",
            DataBallPyWarning,
        )
    return None, metadata


def get_player_info(info_players: list) -> pd.DataFrame:
    """Function that gets the information of the players

    Args:
        info_players (list): list with all players

    Returns:
        pd.DataFrame: dataframe with all player info
    """
    team_info = {
        "id": [],
        "full_name": [],
        "position": [],
        "starter": [],
        "shirt_num": [],
    }

    for player in info_players:
        team_info["full_name"].append(str(player["DisplayName"]))
        team_info["id"].append(int(player["Id"]))
        team_info["position"].append(str(player["Role"]))
        team_info["shirt_num"].append(int(player["ShirtNumber"]))
        if str(player["Role"]) == "bench":
            team_info["starter"].append(False)
        else:
            team_info["starter"].append(True)

    return (
        pd.DataFrame(team_info)
        .sort_values("starter", ascending=False)
        .reset_index(drop=True)
    )


def load_metadata(metadata_loc: str) -> pd.DataFrame:
    """Function that loads metadata from .json file

    Args:
        metadata_loc (str): location of the .json file

    Returns:
        pd.DataFrame: metadata of the match
    """
    with open(metadata_loc, "r") as f:
        data = f.read()
    metadata_json = json.loads(data)

    periods = {
        "period": [1, 2, 3, 4, 5],
        "start_datetime_ed": [pd.to_datetime("NaT")] * 5,
        "end_datetime_ed": [pd.to_datetime("NaT")] * 5,
    }
    country = metadata_json["Competition"]["Name"].split()[0]
    datetime = pd.to_datetime(metadata_json["DateTime"], utc=True)
    start_datetime = utc_to_local_datetime(datetime, country)
    periods["start_datetime_ed"][0] = start_datetime
    periods["end_datetime_ed"][0] = start_datetime + pd.Timedelta(45, "minutes")
    periods["start_datetime_ed"][1] = start_datetime + pd.Timedelta(60, "minutes")
    periods["end_datetime_ed"][1] = start_datetime + pd.Timedelta(105, "minutes")

    info_home_players = metadata_json["HomeTeam"]["Persons"]
    info_away_players = metadata_json["AwayTeam"]["Persons"]
    home_players = get_player_info(info_home_players)
    away_players = get_player_info(info_away_players)

    home_formation = _get_formation(home_players)
    away_formation = _get_formation(away_players)

    metadata = Metadata(
        match_id=metadata_json["Id"],
        pitch_dimensions=[np.nan, np.nan],
        periods_frames=pd.DataFrame(periods),
        frame_rate=MISSING_INT,
        home_team_id=metadata_json["HomeTeam"]["Id"],
        home_team_name=str(metadata_json["HomeTeam"]["DisplayName"]),
        home_players=home_players,
        home_score=MISSING_INT,
        home_formation=home_formation,
        away_team_id=metadata_json["AwayTeam"]["Id"],
        away_team_name=str(metadata_json["AwayTeam"]["DisplayName"]),
        away_players=away_players,
        away_score=MISSING_INT,
        away_formation=away_formation,
        country=country,
    )
    return metadata


def _get_formation(players_info: pd.DataFrame) -> str:
    """Function that gets the formation of the team

    Args:
        players_info (pd.DataFrame): dataframe with all player info

    Returns:
        str: formation of the team
    """
    gk = 0
    defenders = 0
    midfielders = 0
    attackers = 0

    for position in players_info.loc[players_info["position"] != "bench", "position"]:
        if "keeper" in position.lower():
            gk += 1
        if "back" in position.lower():
            defenders += 1
        if "midfield" in position.lower():
            midfielders += 1
        if "forward" in position.lower():
            attackers += 1
    return f"{gk}{defenders}{midfielders}{attackers}"
