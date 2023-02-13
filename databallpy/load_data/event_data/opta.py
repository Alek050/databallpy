import datetime as dt
from typing import Tuple

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from databallpy.load_data.metadata import Metadata

EVENT_TYPE_IDS = {
    1: "pass",
    2: "offside pass",
    3: "take on",
    4: "foul",
    5: "out",
    6: "corner awarded",
    7: "tackle",
    8: "interception",
    9: "turnover",
    10: "save",
    11: "claim",
    12: "clearance",
    13: "miss",
    14: "post",
    15: "attempt saved",
    16: "goal",
    17: "card",
    18: "player off",
    19: "player on",
    20: "player retired",
    21: "player returns",
    22: "player becomes goalkeeper",
    23: "goalkeeper becomes player",
    24: "condition change",
    25: "official change",
    27: "start delay",
    28: "end delay",
    29: "unknown event 29",
    30: "end",
    31: "unknown event 31",
    32: "start",
    33: "unknown event 33",
    34: "team set up",
    35: "player changed position",
    36: "player changed jersey number",
    37: "collection end",
    38: "temp_goal",
    39: "temp_attempt",
    40: "formation change",
    41: "punch",
    42: "good skill",
    43: "deleted event",
    44: "aerial",
    45: "challenge",
    46: "unknown event 46",
    47: "rescinded card",
    48: "unknown event 46",
    49: "ball recovery",
    50: "dispossessed",
    51: "error",
    52: "keeper pick-up",
    53: "cross not claimed",
    54: "smother",
    55: "offside provoked",
    56: "shield ball opp",
    57: "foul throw in",
    58: "penalty faced",
    59: "keeper sweeper",
    60: "chance missed",
    61: "ball touch",
    62: "unknown event 62",
    63: "temp_save",
    64: "resume",
    65: "contentious referee decision",
    66: "possession data",
    67: "50/50",
    68: "referee drop ball",
    69: "failed to block",
    70: "injury time announcement",
    71: "coach setup",
    72: "caught offside",
    73: "other ball contact",
    74: "blocked pass",
    75: "delayed start",
    76: "early end",
    77: "player off pitch",
}


def load_opta_event_data(
    f7_loc: str, f24_loc: str, pitch_dimensions: list = [106.0, 68.0]
) -> Tuple[pd.DataFrame, Metadata]:
    """This function retrieves the metadata and event data of a specific match. The x
    and y coordinates provided have been scaled to the dimensions of the pitch, with
    (0, 0) being the center. Additionally, the coordinates have been standardized so
    that the home team is represented as playing from left to right for the entire
    match, and the away team is represented as playing from right to left.

    Args:
        f7_loc (str): location of the f7.xml file.
        f24_loc (str): location of the f24.xml file.
        pitch_dimensions (list, optional): the length and width of the pitch in meters

    Returns:
        Tuple[pd.DataFrame, Metadata]: the event data of the match and the metadata
    """

    assert isinstance(f7_loc, str), f"f7_loc should be a string, not a {type(f7_loc)}"
    assert isinstance(
        f24_loc, str
    ), f"f24_loc should be a string, not a {type(f24_loc)}"
    assert f7_loc[-4:] == ".xml", "f7 opta file should by of .xml format"
    assert f24_loc[-4:] == ".xml", "f24 opta file should be of .xml format"

    event_data = _load_event_data(f24_loc)
    metadata = _load_metadata(f7_loc, pitch_dimensions=pitch_dimensions)

    # Add player names to the event data dataframe
    home_players = dict(
        zip(metadata.home_players["id"], metadata.home_players["full_name"])
    )
    away_players = dict(
        zip(metadata.away_players["id"], metadata.away_players["full_name"])
    )

    home_mask = (event_data["team_id"] == metadata.home_team_id) & ~pd.isnull(
        event_data["player_id"]
    )
    away_mask = (event_data["team_id"] == metadata.away_team_id) & ~pd.isnull(
        event_data["player_id"]
    )

    event_data.loc[home_mask, "player_name"] = event_data.loc[
        home_mask, "player_id"
    ].map(home_players)
    event_data.loc[away_mask, "player_name"] = event_data.loc[
        away_mask, "player_id"
    ].map(away_players)

    # Rescale the x and y coordinates relative to the pitch dimensions
    # The original dimension of the x and y coordinates range from 0 to 100
    event_data.loc[:, ["start_x"]] = (
        event_data.loc[:, ["start_x"]] / 100 * pitch_dimensions[0]
    ) - (pitch_dimensions[0] / 2.0)
    event_data.loc[:, ["start_y"]] = (
        event_data.loc[:, ["start_y"]] / 100 * pitch_dimensions[1]
    ) - (pitch_dimensions[1] / 2.0)

    # Change direction of play of the away team so it is represented from right to left
    event_data.loc[
        event_data["team_id"] == metadata.away_team_id, ["start_x", "start_y"]
    ] *= -1

    return event_data, metadata


def _load_metadata(f7_loc: str, pitch_dimensions: list) -> Metadata:
    """Function to load metadata from the f7.xml opta file

    Args:
        f7_loc (str): location of the f7.xml opta file
        pitch_dimensions (list): the length and width of the pitch in meters

    Returns:
        MetaData: all metadata information of the current match
    """
    file = open(f7_loc, "r")
    lines = file.read()
    soup = BeautifulSoup(lines, "xml")

    # Obtain match id
    match_id = int(soup.find("SoccerDocument").attrs["uID"][1:])

    # Obtain match start and end of period datetime
    periods = {
        "period": [1, 2],
        "start_datetime_opta": [],
        "end_datetime_opta": [],
    }
    start_period_1 = soup.find("Stat", attrs={"Type": "first_half_start"})
    end_period_1 = soup.find("Stat", attrs={"Type": "first_half_stop"})
    start_period_2 = soup.find("Stat", attrs={"Type": "second_half_start"})
    end_period_2 = soup.find("Stat", attrs={"Type": "second_half_stop"})
    for start, end in zip(
        [start_period_1, start_period_2], [end_period_1, end_period_2]
    ):
        # Add one hour to go from utc to Dutch time
        periods["start_datetime_opta"].append(
            pd.to_datetime(start.contents[0]) + dt.timedelta(hours=1)
        )
        periods["end_datetime_opta"].append(
            pd.to_datetime(end.contents[0]) + dt.timedelta(hours=1)
        )

    # Opta has a TeamData and Team attribute in the f7 file
    team_datas = soup.find_all("TeamData")
    teams = soup.find_all("Team")
    teams_info = {}
    teams_player_info = {}
    for team_data, team in zip(team_datas, teams):

        # Team information
        team_name = team.findChildren("Name")[0].contents[0]
        team_info = {}
        team_info["team_name"] = team_name
        team_info["side"] = team_data["Side"].lower()
        team_info["formation"] = team_data["Formation"]
        team_info["score"] = int(team_data["Score"])
        team_info["team_id"] = int(team_data["TeamRef"][1:])
        teams_info[team_info["side"]] = team_info

        # Player information
        players_data = [
            player.attrs for player in team_data.findChildren("MatchPlayer")
        ]
        players_names = {}
        for player in team.findChildren("Player"):
            player_id = int(player.attrs["uID"][1:])
            first_name = player.contents[1].contents[1].text

            if "Last" in str(player.contents[1].contents[3]):
                last_name_idx = 3
            else:
                last_name_idx = 5
            last_name = player.contents[1].contents[last_name_idx].contents[0]
            if first_name:
                players_names[str(player_id)] = f"{first_name} {last_name}"
            else:
                players_names[str(player_id)] = last_name

        player_info = _get_player_info(players_data, players_names)
        teams_player_info[team_info["side"]] = player_info

    file.close()

    metadata = Metadata(
        match_id=match_id,
        pitch_dimensions=pitch_dimensions,
        periods_frames=pd.DataFrame(periods),
        frame_rate=np.nan,
        home_team_id=teams_info["home"]["team_id"],
        home_team_name=teams_info["home"]["team_name"],
        home_players=teams_player_info["home"],
        home_score=teams_info["home"]["score"],
        home_formation=teams_info["home"]["formation"],
        away_team_id=teams_info["away"]["team_id"],
        away_team_name=teams_info["away"]["team_name"],
        away_players=teams_player_info["away"],
        away_score=teams_info["away"]["score"],
        away_formation=teams_info["away"]["formation"],
    )
    return metadata


def _get_player_info(players_data: list, players_names: dict) -> pd.DataFrame:
    """Function to loop over all players and save data in a pd.DataFrame

    Args:
        players_data (list): for every player a dictionary with info about the player
        except the player name
        players_names (dict): dictionary with player id as key and the player name as
        value

    Returns:
        pd.DataFrame: all information of the players
    """
    result_dict = {
        "id": [],
        "full_name": [],
        "formation_place": [],
        "position": [],
        "starter": [],
        "shirt_num": [],
    }

    for player in players_data:
        player_id = int(player["PlayerRef"][1:])
        result_dict["id"].append(player_id)
        result_dict["full_name"].append(players_names[str(player_id)])
        result_dict["formation_place"].append(int(player["Formation_Place"]))
        position = (
            player["Position"]
            if player["Position"] != "Substitute"
            else player["SubPosition"]
        )
        result_dict["position"].append(position.lower())
        result_dict["starter"].append(player["Status"] == "Start")
        result_dict["shirt_num"].append(int(player["ShirtNumber"]))

    return pd.DataFrame(result_dict)


def _load_event_data(f24_loc: str) -> pd.DataFrame:
    """Function to load the f27 .xml, the events of the match.
    Note: this function does ignore qualifiers for now

    Args:
        f24_loc (str): location of the f24.xml file

    Returns:
        pd.DataFrame: all events of the match in a pd dataframe
    """

    file = open(f24_loc, "r")
    lines = file.read()
    soup = BeautifulSoup(lines, "xml")

    result_dict = {
        "event_id": [],
        "type_id": [],
        "event": [],
        "period_id": [],
        "minutes": [],
        "seconds": [],
        "player_id": [],
        "team_id": [],
        "outcome": [],
        "start_x": [],
        "start_y": [],
        "datetime": [],
    }

    events = soup.find_all("Event")
    for event in events:
        result_dict["event_id"].append(int(event.attrs["id"]))
        event_type_id = int(event.attrs["type_id"])
        result_dict["type_id"].append(event_type_id)

        if event_type_id in EVENT_TYPE_IDS.keys():
            event_name = EVENT_TYPE_IDS[event_type_id].lower()
        else:
            # Unknown event
            event_name = None

        result_dict["event"].append(event_name)
        result_dict["period_id"].append(int(event.attrs["period_id"]))
        result_dict["minutes"].append(int(event.attrs["min"]))
        result_dict["seconds"].append(int(event.attrs["sec"]))

        if "player_id" in event.attrs.keys():
            result_dict["player_id"].append(int(event.attrs["player_id"]))
        else:
            result_dict["player_id"].append(np.nan)

        result_dict["team_id"].append(int(event.attrs["team_id"]))
        result_dict["outcome"].append(int(event.attrs["outcome"]))
        result_dict["start_x"].append(float(event.attrs["x"]))
        result_dict["start_y"].append(float(event.attrs["y"]))
        result_dict["datetime"].append(np.datetime64(event.attrs["timestamp"]))

    file.close()
    return pd.DataFrame(result_dict)
