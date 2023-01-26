from typing import Tuple
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
# from databallpy.load_data.metadata.py import MetaData

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

def load_opta_event_data(f7_loc:str, f24_loc:str) -> Tuple[pd.Series, pd.DataFrame]:
    
    assert f7_loc[-4:] == ".xml", "f7 opta file should by of .xml format!"
    assert f24_loc[-4:] == ".xml", "f24 opta file should be of .xml format!"

    # metadata = _load_metadata(f7_loc)
    event_data = _load_event_data(f24_loc)
    return metadata, event_data

def _load_metadata(f7_loc:str):
    """Function to load metadata from the f7.xml opta file

    Args:
        f7_loc (str): location of the f7.xml opta file

    Returns:
        MetaData: all metadata information of the current match
    """
    file = open(f7_loc, "r").read()
    soup = BeautifulSoup(file, "xml")
    
    # opta has a TeamData and Team attribute with information...
    team_datas = soup.find_all("TeamData")
    teams = soup.find_all("Team")
    teams_info = {}
    teams_player_info = {}
    for team_data, team in zip(team_datas, teams):
        
        # team information
        team_name = team.findChildren("Name")[0].contents[0]
        team_info = {}
        team_info["team_name"] = team_name
        team_info["side"] = team["Side"].lower()
        team_info["formation"] = team["Formation"]
        team_info["score"] = int(team["Score"])
        team_info["team_id"] = int(team["TeamRef"][1:])
        teams_info[team_info["side"]] = team_info
        
        # player information
        player_data = [player.attrs for player in team_data.findChildren("MatchPlayer")]
        player_names = {}
        for player in team.findChildren("Player"):
            player_id = int(player.attrs["uID"][1:])
            first_name = player.contents[1].contents[1].contents[0]
            last_name = player.contents[1].contents[3].contents[0]
            player_names[str(player_id)] = f"{first_name} {last_name}"
        
        player_info = _get_player_info(player_data, player_names)
        teams_player_info[team_info["side"]] = player_info

    metadata = MetaData(teams_info, teams_player_info)
    return metadata

def _get_player_info(player_data:list, player_names:dict) -> pd.DataFrame:
    """Function to loop over all players and save data in a pd.DataFrame

    Args:
        player_data (list): for every player a dictionary with info about the player, except the player name
        player_names (dict): dictionary with player id as key and the player name as value

    Returns:
        pd.DataFrame: all information of the players
    """
    result_dict = {"player_id": [], "player_name": [], "formation_place": [], "position": [], "starter":[], "shirt_number":[]}
    for player in player_data:
        player_id = int(player["PlayerRef"][1:])
        result_dict["player_id"].append(player_id)
        result_dict["player_name"].append(player_names[str(player_id)])
        result_dict["formation_place"].append(int(player["Formation_Place"]))
        position = player["Position"] if player["Position"] != "Substitute" else player["SubPosition"]
        result_dict["position"].append(position.lower())
        result_dict["starter"].append(player["Status"] == "Start")
        result_dict["shirt_number"].append(player["ShirtNumber"])
    
    return pd.DataFrame(result_dict)


def _load_event_data(f24_loc:str) -> pd.DataFrame:
    """Function to load the f27 .xml, the events of the match. 
    Note: this function does ignore qualifiers for now

    Args:
        f24_loc (str): location of the f24.xml file

    Returns:
        pd.DataFrame: all events of the match in a pd dataframe
    """

    file = open(f24_loc, "r").read()
    soup = BeautifulSoup(file, "xml")

    result_dict = {
        "event_id": [], 
        "type_id": [], 
        "event": [],
        "period_id": [], 
        "minutes": [], 
        "seconds": [] ,  
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
            # unknown event
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
        date = np.datetime64(event.attrs["timestamp"])
        result_dict["datetime"].append(date)

    return pd.DataFrame(result_dict)
