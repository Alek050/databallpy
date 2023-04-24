import json
import pandas as pd
import numpy as np
from databallpy.load_data.metadata import Metadata
import datetime as dt

def load_instat_event_data(event_data_loc:str, metadata_loc:str) -> pd.DataFrame:
    """_summary_

    Args:
        event_data_loc (str): _description_
        event_data_metadata_loc (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    assert isinstance(event_data_loc, str), f"event_data_loc should be a string, not a {type(event_data_loc)}"
    assert isinstance(
        metadata_loc, str
    ), f"event_metadata_loc should be a string, not a {type(metadata_loc)}"
    assert event_data_loc[-5:] == ".json", "instat event file should by of .json format"
    assert metadata_loc[-5:] == ".json", "instat event metadata file should be of .json format"

    metadata = _load_metadata(metadata_loc=metadata_loc)
    metadata = _update_metadata(metadata=metadata, event_data_loc=event_data_loc)
    event_data, pitch_dimensions = _load_event_data(event_data_loc=event_data_loc, metadata=metadata)
    metadata.pitch_dimensions = pitch_dimensions

    return event_data, metadata

def _load_metadata(metadata_loc:str)->pd.DataFrame:
    """_summary_

    Args:
        metdata_loc (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    with open(metadata_loc, 'r') as f:
        data=f.read()
    metadata_json = json.loads(data)
    match_info = metadata_json["data"]["match_info"][0]

    periods = {
        "period": [1, 2, 3, 4, 5],
        "start_datetime_ed": [pd.to_datetime("NaT")]*5,
        "end_datetime_ed": [pd.to_datetime("NaT")]*5,
    }
    periods["start_datetime_ed"][0] = pd.to_datetime(match_info["match_date"]) - dt.timedelta(hours=2)

    metadata = Metadata(
        match_id=int(match_info["id"]),
        pitch_dimensions=[np.nan, np.nan],
        periods_frames=pd.DataFrame(periods),
        frame_rate=np.nan,
        home_team_id=int(match_info["team1_id"]),
        home_team_name=str(match_info["team1_name"]),
        home_players=pd.DataFrame(columns=["id", "full_name", "shirt_num"]),
        home_score=int(match_info["score"].split(":")[0]),
        home_formation="",
        away_team_id=int(match_info["team2_id"]),
        away_team_name=str(match_info["team2_name"]),
        away_players=pd.DataFrame(columns=["id", "full_name", "shirt_num"]),
        away_score=int(match_info["score"].split(":")[1]),
        away_formation="",
    )
    
    return metadata

def _update_metadata(metadata:Metadata, event_data_loc:str) -> pd.DataFrame:
    """

    Args:
        metadata (Metadata): _description_
        event_data_loc (str): _description_

    Returns:
        pd.DataFrame: _description_
    """

    with open(event_data_loc, 'r') as f:
        data=f.read()
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
        if event["action_id"].startswith("16"):
            if int(event["player_id"]) not in players_dict["id"]:
                players_dict["id"].append(int(event["player_id"]))
                players_dict["full_name"].append(str(event["player_name"]))
                players_dict["position"].append(str(event["position_name"]))
                if str(event["position_name"]) == "Substitute player":
                    players_dict["starter"].append(False)
                else:
                    players_dict["starter"].append(True)
                players_dict["shirt_num"].append(int(event["number"]))
                players_dict["team_id"].append(int(event["team_id"]))
        if event["action_id"].startswith("15"):
            if int(event["team_id"]) == metadata.home_team_id and home_formation == "":
                    home_formation = str(event["action_name"]).replace("-", "")
            if int(event["team_id"]) == metadata.away_team_id and away_formation == "":
                    away_formation = str(event["action_name"]).replace("-", "")
                
        
    df_players = pd.DataFrame(players_dict)
    home_players = df_players[df_players["team_id"] == metadata.home_team_id]
    home_players = home_players.sort_values("starter", ascending=False).drop("team_id", axis=1).reset_index(drop=True)
    away_players = df_players[df_players["team_id"] == metadata.away_team_id]
    away_players = away_players.sort_values("starter", ascending=False).drop("team_id", axis=1).reset_index(drop=True)

    metadata.home_formation = home_formation
    metadata.away_formation = away_formation
    metadata.home_players = home_players
    metadata.away_players = away_players
    #import pdb;pdb.set_trace()  
    return metadata


def _load_event_data(event_data_loc:str, metadata:Metadata)->pd.DataFrame:
    """_summary_

    Args:
        event_data_loc (str): _description_
        metadata(Metadata)

    Returns:
        pd.DataFrame: _description_
    """
    EVENT_AND_OUTCOME_INSTAT_EVENTS = {
        "Attacking pass accurate": ["pass", 1],
        "Accurate key pass": ["pass", 1],
        "Attacking pass inaccurate": ["pass", 0],
        "Inaccurate key pass": ["pass", 0],
        "Pass into offside": ["pass", 0],
        "Successful dribbling": ["dribbling", 1],
        "Unsuccessful dribbling": ["dribbling", 0],
        "Dribbling": ["dribbling", np.nan],
        "Crosses accurate": ["cross", 1],
        "Accurate crossing from set piece with a shot": ["cross", 1],
        "Accurate crossing from set piece": ["cross", 1], 
        "Accurate crossing from set piece with a goal": ["cross", 1],     
    	"Crosses inaccurate": ["cross", 0],
        "Cross interception": ["cross", 0],
        "Inaccurate blocked cross": ["cross", 0],
        "Inaccurate set-piece cross": ["cross", 0],
        "Shot on target": ["shot", 1],
        "Blocked shot": ["shot", 0],
        "Wide shot": ["shot", 0],
    }

    with open(event_data_loc, 'r') as f:
        data=f.read()
    event_data_json = json.loads(data)
    events = event_data_json["data"]["row"]

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
        "instat_event": [],
    }

    start_time_period = {
        1: metadata.periods_frames.loc[0,"start_datetime_ed"],
        2: metadata.periods_frames.loc[0,"start_datetime_ed"] + dt.timedelta(minutes=60),
        3: metadata.periods_frames.loc[0,"start_datetime_ed"] + dt.timedelta(minutes=110),
        4: metadata.periods_frames.loc[0,"start_datetime_ed"] + dt.timedelta(minutes=130),
        5: metadata.periods_frames.loc[0,"start_datetime_ed"] + dt.timedelta(minutes=150)
    }

    for event in events:
        #import pdb;pdb.set_trace()
        if not event["action_id"].startswith(("16", "15")):
            result_dict["event_id"].append(int(event["id"]))
            result_dict["type_id"].append(int(event["action_id"]))
            if str(event["action_name"]) in EVENT_AND_OUTCOME_INSTAT_EVENTS.keys():
                event_name, outcome = EVENT_AND_OUTCOME_INSTAT_EVENTS[str(event["action_name"])]
                result_dict["event"].append(event_name)
                result_dict["outcome"].append(outcome)
            else:
                result_dict["event"].append(str(event["action_name"]))
                result_dict["outcome"].append(np.nan)
            result_dict["period_id"].append(int(event["half"]))
            result_dict["minutes"].append(float(event["second"])//60)
            result_dict["seconds"].append(float(event["second"])%60)
            if "player_id" in event.keys():
                result_dict["player_id"].append(int(event["player_id"]))
            else:
                result_dict["player_id"].append(-999)
            if "team_id" in event.keys():
                result_dict["team_id"].append(int(event["team_id"]))
            else:
                result_dict["team_id"].append(-999)
            if "pos_x" in event.keys():
                result_dict["start_x"].append(float(event["pos_x"]))
                result_dict["start_y"].append(float(event["pos_y"]))
            else:
                result_dict["start_x"].append(np.nan)
                result_dict["start_y"].append(np.nan)
            result_dict["datetime"].append(
                start_time_period[int(event["half"])]
                + dt.timedelta(milliseconds=float(event["second"])*1000)
            )
            result_dict["instat_event"].append(str(event["action_name"]))
            

    event_data = pd.DataFrame(result_dict)
    start_events = ["pass", "shot", "Goal"]
    x_start, y_start = event_data[event_data["event"].isin(start_events)].reset_index().loc[0, ["start_x", "start_y"]]
    event_data["start_x"] -= x_start
    event_data["start_y"] -= y_start
    pitch_dimensions = [2*x_start, 2*y_start]

    id_full_name_dict = dict(zip(metadata.home_players["id"], metadata.home_players["full_name"]))
    id_full_name_dict_away = dict(zip(metadata.away_players["id"], metadata.away_players["full_name"]))
    id_full_name_dict.update(id_full_name_dict_away)
    event_data["player_name"] = event_data["player_id"].map(id_full_name_dict)
    away_mask = event_data["team_id"] == metadata.away_team_id
    event_data.loc[away_mask, ["start_x", "start_y"]] *= -1

    return event_data, pitch_dimensions