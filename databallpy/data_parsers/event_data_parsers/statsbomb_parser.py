import json
import os

import numpy as np
import pandas as pd

from databallpy.data_parsers.metadata import Metadata
from databallpy.events import DribbleEvent, PassEvent, ShotEvent
from databallpy.utils.constants import MISSING_INT
from databallpy.utils.logging import create_logger

LOGGER = create_logger(__name__)

def load_statsbomb_event_data(
    events_loc: str, match_loc: str, lineup_loc: str,  pitch_dimensions: tuple = (120.0, 80.0)
) -> tuple[pd.DataFrame, Metadata, dict]:
    """This function retrieves the metadata and event data of a specific match. The x
    and y coordinates provided have been scaled to the dimensions of the pitch, with
    (0, 0) being the center. Additionally, the coordinates have been standardized so
    that the home team is represented as playing from left to right for the entire
    match, and the away team is represented as playing from right to left.

    Args:
        events_loc (str): location of the event.json file.
        match_loc (str): location of the match.json file.
        lineup_loc (str): location of the lineup.json file.
        pitch_dimensions (tuple, optional): the length and width of the pitch in meters

    Returns:
        Tuple[pd.DataFrame, Metadata, dict]: the event data of the match, the metadata,
        and the databallpy_events.
    """
    LOGGER.info(f"Loading Statsbomb event data: events_loc: {events_loc}")
    if not os.path.exists(events_loc):
        LOGGER.error(f"File {events_loc} does not exist.")
        raise FileNotFoundError(f"File {events_loc} does not exist.")
    
    if not os.path.exists(match_loc):
        LOGGER.error(f"File {match_loc} does not exist.")
        raise FileNotFoundError(f"File {match_loc} does not exist.")
    
    if not os.path.exists(lineup_loc):
        LOGGER.error(f"File {lineup_loc} does not exist.")
        raise FileNotFoundError(f"File {lineup_loc} does not exist.")

    if not isinstance(pitch_dimensions, (tuple, list)) or len(pitch_dimensions) != 2:
        LOGGER.error(
            f"Invalid pitch_dimensions: {pitch_dimensions}. "
            "Must be a tuple of length 2."
        )
        raise ValueError(
            f"Invalid pitch_dimensions: {pitch_dimensions}. "
            "Must be a tuple of length 2."
        )
    
    # Load the metadata
    metadata = _load_metadata(match_loc, lineup_loc, pitch_dimensions)
    LOGGER.info("Successfully loaded Statsbomb metadata.")
    # Load the event data
    event_data, databallpy_events, metadata = _load_event_data(events_loc, metadata, pitch_dimensions)
    
    LOGGER.info("Successfully loaded Statsbomb event data and databallpy events.")
    return event_data, metadata, databallpy_events

def _load_metadata(match_loc: str, lineup_loc: str, pitch_dimensions: tuple) -> Metadata:
    """Function to load metadata from the match.json and lineup.json files

    Args:
        match_loc (str): location of the match.json file
        lineup_loc (str): location of the lineup.json file
        pitch_dimensions (tuple): the length and width of the pitch in meters

    Returns:
        MetaData: all metadata information of the current match
    """
    with open(match_loc, "r", encoding="utf-8") as f:
        match_json = json.load(f)[0]

    with open(lineup_loc, "r", encoding="utf-8") as f:
        lineup_json= json.load(f)
    
    home_index = 0 if lineup_json[0]["team_id"] == match_json["home_team"]["home_team_id"] else 1
    away_index = 1 if home_index == 0 else 0

    home_players = _get_player_info(lineup_json[home_index]["lineup"])
    away_players = _get_player_info(lineup_json[away_index]["lineup"])

    periods = {
        "period_id": [1, 2, 3, 4, 5],
        "start_datetime_ed": [],
        "end_datetime_ed": [],
    }
    match_start = pd.to_datetime(match_json["match_date"]+" "+match_json["kick_off"], utc=True)
    for id in range(5):
        if id == 0:
            periods["start_datetime_ed"].append(match_start)
            periods["end_datetime_ed"].append(match_start + pd.to_timedelta(45, unit="minutes"))
        elif id == 1:
            periods["start_datetime_ed"].append(match_start+ pd.to_timedelta(60, unit="minutes"))
            periods["end_datetime_ed"].append(match_start + pd.to_timedelta(105, unit="minutes"))
        else:
            periods["start_datetime_ed"].append(pd.NaT)
            periods["end_datetime_ed"].append(pd.NaT)

    metadata = Metadata(
        match_id=match_json["match_id"],
        pitch_dimensions=pitch_dimensions,
        periods_frames=pd.DataFrame(periods),
        frame_rate=MISSING_INT,
        home_team_id=match_json["home_team"]["home_team_id"],
        home_team_name=match_json["home_team"]["home_team_name"],
        home_players=home_players,
        home_score=match_json["home_score"],
        home_formation="",
        away_team_id=match_json["away_team"]["away_team_id"],
        away_team_name=match_json["away_team"]["away_team_name"],
        away_players=away_players,
        away_score=match_json["away_score"],
        away_formation="",
        country=match_json["competition"]["country_name"],
    )
    return metadata


def _get_player_info(players_data: list) -> pd.DataFrame:
    """Function to loop over all players and save data in a pd.DataFrame

    Args:
        players_data (list): for every player a dictionary with info about the player

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

        if len(player["positions"])>0:
            if player["positions"][0]["from"] == "00:00":
                starter = True
            else:
                starter = False
            formation_place = player["positions"][0]["position_id"]
            position = player["positions"][0]["position"].lower()
        else:
            starter = False
            formation_place = MISSING_INT
            position = "did not play"


        result_dict["id"].append(player["player_id"])
        result_dict["full_name"].append(player["player_name"])
        result_dict["formation_place"].append(formation_place)
        result_dict["position"].append(position)
        result_dict["starter"].append(starter)
        result_dict["shirt_num"].append(player["jersey_number"])

    return pd.DataFrame(result_dict)

def _load_event_data(events_loc: str, metadata: Metadata, pitch_dimensions: tuple)-> tuple[pd.DataFrame, dict, Metadata]:
    """This function retrieves the event data of a specific match. The x
    and y coordinates provided have been scaled to the dimensions of the pitch, with
    (0, 0) being the center. Additionally, the coordinates have been standardized so
    that the home team is represented as playing from left to right for the entire
    match, and the away team is represented as playing from right to left.

    Args:
        events_loc (str): location of the events.json file
        metadata (Metadata): metadata of the match
        pitch_dimensions (tuple): dimensions of the pitch.

    Returns:
        Tuple[pd.DataFrame, dict, Metadata]: the event data of the match, the databallpy_events, and the updated metadata
    """
    with open(events_loc, "r", encoding="utf-8") as f:
        events_json = json.load(f)

    formations = {
        event["team"]["id"]: str(event["tactics"]["formation"])
        for event in events_json if event["type"]["id"] == 35
    }
    metadata.home_formation = formations[metadata.home_team_id]
    metadata.away_formation = formations[metadata.away_team_id]

    event_data = {
        "event_id": [],
        "databallpy_event": [],
        "period_id": [],
        "minutes": [],
        "seconds": [],
        "player_id": [],
        "team_id": [],
        "outcome": [],
        "start_x": [],
        "start_y": [],
        "datetime": [],
        "statsbomb_event": [],
        "statsbomb_event_id": [],
        "player_name": [],
        "team_name": [],
    }
    
    databallpy_mapping = {
        "Pass": "pass",
        "Shot": "shot",
        "Dribble": "dribble",
    }

    shot_events = {}
    pass_events = {}
    dribble_events = {}
    
    event_mask = [event["type"]["id"] not in [5, 18, 19, 26, 27, 34, 35, 36] for event in events_json]
    for id, event in enumerate(np.array(events_json)[event_mask]): 
        event_data["event_id"].append(id)
        event_data["period_id"].append(event["period"])
        event_data["minutes"].append(event["minute"])
        event_data["seconds"].append(event["second"])
        event_data["player_id"].append(event["player"]["id"])
        event_data["team_id"].append(event["possession_team"]["id"])
        if "location" in event.keys():
            event_data["start_x"].append(event["location"][0]-(pitch_dimensions[0]/2))
            event_data["start_y"].append(event["location"][1]-(pitch_dimensions[1]/2))
        else:
            event_data["start_x"].append(MISSING_INT)
            event_data["start_y"].append(MISSING_INT)
        event_data["datetime"].append(
            pd.to_datetime(metadata.periods_frames["start_datetime_ed"][event["period"]-1]+pd.to_timedelta(event["minute"]*60+event["second"], unit="seconds"))
        )
        event_data["statsbomb_event"].append(event["type"]["name"].lower().replace("*",""))
        event_data["statsbomb_event_id"].append(event["id"])
        event_data["player_name"].append(event["player"]["name"])
        event_data["team_name"].append(event["possession_team"]["name"])

        event_type_object = event["type"]["name"].lower().replace(" ", "_").replace("*", "").replace("-", "_").replace("/", "-")

        if event["type"]["name"] in databallpy_mapping:
            databallpy_event = databallpy_mapping[event["type"]["name"]]
            event_data["databallpy_event"].append(databallpy_event)
            try:
                event_data["outcome"].append(event[event_type_object]["outcome"]["name"])
            except:
                event_data["outcome"].append("")
            if databallpy_event == "shot": 
                shot_events[id] = _get_shot_event(event=event, id=event["id"], pitch_dimensions=pitch_dimensions, periods=metadata.periods_frames, away_team_id=metadata.away_team_id)
            elif databallpy_event == "pass":
                pass_events[id] = _get_pass_event(event=event, id=event["id"], pitch_dimensions=pitch_dimensions, periods=metadata.periods_frames, away_team_id=metadata.away_team_id)
            elif databallpy_event == "dribble":
                dribble_events[id] = _get_dribble_event(event=event, id=event["id"], pitch_dimensions=pitch_dimensions, periods=metadata.periods_frames, away_team_id=metadata.away_team_id)
        else:
            event_data["databallpy_event"].append(None)

            try:
                event_data["outcome"].append(event[event_type_object[event["type"]["id"]]]["outcome"]["name"])
            except:
                event_data["outcome"].append("")

    event_data = pd.DataFrame(event_data)

    # for statsbomb, top left is 0,0 so we need an extra flip around y-axis
    event_data.loc[
        event_data["team_id"] == metadata.away_team_id, ["start_x"]
    ] *= -1
    
    event_data.loc[
        event_data["team_id"] == metadata.home_team_id, ["start_y"]
    ] *= -1

    id_jersey_map = pd.concat([metadata.home_players, metadata.away_players])[["id", "shirt_num"]].set_index("id").to_dict()["shirt_num"]
    for event in {**shot_events, **pass_events, **dribble_events}.values():
        if event.team_side == "away":
            event.start_x *= -1
            if isinstance(event, PassEvent):
                event.end_x *= -1
        else:
            event.start_y *= -1
            if isinstance(event, PassEvent):
                event.end_y *= -1
        event.jersey = id_jersey_map[event.player_id]

    return event_data, {
        "shot_events": shot_events,
        "pass_events": pass_events,
        "dribble_events": dribble_events,
    }, metadata


POSSESSION_TYPE_MAPPING = {
    "Corner": "corner_kick",
    "Free Kick": "free_kick",
    "Open Play": "open_play",
    "Penalty": "penalty",
    "Kick Off": "kick_off",
    "Recovery": "unspecified",
    "Throw-in": "throw_in",
    "Interception": "unspecified",
    "Goal Kick": "goal_kick"
}

SET_PIECE_TYPE_MAPPING = {
    "Corner": "corner_kick",
    "Free Kick": "free_kick",
    "Open Play": "no_set_piece",
    "Penalty": "penalty",
    "Kick Off": "kick_off",
    "Throw-in": "throw_in",
    "Goal Kick": "goal_kick",
    "Recovery": "no_set_piece",
    "Interception": "no_set_piece", 
}

BODY_PART_MAPPING = {
    "Head": "head",
    "Left Foot": "left_foot",
    "Right Foot": "right_foot",
    "Keeper Arm": "other",
    "Drop Kick": "other",
    "No Touch": "other", 
    "Other": "other"
}


def _get_shot_event(event: dict, id: int, pitch_dimensions: tuple, periods: pd.DataFrame, away_team_id: int) -> ShotEvent:
    """This function retrieves the shot event of a specific match.

    Args:
        event (dict): the shot event.
        id (int): the id of the event.
        pitch_dimensions (tuple): pitch dimensions in x and y direction.
        periods (pd.Dataframe): metadata.periods_frames dataframe
        away_team_id (int): id of away team

    Returns:
        ShotEvent: the shot event
    """

    SHOT_OUTCOME_MAPPING = {
        "Blocked": "blocked",
        "Goal": "goal",
        "Off T": "miss_off_target",
        "Post": "miss_hit_post",
        "Saved": "miss",
        "Wayward": "miss",
        "Saved Off T": "miss",
        "Saved To Post": "miss_hit_post"
    }

    close_to_ball_event_info = _get_close_to_ball_event_info(event, id, pitch_dimensions, away_team_id, periods)

    return ShotEvent(
        **close_to_ball_event_info,
        related_event_id=event["related_events"],
        outcome=event["shot"]["outcome"]["name"] == "Goal",
        body_part=BODY_PART_MAPPING[event["shot"]["body_part"]["name"]],
        possession_type=POSSESSION_TYPE_MAPPING[event["shot"]["type"]["name"]],
        set_piece=SET_PIECE_TYPE_MAPPING[event["shot"]["type"]["name"]],
        _xt=-1.0,
        outcome_str=SHOT_OUTCOME_MAPPING[event["shot"]["outcome"]["name"]],
    )

def _get_pass_event(event: dict, id: int, pitch_dimensions: tuple, periods: pd.DataFrame, away_team_id: int) -> PassEvent:
    """This function retrieves the pass event of a specific match.

    Args:
        event (dict): the shot event.
        id (int): the id of the event.
        pitch_dimensions (tuple): pitch dimensions in x and y direction.
        periods (pd.Dataframe): metadata.periods_frames dataframe
        away_team_id (int): id of away team

    Returns:
        PassEvent: the pass event
    """

    PASS_OUTCOME_MAPPING = {
        "": "",
    }    

    PASS_TYPE_MAPPING = {
        "Inswinging": "unspecified",
        "Outswinging": "unspecified",
        "Straight": "unspecified",
        "Through Ball": "through_ball"
    }

    if "technique" in event["pass"].keys():
        pass_type = PASS_TYPE_MAPPING[event["pass"]["technique"]["name"]]
    else:
        pass_type = "unspecified"

    if "type" in event["pass"].keys():
        possession_type = POSSESSION_TYPE_MAPPING[event["pass"]["type"]["name"]]
        set_piece = SET_PIECE_TYPE_MAPPING[event["pass"]["type"]["name"]]
    else:
        possession_type = "unspecified"
        set_piece = "unspecified"

    if "body_type" in event["pass"].keys():
        body_part = BODY_PART_MAPPING[event["pass"]["body_part"]["name"]]
    else:
        body_part = "unspecified"

    if "recipient" in event["pass"].keys():
        receiver_player_id = event["pass"]["recipient"]["id"]
    else:
        receiver_player_id = MISSING_INT

    if "related_events" in event["pass"].keys():
        related_events = event["pass"]["related_events"]
    else:
        related_events = None

    close_to_ball_event_info = _get_close_to_ball_event_info(event, id, pitch_dimensions, away_team_id, periods)

    return PassEvent(
        **close_to_ball_event_info,
        related_event_id= related_events,
        end_x = event["pass"]["end_location"][0]-(pitch_dimensions[0]/2),
        end_y = event["pass"]["end_location"][1]-(pitch_dimensions[1]/2),
        outcome=False if "goal-assist" not in event["pass"].keys() else True,
        body_part=body_part,
        possession_type=possession_type,
        set_piece=set_piece,
        receiver_player_id=receiver_player_id,
        _xt=-1.0,
        outcome_str= "unspecified",
        pass_type=pass_type
    )

def _get_dribble_event(event: dict, id: int, pitch_dimensions: tuple, periods: pd.DataFrame, away_team_id: int) -> DribbleEvent:
    """This function retrieves the dribble event

    Args:
        event (dict): the shot event.
        id (int): the id of the event.
        pitch_dimensions (tuple): pitch dimensions in x and y direction.
        periods (pd.Dataframe): metadata.periods_frames dataframe
        away_team_id (int): id of away team

    Returns:
        DribbleEvent: the dribble event
    """
    if "related_events" in event["dribble"].keys():
        related_events = event["dribble"]["related_events"]
    else:
        related_events = None

    close_to_ball_event_info = _get_close_to_ball_event_info(event, id, pitch_dimensions, away_team_id, periods)

    return DribbleEvent(
        **close_to_ball_event_info,
        related_event_id= related_events,
        outcome=event["dribble"]["outcome"]["name"] == "Complete",
        _xt=-1.0,
        body_part="unspecified",
        possession_type="unspecified",
        set_piece="unspecified",
        duel_type="unspecified",
        with_opponent=None
    )


def _get_close_to_ball_event_info(
    event: dict,
    id: int,
    pitch_dimensions: tuple,
    away_team_id: int,
    periods: pd.DataFrame,
) -> dict:
    """Function to get the base event data from the event based on
    the CloseToBallEvent class.

    Args:
        event (dict): event 
        id (int): index
        pitch_dimensions (tuple): pitch dimensions in x and y direction.
        away_team_id (int): id of the away team
        players (pd.DataFrame): dataframe with player information.
        periods: metadata.periods_frames dataframe

    Returns:
        dict: dictionary with the base event data: start_x, start_y, related_event_id
    """

    return {
        "start_x": event["location"][0]-(pitch_dimensions[0]/2),
        "start_y": event["location"][1]-(pitch_dimensions[1]/2),
        "event_id": id,
        "period_id": event["period"],
        "minutes": event["minute"],
        "seconds": event["second"],
        "datetime": pd.to_datetime(periods["start_datetime_ed"][event["period"]-1]+pd.to_timedelta(event["minute"]*60+event["second"], unit="seconds")),
        "team_id": event["team"]["id"],
        "team_side": "away" if event["team"]["id"] == away_team_id else "home",
        "pitch_size": pitch_dimensions,
        "player_id": event["player"]["id"],
        "jersey": MISSING_INT
    }


