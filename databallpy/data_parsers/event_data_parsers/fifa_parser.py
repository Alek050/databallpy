from ast import In
import json
import numpy as np
import pandas as pd

from databallpy.data_parsers import Metadata
from databallpy.events import (
    IndividualCloseToBallEvent,
    PassEvent,
    ShotEvent,
    TackleEvent,
)
from databallpy.utils.constants import MISSING_INT
from databallpy.utils.logging import logging_wrapper


# FIFA event type mappings to databallpy events
FIFA_TO_DATABALLPY_MAP = {
    "pass": "pass",
    "assist": "pass",
    "cross": "pass",
    "attempt_at_goal": "shot",
    "own_goal": "own_goal",
    "tackle": "tackle",
    "no_event": "tackle",
}

# Shot outcome mappings
SHOT_OUTCOMES = {
    "incomplete": "miss_off_target",
    "off_target": "miss_off_target",
    "on_target": "on_target",
    "complete": "goal",
    "own_goal": "own_goal"
}

# Body part mappings
BODY_PART_MAP = {
    "right_foot": "right_foot",
    "left_foot": "left_foot",
    "head": "head",
    "hands": "hands",
    "body": "other",
    "feet": "other",
}

# Set piece mappings
SET_PIECE_MAP = {
    "freekick": "free_kick",
    "throwin": "throw_in",
    "goalkick": "goal_kick",
    "corner": "corner_kick",
    "kickoff": "kick_off",
    "penalty": "penalty",
}

# Phase of the game mappings
PHASE_TYPE_MAP = {
    "referee_action": "referee_action",
    "in_possession": "in_possession",
    "out_of_possession": "out_of_possession",
    "in_contest": "in_contest",
}


@logging_wrapper(__file__)
def load_fifa_event_data(
    metadata_loc: str, 
    events_loc: str, 
    pitch_dimensions: list = [105.0, 68.0]
) -> tuple[
    pd.DataFrame, Metadata, dict[str, dict[str | int, IndividualCloseToBallEvent]]
]:
    """This function retrieves the metadata and event data of a FIFA match. The x
    and y coordinates provided have been scaled to the dimensions of the pitch, with
    (0, 0) being the center. Additionally, the coordinates have been standardized so
    that the home team is represented as playing from left to right for the entire
    game, and the away team is represented as playing from right to left.

    Args:
        metadata_loc (str): location of the metadata JSON file.
        events_loc (str): location of the events JSON file.
        pitch_dimensions (list, optional): the length and width of the pitch in meters

    Returns:
        Tuple[pd.DataFrame, Metadata, dict]: the event data of the game, the metadata,
        and the databallpy_events.
    """
    if not isinstance(metadata_loc, str):
        raise TypeError(f"metadata_loc should be a string, not a {type(metadata_loc)}")
    if not isinstance(events_loc, str):
        raise TypeError(f"events_loc should be a string, not a {type(events_loc)}")
    if not metadata_loc[-5:] == ".json":
        raise ValueError(
            f"metadata file should be of .json format, not {metadata_loc.split('.')[-1]}"
        )
    if not events_loc[-5:] == ".json":
        raise ValueError(
            f"events file should be of .json format, not {events_loc.split('.')[-1]}"
        )

    metadata = _load_metadata(metadata_loc, pitch_dimensions=pitch_dimensions)
    all_players = pd.concat(
        [metadata.home_players, metadata.away_players], ignore_index=True
    )
    kickoff_time = metadata.kickoff_utc

    event_data, databallpy_events = _load_event_data(
        events_loc,
        metadata.home_team_id,
        metadata.away_team_id,
        pitch_dimensions=pitch_dimensions,
        players=all_players,
        kickoff_time=kickoff_time,
    )

    home_score, away_score = _get_game_score(
        event_data, 
        metadata.home_team_id, 
        metadata.away_team_id
    )

    metadata.home_score = home_score
    metadata.away_score = away_score

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

    event_data.insert(6, "player_name", None)
    event_data.loc[home_mask, "player_name"] = event_data.loc[
        home_mask, "player_id"
    ].map(home_players)
    event_data.loc[away_mask, "player_name"] = event_data.loc[
        away_mask, "player_id"
    ].map(away_players)
    event_data["player_name"] = event_data["player_name"].replace({np.nan: None})

    # Rescale the x and y coordinates relative to the pitch dimensions
    # The original dimension of the x and y coordinates range from 0 to 1
    event_data.loc[:, ["start_x"]] = (
        event_data.loc[:, ["start_x"]] * pitch_dimensions[0]
    ) - (pitch_dimensions[0] / 2.0)
    event_data.loc[:, ["start_y"]] = (
        event_data.loc[:, ["start_y"]] * pitch_dimensions[1]
    ) - (pitch_dimensions[1] / 2.0)

    # Change direction of play of the away team so it is represented from right to left
    event_data.loc[
        event_data["team_id"] == metadata.away_team_id, ["start_x", "start_y"]
    ] *= -1

    return event_data, metadata, databallpy_events


@logging_wrapper(__file__)
def _load_metadata(metadata_loc: str, pitch_dimensions: list) -> Metadata:
    """Function to load metadata from the FIFA metadata JSON file

    Args:
        metadata_loc (str): location of the metadata JSON file
        pitch_dimensions (list): the length and width of the pitch in meters

    Returns:
        Metadata: all metadata information of the current game
    """
    with open(metadata_loc, "r", encoding="utf-8") as f:
        metadata_json = json.load(f)

    # Get match information
    match_id = metadata_json["match_id"]
    country = metadata_json.get("country", "UNKNOWN")
    
    # Parse kickoff time
    kickoff_time = pd.to_datetime(metadata_json["kickoff_utc"], utc=True)
    
    # Create periods dataframe from phases
    periods = {
        "period_id": [1, 2, 3, 4, 5],
        "start_datetime_ed": [],
        "end_datetime_ed": [],
    }
    
    for phase in metadata_json["phases"][:2]:
        start_time = kickoff_time + pd.to_timedelta(phase["phase_start"])
        end_time = kickoff_time + pd.to_timedelta(phase["phase_end"])
        periods["start_datetime_ed"].append(start_time)
        periods["end_datetime_ed"].append(end_time)
    
    # Fill remaining periods with NaT
    for _ in range(3):
        periods["start_datetime_ed"].append(pd.to_datetime("NaT", utc=True))
        periods["end_datetime_ed"].append(pd.to_datetime("NaT", utc=True))

    # Get player information
    home_players = _get_player_info(metadata_json["home_team_players"])
    away_players = _get_player_info(metadata_json["away_team_players"])

    # Get game scores
    home_team_id = metadata_json["home_team_id"]
    away_team_id = metadata_json["away_team_id"]
    home_score, away_score = np.nan, np.nan

    # Get team formation
    home_formation = metadata_json["home_formation"]
    away_formation = metadata_json["away_formation"]
    
    metadata = Metadata(
        game_id=match_id,
        pitch_dimensions=pitch_dimensions,
        periods_frames=pd.DataFrame(periods),
        frame_rate=25,
        home_team_id=home_team_id,
        home_team_name=metadata_json["home_team_name"],
        home_players=home_players,
        home_score=home_score,  
        home_formation=home_formation,
        away_team_id=away_team_id,
        away_team_name=metadata_json["away_team_name"],
        away_players=away_players,
        away_score=away_score,  
        away_formation=away_formation,
        country=country,
    )
    metadata.kickoff_utc = kickoff_time
    return metadata


def _get_player_info(players_data: list) -> pd.DataFrame:
    """Function to loop over all players and save data in a pd.DataFrame

    Args:
        players_data (list): for every player a dictionary with info about the player

    Returns:
        pd.DataFrame: all information of the players
    """
    n = len(players_data)
    result_dict = {
        "id": [MISSING_INT] * n,
        "full_name": [""] * n,
        "shirt_num": [MISSING_INT] * n,
        "position": ["unspecified"] * n,
    }

    for idx, player in enumerate(players_data):
        result_dict["id"][idx] = player["player_id"]
        result_dict["full_name"][idx] = player["player_name"]
        result_dict["shirt_num"][idx] = player["player_shirt_number"]

    return pd.DataFrame(result_dict)


@logging_wrapper(__file__)
def _load_event_data(
    events_loc: str,
    home_team_id: int,
    away_team_id: int,
    players: pd.DataFrame,
    pitch_dimensions: list = [105.0, 68.0],
    kickoff_time: pd.Timestamp = None,
) -> tuple[pd.DataFrame, dict[str, dict[str | int, IndividualCloseToBallEvent]]]:
    """Function to load FIFA event data from JSON file

    Args:
        events_loc (str): location of the events JSON file
        home_team_id (int): id of the home team
        away_team_id (int): id of the away team
        players (pd.DataFrame): dataframe with player information
        pitch_dimensions (list, optional): dimensions of the pitch.
            Defaults to [105.0, 68.0].
        kickoff_time (pd.Timestamp): kickoff time of the match

    Returns:
        pd.DataFrame: all events of the game in a pd dataframe
        dict: dict with "shot_events", "dribble_events", "pass_events", "other_events"
             as key and a dict with the IndividualCloseToBallEvent instances
    """
    offer_events = {}
    shot_events = {}
    pass_events = {}
    other_events = {}

    # Load JSON file
    events_list = []
    with open(events_loc, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events_list.append(json.loads(line))

    flip_first_half, flip_second_half = _determine_period_flips(
        events_list, home_team_id, away_team_id
    )

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
        "datetime": [],
        "phase": [],
        "original_event_id": [],
        "original_event": [],
        "original_event_type": [],
        "outcome_additional": [], 
    }

    for i_event, event in enumerate(events_list):
        # Skip referee actions and other non-player events
        if event["team_id"] == 0 or event["from_player_id"] == 0:
            continue

        result_dict["event_id"].append(i_event)
        result_dict["original_event_id"].append(event["event_id"])
        result_dict["original_event_type"].append(event["event_type"])
        result_dict["phase"].append(event["category"])

        event_name = event["event"]
        result_dict["original_event"].append(event_name)
        result_dict["period_id"].append(event["half_time"])
        
        # Convert time from match_time_in_ms
        total_seconds = event["match_time_in_ms"] / 1000.0
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        result_dict["minutes"].append(minutes)
        result_dict["seconds"].append(seconds)
        
        if kickoff_time is not None:
            event_datetime = kickoff_time + pd.to_timedelta(total_seconds, unit='s')
        else:
            event_datetime = pd.NaT
        result_dict["datetime"].append(event_datetime)
        
        result_dict["player_id"].append(
            event["from_player_id"] if event["from_player_id"] != 0 else MISSING_INT
        )
        result_dict["team_id"].append(event["team_id"])
        result_dict["outcome_additional"].append(event.get("outcome_additional", ""))
        
        # Determine success based on outcome
        if event_name in ["pass", "tackle"]:
            outcome = (event.get("outcome") or "").lower()
            result_dict["is_successful"].append(
                1 if ("possession_complete" in outcome or "possession_won" in outcome) else 0
            )
        else:
            result_dict["is_successful"].append(None)

        x_norm = event.get("x_location_start") if "x_location_start" in event else event.get("x")
        y_norm = event.get("y_location_start") if "y_location_start" in event else event.get("y")
        
        if x_norm is None or pd.isna(x_norm):
            x_norm = 0.5
        if y_norm is None or pd.isna(y_norm):
            y_norm = 0.5
            
        x_norm = max(0.0, min(1.0, float(x_norm)))
        y_norm = max(0.0, min(1.0, float(y_norm)))

        period_id = event.get("half_time")
        
        if period_id == 1 and flip_first_half:
            x_norm = 1.0 - x_norm
            y_norm = 1.0 - y_norm
        elif period_id == 2 and flip_second_half:
            x_norm = 1.0 - x_norm
            y_norm = 1.0 - y_norm

        result_dict["start_x"].append(x_norm)
        result_dict["start_y"].append(y_norm)

        # Create databallpy event instances
        if event_name in ["pass", "assist"]:
            pass_events[i_event] = _make_pass_instance(
                event,
                home_team_id,
                away_team_id,
                pitch_dimensions=pitch_dimensions,
                players=players,
                id=i_event,
                period_id=period_id,
                flip_first_half=flip_first_half,
                flip_second_half=flip_second_half,
            )

        if event_name in ["attempt_at_goal", "own_goal"]:
            shot_events[i_event] = _make_shot_event_instance(
                event,
                home_team_id,
                away_team_id,
                pitch_dimensions=pitch_dimensions,
                players=players,
                id=i_event,
                period_id=period_id,
                flip_first_half=flip_first_half,
                flip_second_half=flip_second_half,
            )

        if event_name == "tackle":
            other_events[i_event] = _make_tackle_event_instance(
                event,
                home_team_id,
                away_team_id,
                pitch_dimensions=pitch_dimensions,
                players=players,
                id=i_event,
                period_id=period_id,
                flip_first_half=flip_first_half,
                flip_second_half=flip_second_half,
            )

    result_dict["databallpy_event"] = [None] * len(result_dict["event_id"])
    event_data = pd.DataFrame(result_dict)
    event_data["databallpy_event"] = (
        event_data["original_event"]
        .map(FIFA_TO_DATABALLPY_MAP)
        .replace([np.nan], [None])
    )

    # Handle shot success
    event_data.loc[
        event_data["original_event"] == "attempt_at_goal", "is_successful"
    ] = event_data.loc[
        event_data["original_event"] == "attempt_at_goal"
    ].apply(
        lambda row: 1 if str(row.get("outcome_additional", "")).lower() == "goal" else 0,
        axis=1
    )

    # Ensure boolean dtype
    event_data["is_successful"] = event_data["is_successful"].astype("boolean")
    event_data.loc[event_data["period_id"] > 5, "period_id"] = -1
    event_data = event_data.drop(columns=["outcome_additional"]) 
    
    return event_data, {
        "shot_events": shot_events,
        "pass_events": pass_events,
        "offer_events": offer_events,
        "other_events": other_events,
    }


def _make_pass_instance(
    event: dict,
    home_team_id: int,
    away_team_id: int,
    players: pd.DataFrame,
    pitch_dimensions: list = [105.0, 68.0],
    id: int = None,
    period_id: int = None,
    flip_first_half: bool = False,
    flip_second_half: bool = False,
) -> PassEvent:
    """Function to create a PassEvent instance from FIFA event data"""
    on_ball_info = _get_on_ball_event_info(event)
    on_ball_info.update(
        _get_close_to_ball_event_info(
            event, pitch_dimensions, home_team_id, away_team_id, players,
            id, period_id, flip_first_half, flip_second_half
        )
    )

    outcome_str = "successful" if event.get("outcome") == "possession_complete" else "unsuccessful"
    
    # Get pass_type
    line_break = event.get("line_break_direction")
    if line_break in ["around", "over", "through"]:
        pass_type = "line_break"
    elif line_break is not None:
        pass_type = str(line_break)
    else:
        pass_type = "unspecified"

    # Get end coordinates
    x_end_norm = event.get("x_location_end")
    y_end_norm = event.get("y_location_end")
    
    if x_end_norm is not None and y_end_norm is not None:
        if period_id == 1 and flip_first_half:
            x_end_norm = 1.0 - x_end_norm
            y_end_norm = 1.0 - y_end_norm
        elif period_id == 2 and flip_second_half:
            x_end_norm = 1.0 - x_end_norm
            y_end_norm = 1.0 - y_end_norm
        
        x_end = (x_end_norm * pitch_dimensions[0]) - (pitch_dimensions[0] / 2.0)
        y_end = (y_end_norm * pitch_dimensions[1]) - (pitch_dimensions[1] / 2.0)
        
        if event.get("team_id") == away_team_id:
            x_end *= -1
            y_end *= -1
    else:
        x_end, y_end = np.nan, np.nan

    return PassEvent(
        **on_ball_info,
        _xt=-1.0,
        outcome_str=outcome_str,
        end_x=x_end,
        end_y=y_end,
        pass_type=pass_type,
    )


def _make_shot_event_instance(
    event: dict,
    home_team_id: int,
    away_team_id: int,
    players: pd.DataFrame,
    pitch_dimensions: list = [105.0, 68.0],
    id: int = None,
    period_id: int = None,
    flip_first_half: bool = False,
    flip_second_half: bool = False,
) -> ShotEvent:
    """Function to create a ShotEvent instance from FIFA event data"""
    on_ball_info = _get_on_ball_event_info(event)
    on_ball_info.update(
        _get_close_to_ball_event_info(
            event, pitch_dimensions, home_team_id, away_team_id, players,
            id, period_id, flip_first_half, flip_second_half
        )
    )
    on_ball_info.pop("outcome")

    event_name = (event.get("event") or "").lower()
    outcome_additional = (event.get("outcome_additional") or "").lower()
    
    if event_name == "own_goal":
        shot_outcome = "own_goal"
    elif "goal" in outcome_additional:
        shot_outcome = "goal"
    else:
        shot_outcome = "miss_off_target"

    return ShotEvent(
        **on_ball_info,
        _xt=-1.0,
        outcome=shot_outcome == "goal",
        outcome_str=shot_outcome,
    )


def _make_tackle_event_instance(
    event: dict,
    home_team_id: int,
    away_team_id: int,
    players: pd.DataFrame,
    pitch_dimensions: list = [105.0, 68.0],
    id: int = None,
    period_id: int = None,
    flip_first_half: bool = False,
    flip_second_half: bool = False,
) -> TackleEvent:
    """Function to create a TackleEvent instance from FIFA event data"""
    close_to_ball_info = _get_close_to_ball_event_info(
        event, pitch_dimensions, home_team_id, away_team_id, players,
        id, period_id, flip_first_half, flip_second_half
    )
    return TackleEvent(**close_to_ball_info)


def _get_on_ball_event_info(event: dict) -> dict:
    """Function to get the on-ball event data from the event based on
    the IndividualOnBallEvent class.

    Args:
        event (dict): event from FIFA data

    Returns:
        dict: dictionary with body_part, set_piece, and possession_type
    """
    # Get body part
    body_part = BODY_PART_MAP.get(event.get("body_type", "other"), "unspecified")
    
    # Get set piece
    origin = event.get("origin", "")
    set_piece = SET_PIECE_MAP.get(origin, "no_set_piece")
    
    possession_type = "open_play"
    
    return {
        "body_part": body_part,
        "set_piece": set_piece,
        "possession_type": possession_type,
    }


def _get_close_to_ball_event_info(
    event: dict,
    pitch_dimensions: list,
    home_team_id: int,
    away_team_id: int,
    players: pd.DataFrame,
    id: int,
    period_id: int,
    flip_first_half: bool,
    flip_second_half: bool,
) -> dict:
    """Function to get the base event data from the event based on
    the CloseToBallEvent class.

    Args:
        event (dict): event from FIFA data
        pitch_dimensions (list): pitch dimensions in x and y direction
        home_team_id (int): id of the home team
        away_team_id (int): id of the away team
        players (pd.DataFrame): dataframe with player information
        id (int): event id
        period_id (int): period id
        flip_first_half (bool): whether to flip first half coordinates
        flip_second_half (bool): whether to flip second half coordinates

    Returns:
        dict: dictionary with the base event data
    """

    x_norm = event.get("x_location_start") if "x_location_start" in event else event.get("x")
    y_norm = event.get("y_location_start") if "y_location_start" in event else event.get("y")
    
    if x_norm is None or pd.isna(x_norm):
        x_norm = 0.5
    if y_norm is None or pd.isna(y_norm):
        y_norm = 0.5
    
    x_norm = max(0.0, min(1.0, float(x_norm)))
    y_norm = max(0.0, min(1.0, float(y_norm)))
    
    if period_id == 1 and flip_first_half:
        x_norm = 1.0 - x_norm
        y_norm = 1.0 - y_norm
    elif period_id == 2 and flip_second_half:
        x_norm = 1.0 - x_norm
        y_norm = 1.0 - y_norm
    
    x_start = (x_norm * pitch_dimensions[0]) - (pitch_dimensions[0] / 2.0)
    y_start = (y_norm * pitch_dimensions[1]) - (pitch_dimensions[1] / 2.0)
    
    if event.get("team_id") == away_team_id:
        x_start *= -1
        y_start *= -1

    # Calculate time
    total_seconds = event.get("match_time_in_ms", 0) / 1000.0
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)

    # Get player jersey number
    player_id = event.get("from_player_id")
    jersey = players.loc[players["id"] == player_id, "shirt_num"].iloc[0] if len(
        players[players["id"] == player_id]
    ) > 0 else MISSING_INT

    return {
        "start_x": x_start,
        "start_y": y_start,
        "related_event_id": MISSING_INT,
        "event_id": id,
        "period_id": period_id,
        "minutes": minutes,
        "seconds": seconds,
        "datetime": pd.NaT,
        "team_id": event.get("team_id"),
        "team_side": "home" if event.get("team_id") != away_team_id else "away",
        "pitch_size": pitch_dimensions,
        "player_id": player_id,
        "jersey": jersey,
        "outcome": event.get("outcome") == "possession_complete" if "outcome" in event else True,
    }


def _get_game_score(
    events: pd.DataFrame, 
    home_team_id: int, 
    away_team_id: int
) -> tuple[int, int]:
    """
    Function to extract game scores by counting goals from event data.
    
    Args:
        events (pd.DataFrame): DataFrame of event data
        home_team_id (int): ID of the home team
        away_team_id (int): ID of the away team
    
    Returns:
        tuple[int, int]: (home_score, away_score)
    """
    home_score = 0
    away_score = 0
    
    shot_events = events[
        events["original_event"].isin(["attempt_at_goal", "own_goal"])
    ]

    for _, row in shot_events.iterrows():
        event_name = (row.get("original_event") or "").lower()
        team_id = row.get("team_id")    
        
        is_goal = False
        is_own_goal = False
        
        if event_name == "own_goal":
            is_own_goal = True
            is_goal = True
        elif event_name == "attempt_at_goal":
            if row.get("is_successful") == True:
                is_goal = True
        
        if is_goal:
            if is_own_goal:
                # Own goal: goal counts for opponent
                if team_id == home_team_id:
                    away_score += 1
                elif team_id == away_team_id:
                    home_score += 1
            else:
                # Normal goal
                if team_id == home_team_id:
                    home_score += 1
                elif team_id == away_team_id:
                    away_score += 1
    
    return home_score, away_score


def _determine_period_flips(
    events_list: list,
    home_team_id: int,
    away_team_id: int
) -> tuple[bool, bool]:
    """
    Determine whether to flip coordinates for first and second half
    
    Args:
        events_list: List of all events
        home_team_id: Home team ID
        away_team_id: Away team ID
    
    Returns:
        tuple[bool, bool]: (flip_first_half, flip_second_half)
    """
    flip_first_half = False
    flip_second_half = False
    
    # Find first half kickoff event
    for event in events_list:
        if (event.get("half_time") == 1 and 
            event.get("event", "").lower() == "game_period_start"):
            team_id = event.get("team_id")
            side = (event.get("side") or "").lower()
            
            # If home team kicks off with side='r', or away team kicks off with side='l', flip first half
            if (team_id == home_team_id and side == "r") or \
               (team_id == away_team_id and side == "l"):
                flip_first_half = True
            
            # If home team kicks off with side='l', or away team kicks off with side='r', flip second half
            if (team_id == home_team_id and side == "l") or \
               (team_id == away_team_id and side == "r"):
                flip_second_half = True
            
            break
    
    return flip_first_half, flip_second_half