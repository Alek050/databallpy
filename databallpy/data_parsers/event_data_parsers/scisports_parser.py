import json

import numpy as np
import pandas as pd

from databallpy.data_parsers.metadata import Metadata
from databallpy.events import DribbleEvent, PassEvent, ShotEvent, TackleEvent
from databallpy.utils.constants import MISSING_INT
from databallpy.utils.logging import logging_wrapper

BODY_PART_MAPPING = {
    "FEET": "foot",
    "HEAD": "head",
    "NOT_APPLICABLE": "unspecified",
    "RIGHT_FOOT": "right_foot",
    "LEFT_FOOT": "left_foot",
}


@logging_wrapper(__file__)
def load_scisports_event_data(
    events_json: str, pitch_dimensions: tuple = (106.0, 68.0)
) -> tuple[pd.DataFrame, Metadata, dict]:
    """This function retrieves the metadata and event data of a specific game. The x
    and y coordinates provided have been scaled to the dimensions of the pitch, with
    (0, 0) being the center. Additionally, the coordinates have been standardized so
    that the home team is represented as playing from left to right for the entire
    game, and the away team is represented as playing from right to left.

    Args:
        events_json (str): location of the event.json file.
        pitch_dimensions (tuple, optional): the length and width of the pitch in meters

    Returns:
        Tuple[pd.DataFrame, Metadata, dict]: the event data of the game, the metadata,
        and the databallpy_events.
    """
    if not isinstance(pitch_dimensions, (tuple, list)) or len(pitch_dimensions) != 2:
        raise ValueError(
            f"Invalid pitch_dimensions: {pitch_dimensions}. "
            "Must be a tuple of length 2."
        )

    metadata = _load_metadata(events_json, pitch_dimensions)
    event_data, databallpy_events = _load_event_data(events_json, metadata)
    return event_data, metadata, databallpy_events


@logging_wrapper(__file__)
def _load_metadata(events_json: str, pitch_dimensions: tuple) -> Metadata:
    """This function retrieves the metadata of a specific game.

    Args:
        events_json (str): location of the events.json file.
        pitch_dimensions (tuple): the length and width of the pitch in meters.

    Returns:
        Metadata: the metadata of the game.
    """
    with open(events_json, "r", encoding="utf-8") as f:
        events_json = json.load(f)

    date = pd.to_datetime(events_json["metaData"]["name"].split(" ")[0], dayfirst=True)
    periods_frames = _get_periods_frames(events_json, date, "Europe/Amsterdam")
    home_players, away_players = _get_players(
        events_json, events_json["metaData"]["homeTeamId"]
    )

    metadata = Metadata(
        game_id=events_json["metaData"]["id"],
        pitch_dimensions=pitch_dimensions,
        periods_frames=periods_frames,
        frame_rate=MISSING_INT,
        home_team_id=events_json["metaData"]["homeTeamId"],
        home_team_name=events_json["metaData"]["homeTeamName"],
        home_players=home_players,
        home_score=events_json["metaData"]["homeTeamGoals"],
        home_formation=None,
        away_team_id=events_json["metaData"]["awayTeamId"],
        away_team_name=events_json["metaData"]["awayTeamName"],
        away_players=away_players,
        away_score=events_json["metaData"]["awayTeamGoals"],
        away_formation=None,
        country="Netherlands",
        periods_changed_playing_direction=None,
    )
    return metadata


def _get_players(
    events_json: dict, home_team_id: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """This function retrieves the players of a specific game.

    Args:
        events_json (dict): the events.json file.
        home_team_id (int): the id of the home team.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: the home and away players of the game.
    """

    home_players = {
        "id": [],
        "full_name": [],
        "formation_place": [],
        "position": [],
        "starter": [],
        "shirt_num": [],
    }
    away_players = {
        "id": [],
        "full_name": [],
        "formation_place": [],
        "position": [],
        "starter": [],
        "shirt_num": [],
    }

    position_mapping = {
        "GK": "goalkeeper",
        "CB": "defender",
        "LB": "defender",
        "RB": "defender",
        "CM": "midfielder",
        "LM": "midfielder",
        "RM": "midfielder",
        "DMF": "midfielder",
        "AMF": "midfielder",
        "CMF": "midfielder",
        "LW": "forward",
        "RW": "forward",
        "CF": "forward",
        "UNKNOWN": "",
    }

    for player in events_json["players"]:
        players = home_players if player["teamId"] == home_team_id else away_players
        players["id"].append(player["playerId"])
        players["full_name"].append(player["playerName"])
        players["formation_place"].append(player["positionId"])
        players["position"].append(position_mapping[player["positionName"]])
        players["starter"].append(False)
        players["shirt_num"].append(player["shirtNumber"])

    home_players = pd.DataFrame(home_players)
    away_players = pd.DataFrame(away_players)

    for start_event in [
        event
        for event in events_json["events"]
        if event["subTypeName"] == "PLAYER_STARTING_POSITION"
        and event["startTimeMs"] == 0
    ]:
        df = home_players if start_event["teamId"] == home_team_id else away_players
        df.loc[df["id"] == start_event["playerId"], "starter"] = True

    return home_players, away_players


def _get_periods_frames(events_json: dict, date: pd.Timestamp, tz: str) -> pd.DataFrame:
    """This function retrieves the periods and frames of a specific game.

    Args:
        events_json (dict): the events.json file.
        date (pd.Timestamp): the date of the game.
        tz (str): the timezone of the game.

    Returns:
        pd.DataFrame: the periods and frames of the game.
    """
    first_half_start_ms = [
        event["startTimeMs"]
        for event in events_json["events"]
        if event["partName"] == "FIRST_HALF" and event["subTypeName"] == "KICK_OFF"
    ][0]
    first_half_end_ms = [
        event["endTimeMs"]
        for event in events_json["events"]
        if event["partName"] == "FIRST_HALF"
    ][-1]
    second_half_start_ms = [
        event["startTimeMs"]
        for event in events_json["events"]
        if event["partName"] == "SECOND_HALF"
        and event["subTypeName"] in ["PASS", "KICK_OFF"]
    ][0]
    second_half_end_ms = [
        event["endTimeMs"]
        for event in events_json["events"]
        if event["partName"] == "SECOND_HALF"
    ][-1]

    periods_frames = pd.DataFrame(
        {
            "period_id": [1, 2, 3, 4, 5],
            "start_datetime_ed": [
                date + pd.to_timedelta(first_half_start_ms, unit="ms"),
                date + pd.to_timedelta(second_half_start_ms, unit="ms"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
            ],
            "end_datetime_ed": [
                date + pd.to_timedelta(first_half_end_ms, unit="ms"),
                date + pd.to_timedelta(second_half_end_ms, unit="ms"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
            ],
        }
    )
    periods_frames["start_datetime_ed"] = periods_frames[
        "start_datetime_ed"
    ].dt.tz_localize(tz)
    periods_frames["end_datetime_ed"] = periods_frames["end_datetime_ed"].dt.tz_localize(
        tz
    )
    return periods_frames


@logging_wrapper(__file__)
def _load_event_data(events_json: str, metadata: Metadata) -> tuple[pd.DataFrame, dict]:
    """This function retrieves the event data of a specific game. The x
    and y coordinates provided have been scaled to the dimensions of the pitch, with
    (0, 0) being the center. Additionally, the coordinates have been standardized so
    that the home team is represented as playing from left to right for the entire
    game, and the away team is represented as playing from right to left.

    Args:
        events_json (str): location of the events.json file.
        metadata (Metadata): the metadata of the game.

    Returns:
        Tuple[pd.DataFrame, dict]: the event data of the game and the databallpy_events
    """
    with open(events_json, "r", encoding="utf-8") as f:
        events_json = json.load(f)

    event_data = {
        "event_id": [],
        "databallpy_event": [],
        "period_id": [],
        "minutes": [],
        "seconds": [],
        "player_id": [],
        "player_name": [],
        "team_id": [],
        "is_successful": [],
        "start_x": [],
        "start_y": [],
        "datetime": [],
        "original_event_id": [],
        "original_event": [],
    }

    databallpy_event_mapping = {
        "PASS": "pass",
        "CROSS": "pass",
        "SHOT": "shot",
        "DRIBBLE": "dribble",
        "DEFENSIVE_DUEL": "tackle",
        "FOUL": "tackle",
    }
    all_players = pd.concat(
        [metadata.home_players, metadata.away_players], ignore_index=True
    )
    shot_events = {}
    pass_events = {}
    dribble_events = {}
    other_events = {}
    date = pd.to_datetime(
        metadata.periods_frames["start_datetime_ed"].iloc[0].date()
    ).tz_localize(metadata.periods_frames["start_datetime_ed"].iloc[0].tz)
    for id, event in enumerate(events_json["events"]):
        event_data["event_id"].append(id)
        event_data["original_event_id"].append(id)

        event_data["period_id"].append(event["partId"])
        event_data["minutes"].append(MISSING_INT)
        event_data["seconds"].append(event["startTimeMs"] / 1000)
        event_data["player_id"].append(
            event["playerId"] if event["playerId"] != -1 else MISSING_INT
        )
        event_data["team_id"].append(
            event["teamId"] if event["teamId"] != -1 else MISSING_INT
        )

        event_data["start_x"].append(event["startPosXM"])
        event_data["start_y"].append(event["startPosYM"])
        event_data["datetime"].append(
            date + pd.to_timedelta(event["startTimeMs"], unit="ms")
        )
        event_data["original_event"].append(event["baseTypeName"].lower())
        event_data["player_name"].append(event["playerName"])

        multiplier = [-1, 1][int(event["groupName"].lower() == "home")]

        if event["baseTypeName"] in databallpy_event_mapping:
            databallpy_event = databallpy_event_mapping[event["baseTypeName"]]
            if databallpy_event == "tackle":
                databallpy_event = (
                    None
                    if "TACKLE" not in [event["subTypeName"], event["foulTypeName"]]
                    else "tackle"
                )
            event_data["databallpy_event"].append(databallpy_event)
            event_data["is_successful"].append(event["resultId"] == 1)
            if databallpy_event == "shot" and not event["playerId"] == -1:
                shot_events[id] = _get_shot_event(event, id, all_players, multiplier)
            elif databallpy_event == "pass" and not event["playerId"] == -1:
                pass_events[id] = _get_pass_event(event, id, all_players, multiplier)
            elif databallpy_event == "dribble" and not event["playerId"] == -1:
                dribble_events[id] = _get_dribble_event(
                    event, id, all_players, multiplier
                )
            elif databallpy_event == "tackle" and not event["playerId"] == -1:
                other_events[id] = _get_tackle_event(event, id, all_players, multiplier)
        else:
            event_data["databallpy_event"].append(None)
            event_data["is_successful"].append(None)

    event_data = pd.DataFrame(event_data)
    event_data["player_name"] = event_data["player_name"].str.replace(
        "NOT_APPLICABLE", "not_applicable"
    )
    event_data.loc[event_data["period_id"] == 2, "seconds"] -= event_data.loc[
        event_data["period_id"] == 2, "seconds"
    ].min() - (45 * 60)
    event_data["minutes"] = (event_data["seconds"] // 60).astype(np.int64)
    event_data["seconds"] = event_data["seconds"] % 60
    event_data.loc[
        event_data["team_id"] == metadata.away_team_id, ["start_x", "start_y"]
    ] *= -1
    event_data["is_successful"] = event_data["is_successful"].astype("boolean")

    for event in {
        **shot_events,
        **pass_events,
        **dribble_events,
        **other_events,
    }.values():
        row = event_data.loc[event_data["event_id"] == event.event_id]
        event.minutes = row["minutes"].iloc[0]
        event.seconds = row["seconds"].iloc[0]
        event.datetime = row["datetime"].iloc[0]
        event.pitch_size = metadata.pitch_dimensions

    return event_data, {
        "shot_events": shot_events,
        "pass_events": pass_events,
        "dribble_events": dribble_events,
        "other_events": other_events,
    }


def _get_shot_event(
    event: dict, id: int, players: pd.DataFrame, multiplier: int
) -> ShotEvent:
    """This function retrieves the shot event of a specific match.

    Args:
        event (dict): the shot event.
        id (int): the id of the event.
        players (pd.DataFrame): the players of the match.
        multiplier (int): the multiplier for the coordinates.

    Returns:
        ShotEvent: the shot event of the game.
    """

    shot_result_mappping = {
        "ON_TARGET": "miss_on_target",
        "BLOCKED": "blocked",
        "WIDE": "miss_off_target",
        "POST": "miss_hit_post",
    }
    return ShotEvent(
        event_id=id,
        period_id=event["partId"],
        minutes=MISSING_INT,
        seconds=MISSING_INT,
        datetime=pd.to_datetime("NaT"),
        start_x=event["startPosXM"] * multiplier,
        start_y=event["startPosYM"] * multiplier,
        team_id=event["teamId"],
        team_side=event["groupName"].lower(),
        pitch_size=(106.0, 68.0),
        player_id=event["playerId"],
        jersey=players.loc[players["id"] == event["playerId"], "shirt_num"].iloc[0],
        outcome=bool(event["resultId"]),
        related_event_id=MISSING_INT,
        body_part=BODY_PART_MAPPING.get(event["bodyPartName"], "other"),
        possession_type="unspecified",
        set_piece="unspecified",
        _xt=-1.0,
        outcome_str=shot_result_mappping[event["shotTypeName"]]
        if event["resultId"] == 0
        else "goal",
    )


def _get_pass_event(
    event: dict, id: int, players: pd.DataFrame, multiplier: int
) -> PassEvent:
    """This function retrieves the pass event of a specific match.

    Args:
        event (dict): the pass event.
        id (int): the id of the event.
        players (pd.DataFrame): the players of the match.
        multiplier (int): the multiplier for the coordinates.
    Returns:
        PassEvent: the pass event of the game.
    """

    pass_type_mapping = {
        "FREE_KICK_CROSSED": "cross",
        "THROW_IN_CROSSED": "cross",
        "CROSS": "cross",
        "CORNER_CROSSED": "cross",
        "CROSS_CUTBACK": "pull_back",
        "PASS": "unspecified",
        "GOAL_KICK": "unspecified",
        "CORNER_SHORT": "unspecified",
        "OFFSIDE_PASS": "unspecified",
        "KICK_OFF": "unspecified",
        "THROW_IN": "unspecified",
        "FREE_KICK": "unspecified",
        "GOALKEEPER_THROW": "unspecified",
    }

    set_piece_mapping = {
        "FREE_KICK_CROSSED": "free_kick",
        "THROW_IN_CROSSED": "throw_in",
        "CROSS": "no_set_piece",
        "CORNER_CROSSED": "corner_kick",
        "CROSS_CUTBACK": "no_set_piece",
        "PASS": "no_set_piece",
        "GOAL_KICK": "goal_kick",
        "CORNER_SHORT": "corner_kick",
        "OFFSIDE_PASS": "no_set_piece",
        "KICK_OFF": "kick_off",
        "THROW_IN": "throw_in",
        "FREE_KICK": "free_kick",
        "GOALKEEPER_THROW": "no_set_piece",
    }
    return PassEvent(
        event_id=id,
        period_id=event["partId"],
        minutes=MISSING_INT,
        seconds=MISSING_INT,
        datetime=pd.to_datetime("NaT"),
        start_x=event["startPosXM"] * multiplier,
        start_y=event["startPosYM"] * multiplier,
        team_id=event["teamId"],
        team_side=event["groupName"].lower(),
        pitch_size=(106.0, 68.0),
        player_id=event["playerId"],
        jersey=players.loc[players["id"] == event["playerId"], "shirt_num"].iloc[0],
        outcome=bool(event["resultId"]),
        related_event_id=MISSING_INT,
        body_part=BODY_PART_MAPPING.get(event["bodyPartName"], "other"),
        possession_type="unspecified",
        set_piece=set_piece_mapping[event["subTypeName"]],
        _xt=-1.0,
        outcome_str=event["resultName"].lower()
        if event["subTypeName"] != "OFFSIDE_PASS"
        else "offside",
        end_x=event["endPosXM"] * multiplier,
        end_y=event["endPosYM"] * multiplier,
        pass_type=pass_type_mapping[event["subTypeName"]],
        receiver_player_id=event["receiverId"]
        if event["receiverTeamId"] == event["teamId"]
        else MISSING_INT,
    )


def _get_tackle_event(
    event: dict, id: int, players: pd.DataFrame, multiplier: int
) -> TackleEvent:
    """This function retrieves the tackle event of a specific match.

    Args:
        event (dict): the pass event.
        id (int): the id of the event.
        players (pd.DataFrame): the players of the match.
        multiplier (int): the multiplier for the coordinates.

    Returns:
        TackleEvent: the tackle event of the game.
    """
    return TackleEvent(
        event_id=id,
        period_id=event["partId"],
        minutes=MISSING_INT,
        seconds=MISSING_INT,
        datetime=pd.to_datetime("NaT"),
        start_x=event["startPosXM"] * multiplier,
        start_y=event["startPosYM"] * multiplier,
        team_id=event["teamId"],
        team_side=event["groupName"].lower(),
        pitch_size=(106.0, 68.0),
        player_id=event["playerId"],
        jersey=players.loc[players["id"] == event["playerId"], "shirt_num"].iloc[0],
        outcome=event["resultId"] == 1,
        related_event_id=MISSING_INT,
    )


def _get_dribble_event(
    event: dict, id: int, players: pd.DataFrame, multiplier: int
) -> DribbleEvent:
    """This function retrieves the dribble event of a specific match.

    Args:
        event (dict): the dribble event.
        id (int): the id of the event.
        players (pd.DataFrame): the players of the match.
        multiplier (int): the multiplier for the coordinates

    Returns:
        DribbleEvent: the dribble event of the game.
    """
    return DribbleEvent(
        event_id=id,
        period_id=event["partId"],
        minutes=MISSING_INT,
        seconds=MISSING_INT,
        datetime=pd.to_datetime("NaT"),
        start_x=event["startPosXM"] * multiplier,
        start_y=event["startPosYM"] * multiplier,
        team_id=event["teamId"],
        team_side=event["groupName"].lower(),
        pitch_size=(106.0, 68.0),
        player_id=event["playerId"],
        jersey=players.loc[players["id"] == event["playerId"], "shirt_num"].iloc[0],
        outcome=bool(event["resultId"]),
        related_event_id=MISSING_INT,
        body_part=BODY_PART_MAPPING.get(event["bodyPartName"], "other"),
        possession_type="unspecified",
        set_piece="no_set_piece",
        _xt=-1.0,
        duel_type="offensive",
        with_opponent=event["subTypeName"] == "TAKE_ON",
    )
