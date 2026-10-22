import datetime as dt
import html
import io
import json
import os
import re

import chardet
import numpy as np
import pandas as pd
import requests

from databallpy.data_parsers import Metadata
from databallpy.data_parsers.event_data_parsers.utils import (
    _add_spadl_columns,
    _normalize_playing_direction_events,
)
from databallpy.data_parsers.metrica_metadata_parser import (
    _get_metadata,
    _get_td_channels,
    _update_metadata,
)
from databallpy.events import DribbleEvent, PassEvent, ShotEvent
from databallpy.events.spadl import MIN_DRIBBLE_LENGTH
from databallpy.utils.constants import MISSING_INT
from databallpy.utils.logging import logging_wrapper
from databallpy.utils.utils import _to_float, _to_int

metrica_databallpy_map = {
    "pass": "pass",
    "carry": "dribble",
    "shot": "shot",
    "tackle": "tackle",
}

SPADL_SET_PIECE_TYPES = {
    "THROW IN": "throw_in",
    "CORNER KICK": "corner_short",
    "FREE KICK": "freekick_short",
    "KICK OFF": "pass",
}

SPADL_CARD_RESULTS = {
    "YELLOW": "yellow_card",
    "RED": "red_card",
}


@logging_wrapper(__file__)
def load_metrica_event_data(
    event_data_loc: str, metadata_loc: str
) -> tuple[pd.DataFrame, Metadata, dict]:
    """Function to load the metrica event data.

    Args:
        event_data_loc (str): location of the event data .json file
        metadata_loc (str): location of the metadata .xml file

    Raises:
        TypeError: type error if event_data_loc, or metadata_loc is not a valid input
        type (str)

    Returns:
        Tuple[pd.DataFrame, Metadata, dict]: The event data and the metadata, and databallpy events
    """
    if isinstance(event_data_loc, str) and "{" not in event_data_loc:
        if not os.path.exists(metadata_loc):
            raise FileNotFoundError(f"Could not find {metadata_loc}")
    elif isinstance(event_data_loc, str) and "{" in event_data_loc:
        # event_data_loc has a '{' in it. Expecting it to be the json file
        # with the event data, not the location of the event data .json.
        pass
    else:
        raise TypeError(
            "tracking_data_loc must be either a str or a StringIO object,"
            f" not a {type(event_data_loc)}"
        )

    metadata = _get_metadata(metadata_loc, is_tracking_data=False, is_event_data=True)
    td_channels = _get_td_channels(metadata_loc, metadata)
    metadata = _update_metadata(td_channels, metadata)
    event_data, cards = _get_event_data(event_data_loc)

    # rescale the event locations, metrica data is scaled between 0 and 1.
    for col in [x for x in event_data.columns if "_x" in x]:
        event_data[col] = (
            event_data[col] * metadata.pitch_dimensions[0]
            - metadata.pitch_dimensions[0] / 2.0
        )
    for col in [x for x in event_data.columns if "_y" in x]:
        event_data[col] = (
            event_data[col] * metadata.pitch_dimensions[1]
            - metadata.pitch_dimensions[1] / 2.0
        )

    # add datetime based on frame numbers
    first_frame = metadata.periods_frames.loc[
        metadata.periods_frames["period_id"] == 1, "start_frame"
    ].iloc[0]
    start_time = metadata.periods_frames.loc[
        metadata.periods_frames["period_id"] == 1, "start_datetime_ed"
    ].iloc[0]
    frame_rate = metadata.frame_rate
    rel_timedelta = [
        dt.timedelta(milliseconds=(x - first_frame) / frame_rate * 1000)
        for x in event_data["td_frame"]
    ]

    # no idea about time zone since we have no real data, so just assume utc
    event_data["datetime"] = [
        pd.to_datetime(start_time, utc=True) + x for x in rel_timedelta
    ]

    event_data = _normalize_playing_direction_events(
        event_data, metadata.home_team_id, metadata.away_team_id
    )
    event_data = _update_metrica_spadl_actions(event_data, cards)
    databallpy_events = _get_databallpy_events(
        event_data,
        metadata.pitch_dimensions,
        metadata.home_team_id,
        pd.concat([metadata.home_players, metadata.away_players], ignore_index=True),
    )
    return event_data, metadata, databallpy_events


@logging_wrapper(__file__)
def load_metrica_open_event_data() -> tuple[pd.DataFrame, Metadata, dict]:
    """Function to load the open event data of metrica

    Returns:
        Tuple[pd.DataFrame, Metadata]: event data and metadata of the game and databallpy events
    """
    metadata_link = "https://raw.githubusercontent.com/metrica-sports/sample-data\
        /master/data/Sample_Game_3/Sample_Game_3_metadata.xml"
    ed_link = "https://raw.githubusercontent.com/metrica-sports/sample-data\
        /master/data/Sample_Game_3/Sample_Game_3_events.json"

    raw_ed = requests.get(ed_link).text
    raw_metadata = requests.get(metadata_link).text

    return load_metrica_event_data(raw_ed, raw_metadata)


@logging_wrapper(__file__)
def _get_event_data(event_data_loc: str | io.StringIO) -> tuple[pd.DataFrame, dict]:
    """Function to load metrica event data

    Args:
        event_data_loc (Union[str, io.StringIO]): location of the event data file

    Returns:
        tuple[pd.DataFrame, dict]: event data and the index of the card events with
            their SPADL result
    """

    if isinstance(event_data_loc, str) and "{" not in event_data_loc:
        with open(event_data_loc, "rb") as file:
            encoding = chardet.detect(file.read())["encoding"]
        with open(event_data_loc, "r", encoding=encoding) as file:
            lines = file.readlines()
        raw_data = "".join(str(i) for i in lines)
        events_dict = json.loads(html.unescape(re.sub(r"<[^>]+>", "", raw_data)))
    else:
        events_dict = json.loads(
            html.unescape(re.sub(r"<[^>]+>", "", event_data_loc.strip()))
        )

    result_dict = {
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
        "to_player_id": [],
        "to_player_name": [],
        "end_x": [],
        "end_y": [],
        "original_event_id": [],
        "original_event": [],
        "event_type_id": [],
        "td_frame": [],
        "action_type": [],
        "action_result": [],
        "action_bodypart": [],
    }

    check_outcome_last_event = False
    cards = {}
    set_piece = None

    in_possession_events = ["pass", "carry", "recovery", "shot"]
    out_of_possession_events = ["fault received", "ball out", "ball lost"]

    for i_event, event in enumerate(events_dict["data"]):
        result_dict["event_id"].append(i_event)
        result_dict["original_event_id"].append(event["index"])
        result_dict["event_type_id"].append(event["type"]["id"])
        event_name = event["type"]["name"].lower()
        if event_name == "challenge":
            if _is_in_subtypes(event["subtypes"], "TACKLE"):
                event_name = "tackle"
        result_dict["original_event"].append(event_name)
        result_dict["period_id"].append(event["period"])
        result_dict["minutes"].append(_to_int((event["start"]["time"] // 60)))
        result_dict["seconds"].append(_to_float(event["start"]["time"] % 60))
        result_dict["player_id"].append(_to_int(event["from"]["id"][1:]))
        result_dict["player_name"].append(event["from"]["name"])

        # set outcome for pass or dribble/carry events
        if check_outcome_last_event:
            if (
                event_name in out_of_possession_events
                and result_dict["team_id"][-1] == event["team"]["id"]
            ) or (
                event_name in in_possession_events
                and result_dict["team_id"][-1] != event["team"]["id"]
            ):
                result_dict["is_successful"][-1] = False
            else:
                result_dict["is_successful"][-1] = True
            check_outcome_last_event = False

        # set outcome for shot events
        if event_name in ["shot", "tackle"]:
            if _is_in_subtypes(event["subtypes"], "GOAL") or _is_in_subtypes(
                event["subtypes"], "WON"
            ):
                outcome = True
            else:
                outcome = False
            result_dict["is_successful"].append(outcome)
        else:
            result_dict["is_successful"].append(None)

        # Check if outcome needs to be set based on next event
        if event_name in ["pass", "carry"]:
            check_outcome_last_event = True

        result_dict["team_id"].append(event["team"]["id"])
        result_dict["start_x"].append(_to_float(event["start"]["x"]))
        result_dict["start_y"].append(_to_float(event["start"]["y"]))
        if event["to"] is not None:
            result_dict["to_player_id"].append(_to_int(event["to"]["id"][1:]))
            result_dict["to_player_name"].append(event["to"]["name"])
        else:
            result_dict["to_player_id"].append(MISSING_INT)
            result_dict["to_player_name"].append(None)
        result_dict["end_x"].append(_to_float(event["end"]["x"]))
        result_dict["end_y"].append(_to_float(event["end"]["y"]))
        result_dict["td_frame"].append(event["start"]["frame"])

        # set pieces are no actions, but the type of the next action of the same
        # player in the same frame
        if (
            set_piece is not None
            and set_piece["start"]["frame"] == event["start"]["frame"]
            and set_piece["from"]["id"] == event["from"]["id"]
        ):
            set_piece_subtypes = set_piece["subtypes"]
        else:
            set_piece_subtypes = None
        set_piece = event if event["type"]["name"] == "SET PIECE" else None

        action_type, action_result, action_bodypart = _get_metrica_spadl_action(
            event, set_piece_subtypes
        )
        result_dict["action_type"].append(action_type)
        result_dict["action_result"].append(action_result)
        result_dict["action_bodypart"].append(action_bodypart)
        if event["type"]["name"] == "CARD":
            for card_type, card_result in SPADL_CARD_RESULTS.items():
                if _has_subtype(event, card_type):
                    cards[i_event] = card_result

    result_dict["databallpy_event"] = [None] * len(result_dict["event_id"])
    events = pd.DataFrame(result_dict)
    events["databallpy_event"] = (
        events["original_event"].map(metrica_databallpy_map).replace([np.nan], [None])
    )
    events["is_successful"] = events["is_successful"].astype("boolean")
    events = _add_spadl_columns(events)
    return events, cards


def _has_subtype(event: dict, name: str) -> bool:
    """Function to check if an event has a subtype, also for events without subtypes

    Args:
        event (dict): event from the metrica json
        name (str): name of the subtype

    Returns:
        bool: True if the event has the subtype, False otherwise
    """
    subtypes = event.get("subtypes")
    return subtypes is not None and _is_in_subtypes(subtypes, name)


def _get_metrica_spadl_action(
    event: dict, set_piece_subtypes: list[dict] | dict | None
) -> tuple[str | None, str | None, str | None]:
    """Function to map a metrica event to the SPADL vocabulary. Ball lost and ball out
    events with an end location are failed passes, plain ball recoveries are mapped to
    dribbles and updated in _update_metrica_spadl_actions.

    Args:
        event (dict): event from the metrica json
        set_piece_subtypes (list[dict] | dict | None): subtypes of the set piece
            that started this event, None if the event did not start with a set piece

    Returns:
        tuple[str | None, str | None, str | None]: The SPADL action type, result and
            bodypart, all None for events that are not SPADL actions
    """
    event_type = event["type"]["name"]
    bodypart = "head" if _has_subtype(event, "HEAD") else "foot"
    set_piece = {"subtypes": set_piece_subtypes}
    has_end_location = event["end"]["x"] is not None and (
        event["start"]["x"] != event["end"]["x"]
        or event["start"]["y"] != event["end"]["y"]
    )

    if event_type == "BALL LOST" and _has_subtype(event, "HAND BALL"):
        return "foul", "fail", "other"
    elif event_type == "BALL LOST" and (
        _has_subtype(event, "THEFT") or _has_subtype(event, "FORCED")
    ):
        return "bad_touch", "fail", bodypart
    elif event_type == "PASS" or (
        event_type in ["BALL LOST", "BALL OUT"] and has_end_location
    ):
        result = "success" if event_type == "PASS" else "fail"
        if _has_subtype(event, "OFFSIDE"):
            result = "offside"
        action_type = "pass"
        for set_piece_type, spadl_type in SPADL_SET_PIECE_TYPES.items():
            if _has_subtype(set_piece, set_piece_type):
                action_type = spadl_type
        if action_type in ["corner_short", "freekick_short"] and _has_subtype(
            event, "CROSS"
        ):
            action_type = action_type.replace("short", "crossed")
        elif action_type == "throw_in":
            bodypart = "other"
        elif set_piece_subtypes is None and _has_subtype(event, "GOAL KICK"):
            action_type = "goalkick"
        elif set_piece_subtypes is None and _has_subtype(event, "CLEARANCE"):
            action_type, result = "clearance", "success"
        elif set_piece_subtypes is None and _has_subtype(event, "CROSS"):
            action_type = "cross"
        return action_type, result, bodypart
    elif event_type == "SHOT":
        action_type = "shot"
        if _has_subtype(set_piece, "PENALTY"):
            action_type = "shot_penalty"
        elif _has_subtype(set_piece, "FREE KICK"):
            action_type = "shot_freekick"
        result = "success" if _has_subtype(event, "GOAL") else "fail"
        return action_type, result, bodypart
    elif event_type == "CARRY" and has_end_location:
        return "dribble", "success", "foot"
    elif event_type == "CHALLENGE" and _has_subtype(event, "LOST"):
        if _has_subtype(event, "FAULT"):
            return "foul", "fail", "foot"
        elif _has_subtype(event, "TACKLE"):
            return "tackle", "fail", "foot"
    elif event_type == "CHALLENGE" and _has_subtype(event, "TACKLE"):
        if _has_subtype(event, "WON") and not _has_subtype(event, "FAULT"):
            return "tackle", "success", "foot"
    elif event_type == "RECOVERY":
        if _has_subtype(event, "INTERCEPTION"):
            return "interception", "success", "foot"
        elif _has_subtype(event, "SAVED"):
            return "keeper_save", "success", "other"
        elif event.get("subtypes") is None:
            return "dribble", "success", "foot"

    return None, None, None


def _update_metrica_spadl_actions(event_data: pd.DataFrame, cards: dict) -> pd.DataFrame:
    """Function to update the SPADL actions with information of the surrounding
    events. Plain ball recoveries become dribbles to the start of the next action, or
    no action when the ball did not move, and cards set the result of the last foul
    of the same player.

    Args:
        event_data (pd.DataFrame): The event data in databallpy coordinates, with the
            home team playing from left to right
        cards (dict): The index of the card events with their SPADL result

    Returns:
        pd.DataFrame: The event data with the updated SPADL actions
    """
    spadl_columns = ["action_type", "action_result", "action_bodypart"]
    is_recovery = (event_data["original_event"] == "recovery") & (
        event_data["action_type"] == "dribble"
    )

    # ball recoveries without movement are no actions
    next_start = (
        event_data.loc[
            event_data["action_type"].notna() & ~is_recovery, ["start_x", "start_y"]
        ]
        .reindex(event_data.index)
        .shift(-1)
        .bfill()
    )
    moved = (
        (event_data["start_x"] - next_start["start_x"]).abs() >= MIN_DRIBBLE_LENGTH
    ) | ((event_data["start_y"] - next_start["start_y"]).abs() >= MIN_DRIBBLE_LENGTH)
    event_data.loc[is_recovery & ~moved, spadl_columns] = None
    event_data.loc[is_recovery & moved, ["end_x", "end_y"]] = next_start.loc[
        is_recovery & moved
    ].values

    # cards are no actions, but the result of the last foul of the same player,
    # metrica cards are given some time after the foul. Cards without a foul in the
    # previous 60 seconds (e.g. for dissent) are ignored
    for card_idx, card_result in cards.items():
        card = event_data.loc[card_idx]
        fouls = event_data.loc[
            (event_data["action_type"] == "foul")
            & (event_data["player_id"] == card["player_id"])
            & (event_data["period_id"] == card["period_id"])
            & (event_data["datetime"] <= card["datetime"])
            & (event_data["datetime"] >= card["datetime"] - pd.Timedelta(seconds=60))
        ]
        if len(fouls) > 0:
            event_data.loc[fouls.index[-1], "action_result"] = card_result

    return event_data


def _is_in_subtypes(subtypes: list[dict] | dict, name: str) -> bool:
    """Function to search for a name in the subtypes

    Args:
        subtypes (list[dict] | dict): list of subtypes
        name (str): name to search for

    Returns:
        bool: True if the name is in the subtypes, False otherwise
    """
    result = False
    if isinstance(subtypes, list):
        for sub in subtypes:
            if sub["name"] == name:
                result = True
                break
    else:
        if subtypes["name"] == name:
            result = True
    return result


@logging_wrapper(__file__)
def _get_databallpy_events(
    event_data: pd.DataFrame,
    pitch_dimensions: tuple[float, float],
    home_team_id: int,
    all_players: pd.DataFrame,
) -> dict:
    """Function to get the databallpy events from the event data

    Args:
        event_data (pd.DataFrame): event data
        pitch_dimensions (tuple): dimensions of the pitch
        home_team_id (int): id of the home team
        all_players (pd.DataFrame): metadata of all the players

    Returns:
        dict: dictionary with the databallpy events
    """
    shot_events = {}
    pass_events = {}
    dribble_events = {}

    shot_mask = event_data["databallpy_event"] == "shot"
    shot_events = (
        {
            shot.event_id: shot
            for shot in event_data[shot_mask].apply(
                _get_shot_event,
                pitch_dimensions=pitch_dimensions,
                home_team_id=home_team_id,
                players=all_players,
                axis=1,
            )
        }
        if shot_mask.sum() > 0
        else {}
    )

    pass_maks = event_data["databallpy_event"] == "pass"
    pass_events = (
        {
            pass_.event_id: pass_
            for pass_ in event_data[pass_maks].apply(
                _get_pass_event,
                pitch_dimensions=pitch_dimensions,
                home_team_id=home_team_id,
                players=all_players,
                axis=1,
            )
        }
        if pass_maks.sum() > 0
        else {}
    )

    dribble_mask = event_data["databallpy_event"] == "dribble"
    dribble_events = (
        {
            dribble.event_id: dribble
            for dribble in event_data[dribble_mask].apply(
                _get_dribble_event,
                pitch_dimensions=pitch_dimensions,
                home_team_id=home_team_id,
                players=all_players,
                axis=1,
            )
        }
        if dribble_mask.sum() > 0
        else {}
    )

    databallpy_events = {
        "shot_events": shot_events,
        "pass_events": pass_events,
        "dribble_events": dribble_events,
    }
    return databallpy_events


def _get_shot_event(
    row: pd.Series,
    pitch_dimensions: tuple[float, float],
    home_team_id: int,
    players: pd.DataFrame,
) -> ShotEvent:
    """Function to return a ShotEvent object from a row of the metrica
      event data

    Args:
        row (pd.Series): row of the metrica event data with a shot event
        pitch_dimensions (tuple): dimensions of the pitch
        home_team_id (int): id of the home team
        players: pd.DataFrame: Metadata of the players

    Returns:
        ShotEvent: ShotEvent object
    """
    return ShotEvent(
        event_id=row.event_id,
        period_id=row.period_id,
        minutes=row.minutes,
        seconds=row.seconds,
        datetime=row.datetime,
        start_x=row.start_x,
        start_y=row.start_y,
        team_id=row.team_id,
        team_side="home" if row.team_id == home_team_id else "away",
        pitch_size=pitch_dimensions,
        player_id=row.player_id,
        jersey=players.loc[players["id"] == row.player_id, "shirt_num"].iloc[0],
        outcome=bool(row.is_successful),
        related_event_id=MISSING_INT,
        body_part="unspecified",
        possession_type="unspecified",
        set_piece="unspecified",
        _xt=np.nan,
        outcome_str=["miss", "goal"][row.is_successful],
    )


def _get_pass_event(
    row: pd.Series,
    pitch_dimensions: tuple[float, float],
    home_team_id: int,
    players: pd.DataFrame,
) -> PassEvent:
    """Function to return a PassEvent object from a row of the metrica
     event data.

    Args:
        row (pd.Series): row of the metrica event data with a pass event
        pitch_dimensions (tuple): dimensions of the pitch
        home_team_id (int): id of the home team
        players: pd.DataFrame: Metadata of the players

    Returns:
        PassEvent: PassEvent object
    """
    return PassEvent(
        event_id=row.event_id,
        period_id=row.period_id,
        minutes=row.minutes,
        seconds=row.seconds,
        datetime=row.datetime,
        start_x=row.start_x,
        start_y=row.start_y,
        team_id=row.team_id,
        team_side="home" if row.team_id == home_team_id else "away",
        pitch_size=pitch_dimensions,
        player_id=row.player_id,
        jersey=players.loc[players["id"] == row.player_id, "shirt_num"].iloc[0],
        outcome=bool(row.is_successful),
        related_event_id=MISSING_INT,
        body_part="unspecified",
        possession_type="unspecified",
        set_piece="unspecified",
        _xt=np.nan,
        outcome_str=["unsuccessful", "successful"][row.is_successful]
        if not pd.isnull(row.is_successful)
        else "not_specified",
        end_x=row.end_x,
        end_y=row.end_y,
        pass_type="unspecified",
    )


def _get_dribble_event(
    row: pd.Series,
    pitch_dimensions: tuple[float, float],
    home_team_id: int,
    players: pd.DataFrame,
) -> DribbleEvent:
    """Function to return a DribbleEvent object from a row of the metrica
     event data.

    Args:
        row (pd.Series): row of the metrica event data with a dribble event
        pitch_dimensions (tuple): dimensions of the pitch
        home_team_id (int): id of the home team
        players: pd.DataFrame: Metadata of the players

    Returns:
        DribbleEvent: DribbleEvent object
    """
    return DribbleEvent(
        event_id=row.event_id,
        period_id=row.period_id,
        minutes=row.minutes,
        seconds=row.seconds,
        datetime=row.datetime,
        start_x=row.start_x,
        start_y=row.start_y,
        team_id=row.team_id,
        team_side="home" if row.team_id == home_team_id else "away",
        pitch_size=pitch_dimensions,
        player_id=row.player_id,
        jersey=players.loc[players["id"] == row.player_id, "shirt_num"].iloc[0],
        outcome=bool(row.is_successful),
        related_event_id=MISSING_INT,
        body_part="unspecified",
        possession_type="unspecified",
        set_piece="unspecified",
        _xt=np.nan,
        duel_type="unspecified",
        with_opponent=False,
    )
