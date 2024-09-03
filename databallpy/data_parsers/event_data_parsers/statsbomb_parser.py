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
    events_json: str, pitch_dimensions: tuple = (106.0, 68.0)
) -> tuple[pd.DataFrame, Metadata, dict]:
    """This function retrieves the metadata and event data of a specific match. The x
    and y coordinates provided have been scaled to the dimensions of the pitch, with
    (0, 0) being the center. Additionally, the coordinates have been standardized so
    that the home team is represented as playing from left to right for the entire
    match, and the away team is represented as playing from right to left.

    Args:
        events_json (str): location of the event.json file.
        pitch_dimensions (tuple, optional): the length and width of the pitch in meters

    Returns:
        Tuple[pd.DataFrame, Metadata, dict]: the event data of the match, the metadata,
        and the databallpy_events.
    """
    LOGGER.info(f"Loading Statsbomb event data: events_json: {events_json}")
    if not os.path.exists(events_json):
        LOGGER.error(f"File {events_json} does not exist.")
        raise FileNotFoundError(f"File {events_json} does not exist.")

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
    # metadata = _load_metadata(events_json, pitch_dimensions)
    LOGGER.info("Successfully loaded Statsbomb metadata.")
    # Load the event data
    event_data, databallpy_events = _load_event_data(events_json)#, metadata)

    LOGGER.info("Successfully loaded Statsbomb event data and databallpy events.")
    return event_data, #metadata, databallpy_events

def _load_event_data(events_json: str)-> tuple[pd.DataFrame, dict]:#, metadata: Metadata) -> tuple[pd.DataFrame, dict]:
    """This function retrieves the event data of a specific match. The x
    and y coordinates provided have been scaled to the dimensions of the pitch, with
    (0, 0) being the center. Additionally, the coordinates have been standardized so
    that the home team is represented as playing from left to right for the entire
    match, and the away team is represented as playing from right to left.

    Args:
        events_json (str): location of the events.json file.
        metadata (Metadata): the metadata of the match.

    Returns:
        Tuple[pd.DataFrame, dict]: the event data of the match and the databallpy_events
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
        "team_id": [],
        "outcome": [],
        "start_x": [],
        "start_y": [],
        "datetime": [],
        "statsbomb_event": [],
        "player_name": [],
        "team_name": [],
    }
    
    # databallpy_mapping = {
    #     "Pass": "pass",
    #     "CROSS": "pass",
    #     "SHOT": "shot",
    #     "DRIBBLE": "dribble",
    # }
    # all_players = pd.concat(
    #     [metadata.home_players, metadata.away_players], ignore_index=True
    # )
    shot_events = {}
    pass_events = {}
    dribble_events = {}
    # date = pd.to_datetime(
    #     metadata.periods_frames["start_datetime_ed"].iloc[0].date()
    # ).tz_localize(metadata.periods_frames["start_datetime_ed"].iloc[0].tz)
    
    for id, event in enumerate(events_json): 
        if id < 5:
            print(id)
            pass
        else:
            import pdb;pdb.set_trace()
            print(id)     
            event_data["event_id"].append(id)

            event_data["period_id"].append(event["period"])
            event_data["minutes"].append(event["minute"])
            event_data["seconds"].append(event["second"])
            event_data["player_id"].append(event["player"]["id"])
            event_data["team_id"].append(event["possession_team"]["id"])
            event_data["start_x"].append(event["location"][0])
            event_data["start_y"].append(event["location"][1])
            event_data["datetime"].append(
                "placeholder"
            )
            event_data["statsbomb_event"].append(event["type"]["name"].lower())
            event_data["player_name"].append(event["player"]["name"])
            event_data["team_name"].append(event["possession_team"]["name"])

        # if event["baseTypeName"] in databallpy_mapping:
        #     databallpy_event = databallpy_mapping[event["baseTypeName"]]
        #     event_data["databallpy_event"].append(databallpy_event)
        #     event_data["outcome"].append(event["resultId"])
        #     if databallpy_event == "shot" and not event["playerId"] == -1:
        #         shot_events[id] = _get_shot_event(event, id, all_players)
        #     elif databallpy_event == "pass" and not event["playerId"] == -1:
        #         pass_events[id] = _get_pass_event(event, id, all_players)
        #     elif databallpy_event == "dribble" and not event["playerId"] == -1:
        #         dribble_events[id] = _get_dribble_event(event, id, all_players)
        # else:
        #     event_data["databallpy_event"].append(None)
        #     event_data["outcome"].append(MISSING_INT)

    event_data = pd.DataFrame(event_data)
    # event_data["player_name"] = event_data["player_name"].str.replace(
    #     "NOT_APPLICABLE", "not_applicable"
    # )
    # event_data.loc[event_data["period_id"] == 2, "seconds"] -= event_data.loc[
    #     event_data["period_id"] == 2, "seconds"
    # ].min() - (45 * 60)
    # event_data["minutes"] = (event_data["seconds"] // 60).astype(np.int64)
    # event_data["seconds"] = event_data["seconds"] % 60
    # event_data.loc[
    #     event_data["team_id"] == metadata.away_team_id, ["start_x", "start_y"]
    # ] *= -1

    # for event in {**shot_events, **pass_events, **dribble_events}.values():
    #     row = event_data.loc[event_data["event_id"] == event.event_id]
    #     event.minutes = row["minutes"].iloc[0]
    #     event.seconds = row["seconds"].iloc[0]
    #     event.datetime = row["datetime"].iloc[0]
    #     event.pitch_size = metadata.pitch_dimensions
    #     if event.team_side == "away":
    #         event.start_x *= -1
    #         event.start_y *= -1
    #         if isinstance(event, PassEvent):
    #             event.end_x *= -1
    #             event.end_y *= -1

    return event_data, {
        "shot_events": shot_events,
        "pass_events": pass_events,
        "dribble_events": dribble_events,
    }
