import datetime as dt
import io
import json
import os
from typing import Tuple, Union

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from databallpy.load_data.event_data._normalize_playing_direction_events import (
    _normalize_playing_direction_events,
)
from databallpy.load_data.event_data.dribble_event import DribbleEvent
from databallpy.load_data.event_data.pass_event import PassEvent
from databallpy.load_data.event_data.shot_event import ShotEvent
from databallpy.load_data.metadata import Metadata
from databallpy.load_data.metrica_metadata import (
    _get_metadata,
    _get_td_channels,
    _update_metadata,
)
from databallpy.utils.utils import MISSING_INT, _to_float, _to_int

metrica_databallpy_map = {
    "pass": "pass",
    "carry": "dribble",
    "shot": "shot",
}


def load_metrica_event_data(
    event_data_loc: str, metadata_loc: str
) -> Tuple[pd.DataFrame, Metadata]:
    """Function to load the metrica event data.

    Args:
        event_data_loc (str): location of the event data .json file
        metadata_loc (str): location of the metadata .xml file

    Raises:
        TypeError: type error if event_data_loc, or metadata_loc is not a valid input
        type (str)

    Returns:
        Tuple[pd.DataFrame, Metadata]: The event data and the metadata
    """

    if isinstance(event_data_loc, str) and "{" not in event_data_loc:
        assert os.path.exists(event_data_loc)
        assert os.path.exists(metadata_loc)
    elif isinstance(event_data_loc, str) and "{" in event_data_loc:
        pass
    else:
        raise TypeError(
            "tracking_data_loc must be either a str or a StringIO object,"
            f" not a {type(event_data_loc)}"
        )

    metadata = _get_metadata(metadata_loc, is_tracking_data=False, is_event_data=True)
    td_channels = _get_td_channels(metadata_loc, metadata)
    metadata = _update_metadata(td_channels, metadata)
    event_data = _get_event_data(event_data_loc)

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
        metadata.periods_frames["period"] == 1, "start_frame"
    ].iloc[0]
    start_time = metadata.periods_frames.loc[
        metadata.periods_frames["period"] == 1, "start_datetime_ed"
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
    databallpy_events = _get_databallpy_events(event_data)

    return event_data, metadata, databallpy_events


def load_metrica_open_event_data() -> Tuple[pd.DataFrame, Metadata]:
    """Function to load the open event data of metrica

    Returns:
        Tuple[pd.DataFrame, Metadata]: event data and metadata of the match
    """
    metadata_link = "https://raw.githubusercontent.com/metrica-sports/sample-data\
        /master/data/Sample_Game_3/Sample_Game_3_metadata.xml"
    ed_link = "https://raw.githubusercontent.com/metrica-sports/sample-data\
        /master/data/Sample_Game_3/Sample_Game_3_events.json"

    print("Downloading Metrica open event data...", end="")
    raw_ed = requests.get(ed_link).text
    raw_metadata = requests.get(metadata_link).text
    print(" Done!")

    return load_metrica_event_data(raw_ed, raw_metadata)


def _get_event_data(event_data_loc: Union[str, io.StringIO]) -> pd.DataFrame:
    """Function to load metrica event data

    Args:
        event_data_loc (Union[str, io.StringIO]): location of the event data file

    Returns:
        pd.DataFrame: event data
    """

    if isinstance(event_data_loc, str) and "{" not in event_data_loc:
        file = open(event_data_loc)
        lines = file.readlines()
        file.close()
        raw_data = "".join(str(i) for i in lines)
        soup = BeautifulSoup(raw_data, "html.parser")
    else:
        soup = BeautifulSoup(event_data_loc.strip(), "html.parser")
    events_dict = json.loads(soup.text)

    result_dict = {
        "event_id": [],
        "type_id": [],
        "databallpy_event": [],
        "period_id": [],
        "minutes": [],
        "seconds": [],
        "player_id": [],
        "player_name": [],
        "team_id": [],
        "outcome": [],
        "start_x": [],
        "start_y": [],
        "to_player_id": [],
        "to_player_name": [],
        "end_x": [],
        "end_y": [],
        "td_frame": [],
        "metrica_event": [],
    }

    check_outcome_last_event = False

    in_possession_events = ["pass", "carry", "recovery", "shot"]
    out_of_possession_events = ["fault received", "ball out", "ball lost"]

    for event in events_dict["data"]:
        result_dict["event_id"].append(event["index"])
        result_dict["type_id"].append(event["type"]["id"])
        event_name = event["type"]["name"].lower()
        result_dict["metrica_event"].append(event_name)
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
                result_dict["outcome"][-1] = 0
            else:
                result_dict["outcome"][-1] = 1
            check_outcome_last_event = False

        # set outcome for shot events
        if event_name == "shot":
            if isinstance(event["subtypes"], list):
                outcome = 0
                for sub in event["subtypes"]:
                    if sub["name"] == "GOAL":
                        outcome = 1
                        break
            else:
                subtypes = event["subtypes"]
                outcome = 1 if subtypes["name"] == "GOAL" else 0
            result_dict["outcome"].append(outcome)
        else:
            result_dict["outcome"].append(MISSING_INT)

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

    result_dict["databallpy_event"] = [None] * len(result_dict["event_id"])
    events = pd.DataFrame(result_dict)
    events["databallpy_event"] = (
        events["metrica_event"].map(metrica_databallpy_map).replace([np.nan], [None])
    )
    return events


def _get_databallpy_events(event_data: pd.DataFrame) -> dict:
    """Function to get the databallpy events from the event data

    Args:
        event_data (pd.DataFrame): event data

    Returns:
        dict: dictionary with the databallpy events
    """
    shot_events = {}
    pass_events = {}
    dribble_events = {}

    shot_mask = event_data["databallpy_event"] == "shot"
    shot_events = {
        shot.event_id: shot
        for shot in event_data[shot_mask].apply(_get_shot_event, axis=1)
    }

    pass_maks = event_data["databallpy_event"] == "pass"
    pass_events = {
        pass_.event_id: pass_
        for pass_ in event_data[pass_maks].apply(_get_pass_event, axis=1)
    }

    dribble_mask = event_data["databallpy_event"] == "dribble"
    dribble_events = {
        dribble.event_id: dribble
        for dribble in event_data[dribble_mask].apply(_get_dribble_event, axis=1)
    }

    databallpy_events = {
        "shot_events": shot_events,
        "pass_events": pass_events,
        "dribble_events": dribble_events,
    }
    return databallpy_events


def _get_shot_event(row: pd.Series) -> ShotEvent:
    """Function to return a ShotEvent object from a row of the metrica
      event data

    Args:
        row (pd.Series): row of the metrica event data with a shot event

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
        player_id=row.player_id,
        shot_outcome=["miss", "goal"][row.outcome],
        y_target=np.nan,
        z_target=np.nan,
        body_part=None,
        type_of_play=None,
        first_touch=None,
        created_oppertunity=None,
        related_event_id=MISSING_INT,
    )


def _get_pass_event(row: pd.Series) -> PassEvent:
    """Function to return a PassEvent object from a row of the metrica
     event data.

    Args:
        row (pd.Series): row of the metrica event data with a pass event

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
        player_id=row.player_id,
        outcome=["unsuccessful", "successful"][row.outcome]
        if not pd.isnull(row.outcome)
        else "not_specified",
        end_x=row.end_x,
        end_y=row.end_y,
        pass_type="not_specified",
        set_piece="unspecified_set_piece",
    )


def _get_dribble_event(row: pd.Series) -> DribbleEvent:
    """Function to return a DribbleEvent object from a row of the metrica
     event data.

    Args:
        row (pd.Series): row of the metrica event data with a dribble event

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
        player_id=row.player_id,
        related_event_id=MISSING_INT,
        duel_type=None,
        outcome=bool(row.outcome),
        has_opponent=False,
    )
