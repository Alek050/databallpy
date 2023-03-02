import io
import json
import os
from typing import Tuple, Union

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from databallpy.load_data.metadata import Metadata
from databallpy.load_data.metrica_metadata import (
    _get_metadata,
    _get_td_channels,
    _update_metadata,
)
from databallpy.utils import _to_float, _to_int


def load_metrica_event_data(
    event_data_loc: str, metadata_loc: str
) -> Tuple[pd.DataFrame, Metadata]:
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

    metadata = _get_metadata(metadata_loc)
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
        metadata.periods_frames["period"] == 1, "start_time_td"
    ].iloc[0]
    frame_rate = metadata.frame_rate
    event_data["datetime"] = pd.to_datetime(start_time) + pd.to_timedelta(
        (event_data["td_frame"] - first_frame) / frame_rate, unit="s"
    )

    return event_data, metadata


def load_metrica_open_event_data() -> Tuple[pd.DataFrame, Metadata]:

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
        "event": [],
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
    }

    for event in events_dict["data"]:
        result_dict["event_id"].append(event["index"])
        result_dict["type_id"].append(event["type"]["id"])
        event_name = event["type"]["name"].lower()
        result_dict["event"].append(event_name)
        result_dict["period_id"].append(event["period"])
        result_dict["minutes"].append(_to_int((event["start"]["time"] // 60)))
        result_dict["seconds"].append(_to_float(event["start"]["time"] % 60))
        result_dict["player_id"].append(_to_int(event["from"]["id"][1:]))
        result_dict["player_name"].append(event["from"]["name"])
        result_dict["team_id"].append(event["team"]["id"])
        result_dict["outcome"].append(np.nan)
        result_dict["start_x"].append(_to_float(event["start"]["x"]))
        result_dict["start_y"].append(_to_float(event["start"]["y"]))
        if event["to"] is not None:
            result_dict["to_player_id"].append(_to_int(event["to"]["id"][1:]))
            result_dict["to_player_name"].append(event["to"]["name"])
        else:
            result_dict["to_player_id"].append(np.nan)
            result_dict["to_player_name"].append(None)
        result_dict["end_x"].append(_to_float(event["end"]["x"]))
        result_dict["end_y"].append(_to_float(event["end"]["y"]))
        result_dict["td_frame"].append(event["start"]["frame"])

    events = pd.DataFrame(result_dict)
    return events
