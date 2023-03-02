import io
import os
from typing import Tuple, Union

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from databallpy.load_data.metadata import Metadata
from databallpy.load_data.metrica_metadata import (
    _get_metadata,
    _get_td_channels,
    _update_metadata,
)
from databallpy.load_data.tracking_data._add_ball_data_to_dict import (
    _add_ball_data_to_dict,
)
from databallpy.load_data.tracking_data._add_player_tracking_data_to_dict import (
    _add_player_tracking_data_to_dict,
)
from databallpy.load_data.tracking_data._get_matchtime import _get_matchtime
from databallpy.load_data.tracking_data._insert_missing_rows import _insert_missing_rows


def load_metrica_tracking_data(
    tracking_data_loc: str, metadata_loc: str, verbose: bool = True
) -> Tuple[pd.DataFrame, Metadata]:
    """Function to load metrica tracking data.

    Args:
        tracking_data_loc (str): location of the tracking data .txt file
        metadata_loc (str): location of the metadata .xml file
        verbose (bool, optional): whether to print information about the progress
        in the terminall. Defaults to True.

    Raises:
        TypeError: if tracking_data_loc is not a string or io.StringIO

    Returns:
        Tuple[pd.DataFrame, Metadata]: tracking and metadata of the match
    """

    if isinstance(tracking_data_loc, str):
        assert os.path.exists(tracking_data_loc)
        assert os.path.exists(metadata_loc)
    elif isinstance(tracking_data_loc, io.StringIO):
        pass
    else:
        raise TypeError(
            "tracking_data_loc must be either a str or a StringIO object,"
            f" not a {type(tracking_data_loc)}"
        )

    metadata = _get_metadata(metadata_loc)
    td_channels = _get_td_channels(metadata_loc, metadata)
    metadata = _update_metadata(td_channels, metadata)
    tracking_data = _get_tracking_data(
        tracking_data_loc, td_channels, metadata.pitch_dimensions, verbose=verbose
    )
    tracking_data["matchtime_td"] = _get_matchtime(tracking_data["timestamp"], metadata)

    return tracking_data, metadata


def load_metrica_open_tracking_data() -> Tuple[pd.DataFrame, Metadata]:
    """Function to load open dataset of metrica

    Returns:
        Tuple[pd.DataFrame, Metadata]: tracking and metadata of the match
    """
    td_data_link = "https://raw.githubusercontent.com/metrica-sports/sample-data\
        /master/data/Sample_Game_3/Sample_Game_3_tracking.txt"
    td_metadata_link = "https://raw.githubusercontent.com/metrica-sports/sample-data\
        /master/data/Sample_Game_3/Sample_Game_3_metadata.xml"

    print("Downloading Metrica open tracking data...", end="")
    td_data = io.StringIO(requests.get(td_data_link).text)
    td_metadata = requests.get(td_metadata_link).text
    print(" Done!")
    return load_metrica_tracking_data(
        tracking_data_loc=td_data, metadata_loc=td_metadata
    )


def _get_tracking_data(
    tracking_data_loc: Union[str, io.StringIO],
    channels: dict,
    pitch_dimensions: list,
    verbose: bool = True,
) -> pd.DataFrame:
    """Function to load the tracking data of metrica.

    Args:
        tracking_data_loc (Union[str, io.StringIO]): location of the tracking data .txt
        file channels (dict): dictionary with for all timestamps the order of which
        players are referred to in the raw tracking data
        pitch_dimensions (list): x and y dimensions of the pitch in meters
        verbose (bool, optional): whether to print information about the progress in the
        terminal. Defaults to True.

    Returns:
        pd.DataFrame: tracking data of the match in a pd dataframe
    """

    if isinstance(tracking_data_loc, str):
        file = open(tracking_data_loc)
        lines = file.readlines()
        file.close()
    else:
        lines = tracking_data_loc.readlines()

    size_lines = len(lines)
    data = {
        "timestamp": [np.nan] * size_lines,
        "ball_x": [np.nan] * size_lines,
        "ball_y": [np.nan] * size_lines,
        "ball_z": [np.nan] * size_lines,
        "ball_status": [None] * size_lines,
        "ball_posession": [None] * size_lines,
    }

    if verbose:
        print("Writing lines to dataframe:")
        lines = tqdm(lines)

    for idx, line in enumerate(lines):

        timestamp, players_info, ball_info = line.split(":")
        timestamp = int(timestamp)
        data["timestamp"][idx] = timestamp

        channel = channels.loc[
            (channels["start"] <= timestamp) & (channels["end"] >= timestamp),
            "ids",
        ].iloc[0]

        players = players_info.split(";")
        for i, player in enumerate(players):
            x, y = player.split(",")
            team = channel[i].split("_")[0]
            shirt_num = channel[i].split("_")[1]
            data = _add_player_tracking_data_to_dict(team, shirt_num, x, y, data, idx)

        x, y = ball_info.split(",")
        data = _add_ball_data_to_dict(x, y, np.nan, None, None, data, idx)

    df = pd.DataFrame(data)
    df["ball_status"] = np.where(pd.isnull(df["ball_x"]), "dead", "alive")

    for col in [x for x in df.columns if "_x" in x]:
        df[col] = df[col] * pitch_dimensions[0] - (pitch_dimensions[0] / 2)

    for col in [x for x in df.columns if "_y" in x]:
        df[col] = df[col] * pitch_dimensions[1] - (pitch_dimensions[1] / 2)

    df = _insert_missing_rows(df, "timestamp")

    return df
