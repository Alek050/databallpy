import io
import os

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from databallpy.data_parsers import Metadata
from databallpy.data_parsers.metrica_metadata_parser import (
    _get_metadata,
    _get_td_channels,
    _update_metadata,
)
from databallpy.data_parsers.tracking_data_parsers.utils import (
    _add_ball_data_to_dict,
    _add_datetime,
    _add_periods_to_tracking_data,
    _add_player_tracking_data_to_dict,
    _downcast_tracking_data,
    _get_frame_selection,
    _get_gametime,
    _get_period_start_frames,
    _insert_missing_rows,
    _normalize_playing_direction_tracking,
)
from databallpy.utils.logging import logging_wrapper
from databallpy.utils.utils import _to_int


@logging_wrapper(__file__)
def load_metrica_tracking_data(
    tracking_data_loc: str,
    metadata_loc: str,
    verbose: bool = True,
    *,
    period_id: int | list[int] | None = None,
    frames: tuple[int, int] | None = None,
) -> tuple[pd.DataFrame, Metadata]:
    """Function to load metrica tracking data.

    Args:
        tracking_data_loc (str): location of the tracking data .txt file
        metadata_loc (str): location of the metadata .xml file
        verbose (bool, optional): whether to print information about the progress
        in the terminall. Defaults to True.
        period_id (int | list[int], optional): only load the frames of these
            period(s). Defaults to None (load all periods).
        frames (tuple[int, int], optional): only load the frames between the first
            and the last frame, inclusive. Defaults to None (load all frames).

    Raises:
        TypeError: if tracking_data_loc is not a string or io.StringIO

    Returns:
        Tuple[pd.DataFrame, Metadata]: tracking and metadata of the game

    Note:
        The metadata always describes the full game, also when only a part of the
        tracking data is loaded.
    """

    if isinstance(tracking_data_loc, str):
        if not os.path.exists(tracking_data_loc):
            message = f"Could not find {tracking_data_loc}"
            raise FileNotFoundError(message)
        if not os.path.exists(metadata_loc):
            message = f"Could not find {metadata_loc}"
            raise FileNotFoundError(message)

    elif isinstance(tracking_data_loc, io.StringIO):
        # open tracking data, downloaded from internet.
        pass
    else:
        raise TypeError(
            "tracking_data_loc must be either a str or a StringIO object,"
            f" not a {type(tracking_data_loc)}"
        )

    metadata = _get_metadata(metadata_loc)
    td_channels = _get_td_channels(metadata_loc, metadata)
    metadata = _update_metadata(td_channels, metadata)
    selection = _get_frame_selection(metadata.periods_frames, period_id, frames)
    tracking_data = _get_tracking_data(
        tracking_data_loc,
        td_channels,
        metadata.pitch_dimensions,
        verbose=verbose,
        selection=selection,
        extra_frames=_get_period_start_frames(metadata.periods_frames),
    )

    tracking_data, changed_periods = _normalize_playing_direction_tracking(
        tracking_data, metadata.periods_frames
    )
    metadata.periods_changed_playing_direction = changed_periods

    tracking_data["datetime"] = _add_datetime(
        tracking_data["frame"],
        metadata.frame_rate,
        metadata.periods_frames["start_datetime_td"].iloc[0],
    )
    tracking_data["period_id"] = _add_periods_to_tracking_data(
        tracking_data["frame"], metadata.periods_frames
    )
    tracking_data["gametime_td"] = _get_gametime(
        tracking_data["frame"], tracking_data["period_id"], metadata
    )

    if selection is not None:
        tracking_data = tracking_data[
            tracking_data["frame"].between(*selection)
        ].reset_index(drop=True)

    return _downcast_tracking_data(tracking_data), metadata


@logging_wrapper(__file__)
def load_metrica_open_tracking_data(
    verbose: bool = True,
) -> tuple[pd.DataFrame, Metadata]:
    """Function to load open dataset of metrica

    Args:
        verbose (bool): Whether or not to print info in the terminal. Defaults to True.
        cache_path (Path): path to save the tracking file to as cache.
    Returns:
        Tuple[pd.DataFrame, Metadata]: tracking and metadata of the game
    """
    td_data_link = "https://raw.githubusercontent.com/metrica-sports/sample-data\
        /master/data/Sample_Game_3/Sample_Game_3_tracking.txt"
    td_metadata_link = "https://raw.githubusercontent.com/metrica-sports/sample-data\
        /master/data/Sample_Game_3/Sample_Game_3_metadata.xml"

    if verbose:
        print("Downloading Metrica open tracking data...", end="")
    td_data = io.StringIO(requests.get(td_data_link).text)
    td_metadata = requests.get(td_metadata_link).text
    if verbose:
        print(" Done!")
    return load_metrica_tracking_data(
        tracking_data_loc=td_data, metadata_loc=td_metadata, verbose=verbose
    )


@logging_wrapper(__file__)
def _get_tracking_data(
    tracking_data_loc: str | io.StringIO,
    channels: dict,
    pitch_dimensions: list[float, float],
    verbose: bool = True,
    *,
    selection: tuple[int, int] | None = None,
    extra_frames: set[int] | None = None,
) -> pd.DataFrame:
    """Function to load the tracking data of metrica.

    Args:
        tracking_data_loc (Union[str, io.StringIO]): location of the tracking data .txt
        file channels (dict): dictionary with for all frames the order of which
        players are referred to in the raw tracking data
        pitch_dimensions (list): x and y dimensions of the pitch in meters
        verbose (bool, optional): whether to print information about the progress in the
        terminal. Defaults to True.
        selection (tuple[int, int], optional): the first and last frame to load,
            inclusive. Defaults to None (load all frames).
        extra_frames (set[int], optional): frames that are loaded even when they
            fall outside of the selection. Defaults to None.

    Returns:
        pd.DataFrame: tracking data of the game in a pd dataframe
    """

    if isinstance(tracking_data_loc, str):
        file = open(tracking_data_loc)
        lines = _filter_lines(file, selection, extra_frames)
        file.close()
    else:
        lines = _filter_lines(tracking_data_loc, selection, extra_frames)

    size_lines = len(lines)
    data = {
        "frame": [np.nan] * size_lines,
        "ball_x": [np.nan] * size_lines,
        "ball_y": [np.nan] * size_lines,
        "ball_z": [np.nan] * size_lines,
        "ball_status": [None] * size_lines,
        "team_possession": [None] * size_lines,
    }

    if verbose:
        lines = tqdm(
            lines, desc="Writing lines to dataframe", unit=" lines", leave=False
        )

    for idx, line in enumerate(lines):
        frame, players_info, ball_info = line.split(":")
        frame = _to_int(frame)
        data["frame"][idx] = frame

        channel = channels.loc[
            (channels["start"] <= frame) & (channels["end"] >= frame),
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

    df = _insert_missing_rows(df, "frame", selection=selection)

    return _downcast_tracking_data(df)


def _filter_lines(
    file: io.TextIOBase,
    selection: tuple[int, int] | None,
    extra_frames: set[int] | None,
) -> list[str]:
    """Function to read the lines of the tracking data that should be parsed.

    Args:
        file (io.TextIOBase): the opened tracking data file
        selection (tuple[int, int] | None): the first and last frame to load,
            inclusive. None means all frames are loaded.
        extra_frames (set[int] | None): frames that are loaded even when they fall
            outside of the selection.

    Returns:
        list[str]: the lines of the tracking data that should be parsed
    """
    if selection is None:
        return file.readlines()

    first_frame, last_frame = selection
    extra_frames = extra_frames if extra_frames is not None else set()
    return [
        line
        for line in file
        if first_frame <= (frame := _to_int(line.split(":", 1)[0])) <= last_frame
        or frame in extra_frames
    ]
