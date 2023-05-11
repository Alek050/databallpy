import os
import pickle
from typing import Tuple

import pandas as pd

from databallpy.load_data.event_data.instat import load_instat_event_data
from databallpy.load_data.event_data.metrica_event_data import (
    load_metrica_event_data,
    load_metrica_open_event_data,
)
from databallpy.load_data.event_data.opta import load_opta_event_data
from databallpy.load_data.metadata import Metadata
from databallpy.load_data.tracking_data._quality_check_tracking_data import (
    _quality_check_tracking_data,
)
from databallpy.load_data.tracking_data.inmotio import load_inmotio_tracking_data
from databallpy.load_data.tracking_data.metrica_tracking_data import (
    load_metrica_open_tracking_data,
    load_metrica_tracking_data,
)
from databallpy.load_data.tracking_data.tracab import load_tracab_tracking_data
from databallpy.match import Match
from databallpy.utils.utils import align_player_ids


def get_match(
    tracking_data_loc: str = None,
    tracking_metadata_loc: str = None,
    event_data_loc: str = None,
    event_metadata_loc: str = None,
    tracking_data_provider: str = None,
    event_data_provider: str = None,
    check_quality: bool = True,
) -> Match:
    """Function to get all information of a match given its datasources

    Args:
        tracking_data_loc (str, optional): location of the tracking data. Defaults to
        None.
        tracking_metadata_loc (str, optional): location of the metadata of the tracking
        data. Defaults to None.
        event_data_loc (str, optional): location of the event data. Defaults to None.
        event_metadata_loc (str, optional): location of the metadata of the event data.
        Defaults to None.
        tracking_data_provider (str, optional): provider of the tracking data. Defaults
        to None.
        event_data_provider (str, optional): provider of the event data. Defaults to
        None.
        check_quality (bool, optional): whether you want to check the quality of the
        tracking data. Defaults to True
    Returns:
        (Match): a Match object with all information available of the match.
    """
    uses_tracking_data = False
    uses_event_data = False

    # Check if tracking data should be loaded
    if tracking_data_loc and tracking_metadata_loc and tracking_data_provider:
        tracking_data, tracking_metadata = load_tracking_data(
            tracking_data_loc=tracking_data_loc,
            tracking_metadata_loc=tracking_metadata_loc,
            tracking_data_provider=tracking_data_provider,
        )
        uses_tracking_data = True

    # Check if event data should be loaded
    if event_data_loc and event_metadata_loc and event_data_provider:
        event_data, event_metadata = load_event_data(
            event_data_loc=event_data_loc,
            event_metadata_loc=event_metadata_loc,
            event_data_provider=event_data_provider,
        )
        uses_event_data = True

    if not uses_event_data and not uses_tracking_data:
        raise ValueError("No data loaded, please provide data locations and providers")

    if uses_tracking_data and uses_event_data:
        # Check if the event data is scaled the right way
        if not tracking_metadata.pitch_dimensions == event_metadata.pitch_dimensions:
            x_correction = (
                tracking_metadata.pitch_dimensions[0]
                / event_metadata.pitch_dimensions[0]
            )
            y_correction = (
                tracking_metadata.pitch_dimensions[1]
                / event_metadata.pitch_dimensions[1]
            )
            event_data["start_x"] *= x_correction
            event_data["start_y"] *= y_correction

        # Merge periods
        periods_cols = event_metadata.periods_frames.columns.difference(
            tracking_metadata.periods_frames.columns
        ).to_list()
        periods_cols.sort(reverse=True)
        merged_periods = pd.concat(
            (
                tracking_metadata.periods_frames,
                event_metadata.periods_frames[periods_cols],
            ),
            axis=1,
        )

        # Align player ids
        if (
            not set(event_metadata.home_players["id"])
            == set(tracking_metadata.home_players["id"])
        ) or (
            not set(event_metadata.away_players["id"])
            == set(tracking_metadata.away_players["id"])
        ):
            event_metadata = align_player_ids(event_metadata, tracking_metadata)
            full_name_id_map = dict(
                zip(
                    event_metadata.home_players["full_name"],
                    event_metadata.home_players["id"],
                )
            )
            full_name_id_map.update(
                dict(
                    zip(
                        event_metadata.away_players["full_name"],
                        event_metadata.away_players["id"],
                    )
                )
            )
            event_data["player_id"] = (
                event_data["player_name"]
                .map(full_name_id_map)
                .fillna(-999)
                .astype("int64")
            )

        # Merged player info
        player_cols = event_metadata.home_players.columns.difference(
            tracking_metadata.home_players.columns
        ).to_list()
        player_cols.append("id")
        home_players = tracking_metadata.home_players.merge(
            event_metadata.home_players[player_cols], on="id"
        )
        away_players = tracking_metadata.away_players.merge(
            event_metadata.away_players[player_cols], on="id"
        )

        if check_quality:
            _quality_check_tracking_data(
                tracking_data, tracking_metadata.frame_rate, merged_periods
            )

        match = Match(
            tracking_data=tracking_data,
            tracking_data_provider=tracking_data_provider,
            event_data=event_data,
            event_data_provider=event_data_provider,
            pitch_dimensions=tracking_metadata.pitch_dimensions,
            periods=merged_periods,
            frame_rate=tracking_metadata.frame_rate,
            home_team_id=event_metadata.home_team_id,
            home_formation=event_metadata.home_formation,
            home_score=event_metadata.home_score,
            home_team_name=event_metadata.home_team_name,
            home_players=home_players,
            away_team_id=event_metadata.away_team_id,
            away_formation=event_metadata.away_formation,
            away_score=event_metadata.away_score,
            away_team_name=event_metadata.away_team_name,
            away_players=away_players,
            country=event_metadata.country,
        )

    elif uses_tracking_data and not uses_event_data:
        if check_quality:
            _quality_check_tracking_data(tracking_data)

        match = Match(
            tracking_data=tracking_data,
            tracking_data_provider=tracking_data_provider,
            event_data=pd.DataFrame(),
            event_data_provider=None,
            pitch_dimensions=tracking_metadata.pitch_dimensions,
            periods=tracking_metadata.periods_frames,
            frame_rate=tracking_metadata.frame_rate,
            home_team_id=tracking_metadata.home_team_id,
            home_formation=tracking_metadata.home_formation,
            home_score=tracking_metadata.home_score,
            home_team_name=tracking_metadata.home_team_name,
            home_players=tracking_metadata.home_players,
            away_team_id=tracking_metadata.away_team_id,
            away_formation=tracking_metadata.away_formation,
            away_score=tracking_metadata.away_score,
            away_team_name=tracking_metadata.away_team_name,
            away_players=tracking_metadata.away_players,
            country=tracking_metadata.country,
        )

    elif uses_event_data and not uses_tracking_data:
        match = Match(
            tracking_data=pd.DataFrame(),
            tracking_data_provider=None,
            event_data=event_data,
            event_data_provider=event_data_provider,
            pitch_dimensions=event_metadata.pitch_dimensions,
            periods=event_metadata.periods_frames,
            frame_rate=event_metadata.frame_rate,
            home_team_id=event_metadata.home_team_id,
            home_formation=event_metadata.home_formation,
            home_score=event_metadata.home_score,
            home_team_name=event_metadata.home_team_name,
            home_players=event_metadata.home_players,
            away_team_id=event_metadata.away_team_id,
            away_formation=event_metadata.away_formation,
            away_score=event_metadata.away_score,
            away_team_name=event_metadata.away_team_name,
            away_players=event_metadata.away_players,
            country=event_metadata.country,
        )

    return match


def get_saved_match(name: str, path: str = os.getcwd()) -> Match:
    """Function to load a saved match object

    Args:
        name (str): the name with the to be loaded match, should be a pickle file
        path (str, optional): path of directory where Match is saved. Defaults
        to current working directory.

    Returns:
        Match: All information about the match
    """
    with open(os.path.join(path, name + ".pickle"), "rb") as f:
        match = pickle.load(f)
    return match


def load_tracking_data(
    *, tracking_data_loc: str, tracking_metadata_loc: str, tracking_data_provider: str
) -> Tuple[pd.DataFrame, Metadata]:
    """Function to load the tracking data of a match

    Args:
        tracking_data_loc (str): location of the tracking data file
        tracking_metadata_loc (str): location of the tracking metadata file
        tracking_data_provider (str): provider of the tracking data

    Returns:
        Tuple[pd.DataFrame, Metadata]: tracking data and metadata of the match
    """
    assert tracking_data_provider in [
        "tracab",
        "metrica",
        "inmotio",
    ], f"We do not support '{tracking_data_provider}' as tracking data provider yet, "
    "please open an issue in our Github repository."

    # Get tracking data and tracking metadata
    if tracking_data_provider == "tracab":
        tracking_data, tracking_metadata = load_tracab_tracking_data(
            tracking_data_loc, tracking_metadata_loc
        )
    elif tracking_data_provider == "metrica":
        tracking_data, tracking_metadata = load_metrica_tracking_data(
            tracking_data_loc=tracking_data_loc, metadata_loc=tracking_metadata_loc
        )
    elif tracking_data_provider == "inmotio":
        tracking_data, tracking_metadata = load_inmotio_tracking_data(
            tracking_data_loc=tracking_data_loc, metadata_loc=tracking_metadata_loc
        )

    return tracking_data, tracking_metadata


def load_event_data(
    *, event_data_loc: str, event_metadata_loc: str, event_data_provider: str
) -> Tuple[pd.DataFrame, Metadata]:
    """Function to load the event data of a match

    Args:
        event_data_loc (str): location of the event data file
        event_metadata_loc (str): location of the event metadata file
        event_data_provider (str): provider of the event data

    Returns:
        Tuple[pd.DataFrame, Metadata]: event data and metadata of the match
    """
    assert event_data_provider in [
        "opta",
        "metrica",
        "instat",
    ], f"We do not supper '{event_data_provider}' as event data provider yet, "
    "please open an issue in our Github repository."

    # Get event data and event metadata
    if event_data_provider == "opta":
        event_data, event_metadata = load_opta_event_data(
            f7_loc=event_metadata_loc, f24_loc=event_data_loc
        )
    elif event_data_provider == "metrica":
        event_data, event_metadata = load_metrica_event_data(
            event_data_loc=event_data_loc, metadata_loc=event_metadata_loc
        )
    elif event_data_provider == "instat":
        event_data, event_metadata = load_instat_event_data(
            event_data_loc=event_data_loc, metadata_loc=event_metadata_loc
        )

    return event_data, event_metadata


def get_open_match(provider: str = "metrica", verbose: bool = True) -> Match:
    """Function to load a match object from an open datasource

    Args:
        provider (str, optional): What provider to get the open data from.
        Defaults to "metrica".
        verbose (bool, optional): Whether or not to print info about progress
        in the terminal, Defaults to True.

    Returns:
        Match: All information about the match
    """
    assert provider in ["metrica"]

    if provider == "metrica":
        tracking_data, metadata = load_metrica_open_tracking_data(verbose=verbose)
        event_data, ed_metadata = load_metrica_open_event_data()

    periods_cols = ed_metadata.periods_frames.columns.difference(
        metadata.periods_frames.columns
    ).to_list()
    periods_cols.sort(reverse=True)
    merged_periods = pd.concat(
        (
            metadata.periods_frames,
            ed_metadata.periods_frames[periods_cols],
        ),
        axis=1,
    )

    match = Match(
        tracking_data=tracking_data,
        tracking_data_provider=provider,
        event_data=event_data,
        event_data_provider=provider,
        pitch_dimensions=metadata.pitch_dimensions,
        periods=merged_periods,
        frame_rate=metadata.frame_rate,
        home_team_id=metadata.home_team_id,
        home_formation=metadata.home_formation,
        home_score=metadata.home_score,
        home_team_name=metadata.home_team_name,
        home_players=metadata.home_players,
        away_team_id=metadata.away_team_id,
        away_formation=metadata.away_formation,
        away_score=metadata.away_score,
        away_team_name=metadata.away_team_name,
        away_players=metadata.away_players,
        country="",
    )
    return match
