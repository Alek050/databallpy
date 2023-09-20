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
from databallpy.load_data.event_data.ortec import load_ortec_event_data
from databallpy.load_data.event_data.pass_event import PassEvent
from databallpy.load_data.event_data.scisports import _handle_scisports_data
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
from databallpy.utils.align_player_ids import align_player_ids
from databallpy.utils.utils import MISSING_INT


def get_match(
    tracking_data_loc: str = None,
    tracking_metadata_loc: str = None,
    event_data_loc: str = None,
    event_metadata_loc: str = None,
    tracking_data_provider: str = None,
    event_data_provider: str = None,
    check_quality: bool = True,
    verbose: bool = True,
    _extra_event_data_loc: str = None,
) -> Match:
    """Function to get all information of a match given its datasources

    Args:
        tracking_data_loc (str, optional): location of the tracking data.
                                           Defaults to None.
        tracking_metadata_loc (str, optional): location of the metadata of the tracking
                                               data. Defaults to None.
        event_data_loc (str, optional): location of the event data. Defaults to None.
        event_metadata_loc (str, optional): location of the metadata of the event data.
                                            Defaults to None.
        tracking_data_provider (str, optional): provider of the tracking data. Defaults
                                                to None. Supported providers are:
                                                - Tracab
                                                - Metrica
                                                - Inmotio
        event_data_provider (str, optional): provider of the event data. Defaults to
                                             None. Supported providers are:
                                             - Opta
                                             - Metrica
                                             - Instat
        check_quality (bool, optional): whether you want to check the quality of the
                                        tracking data. Defaults to True
        verbose (bool, optional): whether or not to print info about progress
    Returns:
        (Match): a Match object with all information available of the match.
    """
    uses_tracking_data = False
    uses_event_data = False
    uses_event_metadata = False

    # Check if tracking data should be loaded
    if tracking_data_loc and tracking_metadata_loc and tracking_data_provider:
        tracking_data, tracking_metadata = load_tracking_data(
            tracking_data_loc=tracking_data_loc,
            tracking_metadata_loc=tracking_metadata_loc,
            tracking_data_provider=tracking_data_provider,
            verbose=verbose,
        )
        databallpy_events = {}
        uses_tracking_data = True

    # Check if event data should be loaded
    if event_data_loc and event_metadata_loc and event_data_provider:
        event_data, event_metadata, databallpy_events = load_event_data(
            event_data_loc=event_data_loc,
            event_metadata_loc=event_metadata_loc,
            event_data_provider=event_data_provider,
        )
        if event_data is not None and len(event_data) > 0:
            uses_event_data = True
        uses_event_metadata = True

    # temporary, in the case of ortec, we only have metadata
    if event_metadata_loc and event_data_provider == "ortec":
        _, event_metadata = load_ortec_event_data(
            event_data_loc=None, metadata_loc=event_metadata_loc
        )
        event_data = pd.DataFrame()
        databallpy_events = {}
        uses_event_metadata = True

    # load extra event data if needed
    if (
        _extra_event_data_loc is not None
        and "scisportsevents xml v2" in _extra_event_data_loc.lower()
    ):
        scisports_event_data, databallpy_events, tracking_data = _handle_scisports_data(
            scisports_ed_loc=_extra_event_data_loc,
            tracking_data=tracking_data if uses_tracking_data else None,
            event_metadata=event_metadata if uses_event_metadata else None,
            tracking_metadata=tracking_metadata if uses_tracking_data else None,
            databallpy_events=databallpy_events
            if "databallpy_events" in vars()
            else None,
            verbose=verbose,
        )
        uses_event_data = True
        event_data = (
            scisports_event_data
            if ("event_data" not in vars() or len(event_data) == 0)
            else event_data
        )

    if not uses_event_data and not uses_tracking_data:
        raise ValueError("No data loaded, please provide data locations and providers")

    # extra checks when using event data
    if uses_event_data:
        if "scisports_event_data" in vars():
            extra_data = (
                scisports_event_data
                if not event_data.equals(scisports_event_data)
                else None
            )
        else:
            extra_data = None
        home_players = event_metadata.home_players
        away_players = event_metadata.away_players
        periods = event_metadata.periods_frames
        pitch_dimensions = event_metadata.pitch_dimensions

    # extra checks when using tracking data
    if uses_tracking_data:
        home_players = tracking_metadata.home_players
        away_players = tracking_metadata.away_players
        periods = tracking_metadata.periods_frames
        pitch_dimensions = tracking_metadata.pitch_dimensions

    # extra checks when using both tracking and event data
    if uses_tracking_data and uses_event_metadata:
        periods = merge_metadata_periods(
            tracking_metadata.periods_frames, event_metadata.periods_frames
        )
        if uses_event_data:
            event_data, databallpy_events = rescale_event_data(
                tracking_metadata.pitch_dimensions,
                event_metadata.pitch_dimensions,
                event_data,
                databallpy_events,
            )
            event_data, event_metadata = align_player_and_team_ids(
                event_data, event_metadata, tracking_metadata
            )
        home_players, away_players = merge_player_info(
            tracking_metadata, event_metadata
        )
        pitch_dimensions = tracking_metadata.pitch_dimensions

    # check quality of tracking data
    allow_synchronise = False
    if check_quality and uses_tracking_data:
        allow_synchronise = _quality_check_tracking_data(
            tracking_data, tracking_metadata.frame_rate, periods
        )
        allow_synchronise = False if not uses_event_data else allow_synchronise

    match = Match(
        tracking_data=tracking_data if uses_tracking_data else pd.DataFrame(),
        tracking_data_provider=tracking_data_provider if uses_tracking_data else None,
        event_data=event_data if uses_event_data else pd.DataFrame(),
        event_data_provider=event_data_provider if uses_event_data else None,
        pitch_dimensions=pitch_dimensions,
        periods=periods,
        frame_rate=tracking_metadata.frame_rate if uses_tracking_data else MISSING_INT,
        home_team_id=event_metadata.home_team_id
        if uses_event_metadata
        else tracking_metadata.home_team_id,
        home_formation=event_metadata.home_formation
        if uses_event_metadata
        else tracking_metadata.home_formation,
        home_score=event_metadata.home_score
        if uses_event_metadata
        else tracking_metadata.home_score,
        home_team_name=event_metadata.home_team_name
        if uses_event_metadata
        else tracking_metadata.home_team_name,
        home_players=home_players,
        away_team_id=event_metadata.away_team_id
        if uses_event_metadata
        else tracking_metadata.away_team_id,
        away_formation=event_metadata.away_formation
        if uses_event_metadata
        else tracking_metadata.away_formation,
        away_score=event_metadata.away_score
        if uses_event_metadata
        else tracking_metadata.away_score,
        away_team_name=event_metadata.away_team_name
        if uses_event_metadata
        else tracking_metadata.away_team_name,
        away_players=away_players,
        country=event_metadata.country
        if uses_event_metadata
        else tracking_metadata.country,
        allow_synchronise_tracking_and_event_data=allow_synchronise,
        shot_events=databallpy_events["shot_events"]
        if "shot_events" in databallpy_events.keys()
        else {},
        dribble_events=databallpy_events["dribble_events"]
        if "dribble_events" in databallpy_events.keys()
        else {},
        pass_events=databallpy_events["pass_events"]
        if "pass_events" in databallpy_events.keys()
        else {},
        extra_data=extra_data if "extra_data" in vars() else None,
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
    *,
    tracking_data_loc: str,
    tracking_metadata_loc: str,
    tracking_data_provider: str,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Metadata]:
    """Function to load the tracking data of a match

    Args:
        tracking_data_loc (str): location of the tracking data file
        tracking_metadata_loc (str): location of the tracking metadata file
        tracking_data_provider (str): provider of the tracking data
        verbose (bool, optional): whether or not to print info about progress

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
            tracking_data_loc, tracking_metadata_loc, verbose=verbose
        )
    elif tracking_data_provider == "metrica":
        tracking_data, tracking_metadata = load_metrica_tracking_data(
            tracking_data_loc=tracking_data_loc,
            metadata_loc=tracking_metadata_loc,
            verbose=verbose,
        )
    elif tracking_data_provider == "inmotio":
        tracking_data, tracking_metadata = load_inmotio_tracking_data(
            tracking_data_loc=tracking_data_loc,
            metadata_loc=tracking_metadata_loc,
            verbose=verbose,
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
    ], f"We do not support '{event_data_provider}' as event data provider yet, "
    "please open an issue in our Github repository."

    # Get event data and event metadata
    databallpy_events = {}
    if event_data_provider == "opta":
        event_data, event_metadata, databallpy_events = load_opta_event_data(
            f7_loc=event_metadata_loc, f24_loc=event_data_loc
        )
    elif event_data_provider == "metrica":
        event_data, event_metadata, databallpy_events = load_metrica_event_data(
            event_data_loc=event_data_loc, metadata_loc=event_metadata_loc
        )
    elif event_data_provider == "instat":
        event_data, event_metadata = load_instat_event_data(
            event_data_loc=event_data_loc, metadata_loc=event_metadata_loc
        )

    return event_data, event_metadata, databallpy_events


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
        event_data, ed_metadata, databallpy_events = load_metrica_open_event_data()

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
        allow_synchronise_tracking_and_event_data=True,
        shot_events=databallpy_events["shot_events"]
        if "shot_events" in databallpy_events.keys()
        else {},
        dribble_events=databallpy_events["dribble_events"]
        if "dribble_events" in databallpy_events.keys()
        else {},
        pass_events=databallpy_events["pass_events"]
        if "pass_events" in databallpy_events.keys()
        else {},
    )
    return match


def merge_metadata_periods(
    tracking_periods: pd.DataFrame, event_periods: pd.DataFrame
) -> pd.DataFrame:
    """Function to merge the periods of the event and tracking metadata

    Args:
        tracking_periods (pd.DataFrame): periods of the tracking metadata
        event_periods (pd.DataFrame): periods of the event metadata

    Returns:
        pd.DataFrame: merged periods
    """

    periods_cols = event_periods.columns.difference(tracking_periods.columns).to_list()
    periods_cols.sort(reverse=True)
    merged_periods = pd.concat(
        (
            tracking_periods,
            event_periods[periods_cols],
        ),
        axis=1,
    )
    return merged_periods


def rescale_event_data(
    tracking_pitch_dimensions: list,
    event_pitch_dimensions: list,
    event_data: pd.DataFrame,
    databallpy_events: dict,
) -> Tuple[pd.DataFrame, dict]:
    """Function to rescale the event data and databallpy events to the tracking data
    dimensions if the event data is not scaled in the same dimensions of the tracking
    data.

    Args:
        tracking_pitch_dimensions (list): pitch dimensions of the tracking data
        event_pitch_dimensions (list): pitch dimensions of the event data
        event_data (pd.DataFrame): event data
        databallpy_events (dict): databallpy events

    Returns:
        Tuple[pd.DataFrame, dict]: rescaled event data and databallpy events
    """
    if not (
        not tracking_pitch_dimensions == event_pitch_dimensions
        and not pd.isnull(event_pitch_dimensions).any()
        and not pd.isnull(tracking_pitch_dimensions).any()
    ):
        return event_data, databallpy_events

    x_correction = tracking_pitch_dimensions[0] / event_pitch_dimensions[0]
    y_correction = tracking_pitch_dimensions[1] / event_pitch_dimensions[1]
    event_data["start_x"] *= x_correction
    event_data["start_y"] *= y_correction

    # correct the databallpy event instances as well
    if databallpy_events is not None:
        for dict_of_events in databallpy_events.values():
            for event in dict_of_events.values():
                event.start_x *= x_correction
                event.start_y *= y_correction
                if isinstance(event, PassEvent):
                    event.end_x *= x_correction
                    event.end_y *= y_correction

    return event_data, databallpy_events


def align_player_and_team_ids(
    event_data: pd.DataFrame, event_metadata: Metadata, tracking_metadata: Metadata
) -> pd.DataFrame:
    """Function to align the player and team id's of the event data with the tracking
    data.

    Args:
        event_data (pd.DataFrame): event data
        event_metadata (Metadata): event metadata
        tracking_metadata (Metadata): tracking metadata

    Returns:
        pd.DataFrame: event data with aligned player and team id's
    """

    # check player_id's
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
            .fillna(MISSING_INT)
            .astype("int64")
        )
    # check team_id's
    if not event_metadata.home_team_id == tracking_metadata.home_team_id:
        event_home_team_id = event_metadata.home_team_id
        tracking_home_team_id = tracking_metadata.home_team_id
        event_data["team_id"] = event_data["team_id"].replace(
            {event_home_team_id: tracking_home_team_id}
        )
        if type(tracking_home_team_id) is not type(event_home_team_id):
            if MISSING_INT in event_data["team_id"].unique():
                event_data.loc[event_data["team_id"] == MISSING_INT, "team_id"] = None
            else:
                event_data.loc[
                    pd.isnull(event_data["team_id"]), "team_id"
                ] = MISSING_INT
    if not event_metadata.away_team_id == tracking_metadata.away_team_id:
        event_away_team_id = event_metadata.away_team_id
        tracking_away_team_id = tracking_metadata.away_team_id
        event_data["team_id"] = event_data["team_id"].replace(
            {event_away_team_id: tracking_away_team_id}
        )

    return event_data, event_metadata


def merge_player_info(
    tracking_metadata: Metadata, event_metadata: Metadata
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Function to merge the player information of the tracking and event metadata

    Args:
        tracking_metadata (Metadata): metadata of the tracking data
        event_metadata (Metadata): metadata of the event data

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: merged home player information
            and merged away player information.
    """

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

    return home_players, away_players
