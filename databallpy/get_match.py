import os
import pickle

import pandas as pd

from databallpy.data_parsers import Metadata
from databallpy.data_parsers.event_data_parsers import (
    load_instat_event_data,
    load_metrica_event_data,
    load_metrica_open_event_data,
    load_opta_event_data,
    load_scisports_event_data,
)
from databallpy.data_parsers.tracking_data_parsers import (
    load_inmotio_tracking_data,
    load_metrica_open_tracking_data,
    load_metrica_tracking_data,
    load_tracab_tracking_data,
)
from databallpy.data_parsers.tracking_data_parsers.utils import (
    _quality_check_tracking_data,
)
from databallpy.events import IndividualCloseToBallEvent, PassEvent
from databallpy.match import Match
from databallpy.utils.align_player_ids import align_player_ids
from databallpy.utils.constants import MISSING_INT
from databallpy.utils.logging import create_logger

LOGGER = create_logger(__name__)


def get_match(
    tracking_data_loc: str = None,
    tracking_metadata_loc: str = None,
    event_data_loc: str = None,
    event_metadata_loc: str = None,
    tracking_data_provider: str = None,
    event_data_provider: str = None,
    check_quality: bool = True,
    verbose: bool = True,
) -> Match:
    """
    Function to get all information of a match given its datasources

    Args:
        tracking_data_loc (str, optional): location of the tracking data.
            Defaults to None.
        tracking_metadata_loc (str, optional): location of the metadata of the tracking
            data. Defaults to None.
        event_data_loc (str, optional): location of the event data. Defaults to None.
        event_metadata_loc (str, optional): location of the metadata of the event data.
            Defaults to None.
        tracking_data_provider (str, optional): provider of the tracking data. Defaults
            to None. Supported providers are [tracab, metrica, inmotio]
        event_data_provider (str, optional): provider of the event data. Defaults to
            None. Supported providers are [opta, metrica, instat, scisports]
        check_quality (bool, optional): whether you want to check the quality of the
            tracking data. Defaults to True
        verbose (bool, optional): whether or not to print info about progress

    Returns:
        (Match): a Match object with all information available of the match.

    """
    try:
        LOGGER.info(
            "Trying to load a new match in get_match();"
            f"\n\tTracking data loc: {str(tracking_data_loc)}"
            f"\n\tTracking data provider: {str(tracking_data_provider)}"
            f"\n\tEvent data loc: {str(event_data_loc)}"
            f"\n\tEven data provider: {str(event_data_provider)}"
            f"\n\tCheck quality: {str(check_quality)}"
            f"\n\tVerbose: {str(verbose)}"
        )
        if (
            event_data_loc
            and event_metadata_loc is None
            and not event_data_provider == "scisports"
        ):
            LOGGER.error(
                "Event metadata location is None while event data location is not"
            )
            raise ValueError(
                "Please provide an event metadata location when providing an event"
                " data location"
            )
        elif event_data_loc and event_data_provider is None:
            LOGGER.error("Event data provider is None while event data location is not")
            raise ValueError(
                "Please provide an event data provider when providing an event"
                " data location"
            )
        elif event_metadata_loc and event_data_provider is None:
            LOGGER.error(
                "Event data provider is None while event metadata location is not"
            )
            raise ValueError(
                "Please provide an event data provider when providing an event"
                " metadata location"
            )
        elif tracking_data_loc and tracking_data_provider is None:
            LOGGER.error(
                "Tracking data provider is None while tracking data location is not"
            )
            raise ValueError(
                "Please provide a tracking data provider when providing a tracking"
                " data location"
            )
        elif tracking_data_loc and tracking_metadata_loc is None:
            LOGGER.error(
                "Tracking metadata location is None while tracking data location is not"
            )
            raise ValueError(
                "Please provide a tracking metadata location when providing a tracking"
                " data location"
            )

        uses_tracking_data = False
        uses_event_data = False

        tracking_precise_timestamps = {
            "tracab": True,
            "metrica": True,
            "inmotio": False,
        }

        event_precise_timestamps = {
            "opta": True,
            "metrica": True,
            "instat": False,
            "scisports": False,
        }

        LOGGER.info(
            "Succesfully passed input checks. Attempting to load the base "
            "data (get_match())."
        )

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
        if event_data_loc and event_data_provider:
            event_data, event_metadata, databallpy_events = load_event_data(
                event_data_loc=event_data_loc,
                event_metadata_loc=event_metadata_loc,
                event_data_provider=event_data_provider,
            )
            uses_event_data = True

        if not uses_event_data and not uses_tracking_data:
            LOGGER.error("Neither event nor tracking data was loaded.")
            raise ValueError(
                "No data loaded, please provide data locations and providers"
            )

        LOGGER.info(
            f"Loaded data in get_match():\n\tTracking data: {str(uses_tracking_data)}"
            f"\n\tEvent data: {str(uses_event_data)}"
        )

        # extra checks when using both tracking and event data
        LOGGER.info("Combining info from tracking and event data in get_match()")
        if uses_tracking_data and uses_event_data:
            tracking_metadata.periods_frames = merge_metadata_periods(
                tracking_metadata.periods_frames, event_metadata.periods_frames
            )
            event_data, databallpy_events = rescale_event_data(
                tracking_metadata.pitch_dimensions,
                event_metadata.pitch_dimensions,
                event_data,
                databallpy_events,
            )
            tracking_metadata = align_player_and_team_ids(
                event_metadata, tracking_metadata
            )
            (
                event_metadata.home_players,
                event_metadata.away_players,
            ) = merge_player_info(tracking_metadata, event_metadata)

        # check quality of tracking data
        allow_synchronise = False
        if check_quality and uses_tracking_data:
            allow_synchronise = _quality_check_tracking_data(
                tracking_data,
                tracking_metadata.frame_rate,
                tracking_metadata.periods_frames,
            )
            allow_synchronise = False if not uses_event_data else allow_synchronise

        changed_periods = None
        if uses_tracking_data:
            changed_periods = tracking_metadata.periods_changed_playing_direction

        LOGGER.info("Creating match object in get_match()")
        match = Match(
            tracking_data=tracking_data if uses_tracking_data else pd.DataFrame(),
            tracking_data_provider=tracking_data_provider
            if uses_tracking_data
            else None,
            event_data=event_data if uses_event_data else pd.DataFrame(),
            event_data_provider=event_data_provider if uses_event_data else None,
            pitch_dimensions=tracking_metadata.pitch_dimensions
            if uses_tracking_data
            else event_metadata.pitch_dimensions,
            periods=tracking_metadata.periods_frames
            if uses_tracking_data
            else event_metadata.periods_frames,
            frame_rate=tracking_metadata.frame_rate
            if uses_tracking_data
            else MISSING_INT,
            home_team_id=event_metadata.home_team_id
            if uses_event_data
            else tracking_metadata.home_team_id,
            home_formation=event_metadata.home_formation
            if uses_event_data
            else tracking_metadata.home_formation,
            home_score=event_metadata.home_score
            if uses_event_data
            else tracking_metadata.home_score,
            home_team_name=event_metadata.home_team_name
            if uses_event_data
            else tracking_metadata.home_team_name,
            home_players=event_metadata.home_players
            if uses_event_data
            else tracking_metadata.home_players,
            away_team_id=event_metadata.away_team_id
            if uses_event_data
            else tracking_metadata.away_team_id,
            away_formation=event_metadata.away_formation
            if uses_event_data
            else tracking_metadata.away_formation,
            away_score=event_metadata.away_score
            if uses_event_data
            else tracking_metadata.away_score,
            away_team_name=event_metadata.away_team_name
            if uses_event_data
            else tracking_metadata.away_team_name,
            away_players=event_metadata.away_players
            if uses_event_data
            else tracking_metadata.away_players,
            country=event_metadata.country
            if uses_event_data
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
            other_events=databallpy_events["other_events"]
            if "other_events" in databallpy_events.keys()
            else {},
            _event_timestamp_is_precise=event_precise_timestamps[event_data_provider]
            if uses_event_data
            else False,
            _tracking_timestamp_is_precise=tracking_precise_timestamps[
                tracking_data_provider
            ]
            if uses_tracking_data
            else False,
            _periods_changed_playing_direction=changed_periods,
        )
        LOGGER.info(f"Succesfully created match object: {match.name}")
        return match
    except Exception as e:
        LOGGER.exception(f"Found a unexpected exception in get_match(): \n{e}")
        raise e


def get_saved_match(name: str, path: str = os.getcwd()) -> Match:
    """Function to load a saved match object

    Args:
        name (str): the name with the to be loaded match, should be a pickle file
        path (str, optional): path of directory where Match is saved. Defaults
        to current working directory.

    Returns:
        Match: All information about the match
    """
    LOGGER.info(f"Trying to load saved match: {name} in get_saved_match()")

    loc = (
        os.path.join(path, name + ".pickle")
        if not name[-7:] == ".pickle"
        else os.path.join(path, name)
    )
    if not os.path.exists(loc):
        LOGGER.error(f"Can not load {loc} because it does not exist.")
        raise FileNotFoundError(
            f"Could not find {loc}. Set the `path` variable"
            " to specify the right directory of the saved match."
        )
    with open(loc, "rb") as f:
        match = pickle.load(f)

    if not isinstance(match, Match):
        LOGGER.critical(
            f"Loaded pickle file was not a match object but a {type(match)}. "
            "Check if this is not a Virus! (get_saved_match())"
        )
        raise TypeError(
            f"Expected a databallpy.Match object, but loaded a {type(match)}."
            " Insert the location of a match object to load a match."
        )
    LOGGER.info(f"Succesfully loaded match {match.name}")
    return match


def load_tracking_data(
    *,
    tracking_data_loc: str,
    tracking_metadata_loc: str,
    tracking_data_provider: str,
    verbose: bool = True,
) -> tuple[pd.DataFrame, Metadata]:
    """Function to load the tracking data of a match

    Args:
        tracking_data_loc (str): location of the tracking data file
        tracking_metadata_loc (str): location of the tracking metadata file
        tracking_data_provider (str): provider of the tracking data
        verbose (bool, optional): whether or not to print info about progress

    Returns:
        Tuple[pd.DataFrame, Metadata]: tracking data and metadata of the match
    """
    LOGGER.info("Trying to load tracking in load_tracking_data()")

    if tracking_data_provider not in ["tracab", "metrica", "inmotio"]:
        LOGGER.error(
            f"Found invalid tracking data provider: {tracking_data_provider} in "
            "load_tracking_data()."
        )
        raise ValueError(
            "We do not support '{tracking_data_provider}' as tracking data provider yet"
            ", please open an issue in our Github repository."
        )

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
    LOGGER.info(
        f"Successfully loaded {tracking_data_provider} "
        "tracking data in load_tracking_data()"
    )
    return tracking_data, tracking_metadata


def load_event_data(
    *, event_data_loc: str, event_metadata_loc: str, event_data_provider: str
) -> tuple[pd.DataFrame, Metadata]:
    """Function to load the event data of a match

    Args:
        event_data_loc (str): location of the event data file
        event_metadata_loc (str): location of the event metadata file
        event_data_provider (str): provider of the event data

    Returns:
        Tuple[pd.DataFrame, Metadata]: event data and metadata of the match
    """

    LOGGER.info("Trying to load event data in load_event_data()")
    if event_data_provider not in ["opta", "metrica", "instat", "scisports"]:
        LOGGER.error(
            f"Found invalid tracking data provider: {event_data_provider} in "
            "load_tracking_data()."
        )
        raise ValueError(
            f"We do not support '{event_data_provider}' as event data provider yet, "
            "please open an issue in our Github repository."
        )

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
        event_data, event_metadata, _ = load_instat_event_data(
            event_data_loc=event_data_loc, metadata_loc=event_metadata_loc
        )
    elif event_data_provider == "scisports":
        event_data, event_metadata, databallpy_events = load_scisports_event_data(
            events_json=event_data_loc,
        )
    LOGGER.info(
        f"Successfully loaded {event_data_provider} tracking data in load_event_data()"
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
    LOGGER.info("Trying to load open match in get_open_match()")
    try:
        provider_options = ["metrica"]
        if provider not in provider_options:
            LOGGER.error(
                f"{provider} is not a valid provider for the get_open_match() function"
            )
            raise ValueError(
                f"Open match provider should be in {provider_options}, not {provider}."
            )

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
            other_events=databallpy_events["other_events"]
            if "other_events" in databallpy_events.keys()
            else {},
            _tracking_timestamp_is_precise=True,
            _event_timestamp_is_precise=True,
            _periods_changed_playing_direction=(
                metadata.periods_changed_playing_direction
            ),
        )
        LOGGER.info(f"Successfully loaded open match {match.name}")
        return match
    except Exception as e:
        LOGGER.exception(f"Found a unexpected exception in get_open_match(): \n{e}")
        raise e


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
    LOGGER.info("Trying to merge metadata periods in merge_metadata_periods()")
    periods_cols = event_periods.columns.difference(tracking_periods.columns).to_list()
    periods_cols.sort(reverse=True)
    merged_periods = pd.concat(
        (
            tracking_periods,
            event_periods[periods_cols],
        ),
        axis=1,
    )
    LOGGER.info("Successfully merged metadata periods in merge_metadata_periods()")
    return merged_periods


def rescale_event_data(
    tracking_pitch_dimensions: list[float, float],
    event_pitch_dimensions: list[float, float],
    event_data: pd.DataFrame,
    databallpy_events: dict[str, dict[str | int,]] = None,
) -> tuple[pd.DataFrame, dict[str, dict[str | int, IndividualCloseToBallEvent]]]:
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
    LOGGER.info(
        "Trying to rescale the event data based on the tracking data in "
        "rescale_event_data()."
    )
    if (
        tracking_pitch_dimensions == event_pitch_dimensions
        or pd.isnull(list(event_pitch_dimensions)).any()
        or pd.isnull(list(tracking_pitch_dimensions)).any()
    ):
        LOGGER.info(
            "Scaling is not needed because pitch dimensions are equal, "
            "or scaling is not possible because pitch dimensions have nan values."
            "(rescale_event_data())"
        )
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

    LOGGER.info(
        "Successfully rescaled the pitch dimensions of the event data in "
        "rescale_event_data()."
    )
    return event_data, databallpy_events


def align_player_and_team_ids(
    event_metadata: Metadata, tracking_metadata: Metadata
) -> pd.DataFrame:
    """Function to align the player and team id's of the tracking data with the event
    data.

    Args:
        event_metadata (Metadata): event metadata
        tracking_metadata (Metadata): tracking metadata

    Returns:
        pd.DataFrame: tracking data with aligned player and team id's
    """

    LOGGER.info("Trying to align player and team ids in align_player_and_team_id()")
    # Align player id's
    if (
        not set(event_metadata.home_players["id"])
        == set(tracking_metadata.home_players["id"])
    ) or (
        not set(event_metadata.away_players["id"])
        == set(tracking_metadata.away_players["id"])
    ):
        tracking_metadata = align_player_ids(tracking_metadata, event_metadata)

    # Align team id's
    tracking_metadata.home_team_id = event_metadata.home_team_id
    tracking_metadata.away_team_id = event_metadata.away_team_id
    tracking_metadata.home_team_name = event_metadata.home_team_name
    tracking_metadata.away_team_name = event_metadata.away_team_name

    LOGGER.info(
        "Successfully aligned player and team ids in align_player_and_team_ids()"
    )
    return tracking_metadata


def merge_player_info(
    tracking_metadata: Metadata, event_metadata: Metadata
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Function to merge the player information of the tracking and event metadata

    Args:
        tracking_metadata (Metadata): metadata of the tracking data
        event_metadata (Metadata): metadata of the event data

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: merged home player information
            and merged away player information.
    """
    LOGGER.info("Trying to merge player info in merge_player_info()")
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
    LOGGER.info("Successfully merged player info in merge_player_info()")
    return home_players, away_players
