import json
import os

import pandas as pd

from databallpy.data_parsers import Metadata
from databallpy.data_parsers.event_data_parsers import (
    load_instat_event_data,
    load_metrica_event_data,
    load_metrica_open_event_data,
    load_opta_event_data,
    load_scisports_event_data,
    load_sportec_event_data,
    load_sportec_open_event_data,
    load_statsbomb_event_data,
)
from databallpy.data_parsers.tracking_data_parsers import (
    load_inmotio_tracking_data,
    load_metrica_open_tracking_data,
    load_metrica_tracking_data,
    load_sportec_open_tracking_data,
    load_tracab_tracking_data,
)
from databallpy.data_parsers.tracking_data_parsers.utils import (
    _quality_check_tracking_data,
)
from databallpy.events import IndividualCloseToBallEvent, PassEvent
from databallpy.game import Game
from databallpy.schemas import (
    EventData,
    EventDataSchema,
    TrackingData,
    TrackingDataSchema,
)
from databallpy.utils.align_player_ids import align_player_ids
from databallpy.utils.game_utils import create_event_attributes_dataframe
from databallpy.utils.logging import create_logger, logging_wrapper
from databallpy.utils.warnings import deprecated

LOGGER = create_logger(__name__)


logging_wrapper(__file__)


def get_game(
    tracking_data_loc: str = None,
    tracking_metadata_loc: str = None,
    event_data_loc: str = None,
    event_metadata_loc: str = None,
    event_match_loc: str = None,
    event_lineup_loc: str = None,
    tracking_data_provider: str = None,
    event_data_provider: str = None,
    check_quality: bool = True,
    _check_game_class_: bool = True,
    verbose: bool = True,
) -> Game:
    """
    Function to get all information of a game given its datasources

    Args:
        tracking_data_loc (str, optional): location of the tracking data.
            Defaults to None.
        tracking_metadata_loc (str, optional): location of the metadata of the tracking
            data. Defaults to None.
        event_data_loc (str, optional): location of the event data. Defaults to None.
        event_metadata_loc (str, optional): location of the metadata of the event data.
            Defaults to None.
        event_match_loc (str, optional): location of the game file of the event data.
            Only used for statsbomb event data. Defaults to None.
        event_lineup_loc (str, optional): location of the lineup file of the event data.
            Only used for statsbomb event data. Defaults to None.
        tracking_data_provider (str, optional): provider of the tracking data. Defaults
            to None. Supported providers are [tracab, metrica, inmotio]
        event_data_provider (str, optional): provider of the event data. Defaults to
            None. Supported providers are [opta, metrica, instat, scisports]
        check_quality (bool, optional): whether you want to check the quality of the
            tracking data. Defaults to True
        verbose (bool, optional): whether or not to print info about progress

    Returns:
        (Game): a game object with all information available of the game.

    """
    LOGGER.info(
        "Trying to load a new game in get_game();"
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
        and event_data_provider not in ["scisports", "statsbomb"]
    ):
        raise ValueError(
            "Please provide an event metadata location when providing an event"
            " data location"
        )
    elif event_data_loc and event_data_provider is None:
        raise ValueError(
            "Please provide an event data provider when providing an event"
            " data location"
        )
    elif event_metadata_loc and event_data_provider is None:
        raise ValueError(
            "Please provide an event data provider when providing an event"
            " metadata location"
        )
    elif event_data_provider == "statsbomb" and (
        event_match_loc is None or event_lineup_loc is None
    ):
        raise ValueError(
            "Please provivde both event_match_loc and event_lineup_loc when using statsbomb as event data provider"
        )
    elif tracking_data_loc and tracking_data_provider is None:
        raise ValueError(
            "Please provide a tracking data provider when providing a tracking"
            " data location"
        )
    elif tracking_data_loc and tracking_metadata_loc is None:
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
        "sportec": True,
        "dfl": True,
    }

    event_precise_timestamps = {
        "opta": True,
        "metrica": True,
        "instat": False,
        "scisports": False,
        "statsbomb": False,
        "sportec": True,
        "dfl": True,
    }

    uses_tracking_data = False
    uses_event_data = False

    # Check if event data should be loaded
    if event_data_loc and event_data_provider:
        event_data, event_metadata, databallpy_events = load_event_data(
            event_data_loc=event_data_loc,
            event_metadata_loc=event_metadata_loc,
            event_data_provider=event_data_provider,
            event_match_loc=event_match_loc,
            event_lineup_loc=event_lineup_loc,
        )
        EventDataSchema.validate(event_data)
        uses_event_data = True

    event_precise_timestamps = {
        "opta": True,
        "metrica": True,
        "instat": False,
        "scisports": False,
        "sportec": True,
        "dfl": True,
        "statsbomb": False,
    }

    LOGGER.info(
        "Succesfully passed input checks. Attempting to load the base "
        "data (get_game())."
    )

    # Check if tracking data should be loaded
    if tracking_data_loc and tracking_metadata_loc and tracking_data_provider:
        tracking_data, tracking_metadata = load_tracking_data(
            tracking_data_loc=tracking_data_loc,
            tracking_metadata_loc=tracking_metadata_loc,
            tracking_data_provider=tracking_data_provider,
            verbose=verbose,
        )
        if not uses_event_data:
            databallpy_events = {}

        TrackingDataSchema.validate(tracking_data)
        uses_tracking_data = True

    if not uses_event_data and not uses_tracking_data:
        raise ValueError("No data loaded, please provide data locations and providers")

    LOGGER.info(
        f"Loaded data in get_game():\n\tTracking data: {str(uses_tracking_data)}"
        f"\n\tEvent data: {str(uses_event_data)}"
    )

    # extra checks when using both tracking and event data
    LOGGER.info("Combining info from tracking and event data in get_game()")
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
        tracking_metadata = align_player_and_team_ids(event_metadata, tracking_metadata)
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

    shot_events = (
        create_event_attributes_dataframe(databallpy_events["shot_events"])
        if "shot_events" in databallpy_events.keys()
        else pd.DataFrame()
    )
    pass_events = (
        create_event_attributes_dataframe(databallpy_events["pass_events"])
        if "pass_events" in databallpy_events.keys()
        else pd.DataFrame()
    )
    dribble_events = (
        create_event_attributes_dataframe(databallpy_events["dribble_events"])
        if "dribble_events" in databallpy_events.keys()
        else pd.DataFrame()
    )

    if uses_event_data:
        home_players = event_metadata.home_players
        away_players = event_metadata.away_players
    else:
        home_players = tracking_metadata.home_players
        away_players = tracking_metadata.away_players

    LOGGER.info("Creating game object in get_game()")
    game = Game(
        tracking_data=TrackingData(
            tracking_data,
            provider=tracking_data_provider,
            frame_rate=tracking_metadata.frame_rate,
        )
        if uses_tracking_data
        else TrackingData(),
        event_data=EventData(event_data, provider=event_data_provider)
        if uses_event_data
        else EventData(),
        pitch_dimensions=tracking_metadata.pitch_dimensions
        if uses_tracking_data
        else event_metadata.pitch_dimensions,
        periods=tracking_metadata.periods_frames
        if uses_tracking_data
        else event_metadata.periods_frames,
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
        home_players=home_players,
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
        away_players=away_players,
        country=event_metadata.country if uses_event_data else tracking_metadata.country,
        allow_synchronise_tracking_and_event_data=allow_synchronise,
        shot_events=shot_events,
        dribble_events=dribble_events,
        pass_events=pass_events,
        _event_timestamp_is_precise=event_precise_timestamps[event_data_provider]
        if uses_event_data
        else False,
        _tracking_timestamp_is_precise=tracking_precise_timestamps[
            tracking_data_provider
        ]
        if uses_tracking_data
        else False,
        _periods_changed_playing_direction=changed_periods,
        _check_inputs_=_check_game_class_,
    )
    LOGGER.info(f"Succesfully created game object: {game.name}")
    return game


@logging_wrapper(__file__)
def get_saved_game(name: str, path: str = os.getcwd()) -> Game:
    """Function to load a saved game object

    Args:
        name (str): the name with the to be loaded game, should be a folder.
        The folder should contain:
            - tracking_data.parquet
            - event_data.parquet
            - periods.parquet
            - pass_events.parquet
            - shot_events.parquet
            - dribble_events.parquet
            - away_players.parquet
            - home_players.parquet
            - metadata.json
       path (str, optional): path of directory where game is saved. Defaults
        to current working directory.

    Returns:
        Game: All information about the game
    """

    full_path = os.path.join(path, name)
    if not os.path.isdir(full_path):
        raise ValueError(f"Directory {full_path} does not exist")

    with open(os.path.join(full_path, "metadata.json"), "rb") as f:
        metadata = json.load(f)

    return Game(
        tracking_data=TrackingData(
            pd.read_parquet(os.path.join(full_path, "tracking_data.parquet")),
            provider=metadata["tracking_data_provider"],
            frame_rate=metadata["tracking_data_frame_rate"],
        ),
        event_data=EventData(
            pd.read_parquet(os.path.join(full_path, "event_data.parquet")),
            provider=metadata["event_data_provider"],
        ),
        pitch_dimensions=metadata["pitch_dimensions"],
        periods=pd.read_parquet(os.path.join(full_path, "periods.parquet")),
        home_team_id=metadata["home_team_id"],
        home_formation=metadata["home_formation"],
        home_score=metadata["home_score"],
        home_team_name=metadata["home_team_name"],
        home_players=pd.read_parquet(os.path.join(full_path, "home_players.parquet")),
        away_team_id=metadata["away_team_id"],
        away_formation=metadata["away_formation"],
        away_score=metadata["away_score"],
        away_team_name=metadata["away_team_name"],
        away_players=pd.read_parquet(os.path.join(full_path, "away_players.parquet")),
        country=metadata["country"],
        allow_synchronise_tracking_and_event_data=metadata[
            "allow_synchronise_tracking_and_event_data"
        ],
        shot_events=pd.read_parquet(os.path.join(full_path, "shot_events.parquet")),
        dribble_events=pd.read_parquet(
            os.path.join(full_path, "dribble_events.parquet")
        ),
        pass_events=pd.read_parquet(os.path.join(full_path, "pass_events.parquet")),
        _tracking_timestamp_is_precise=metadata["_tracking_timestamp_is_precise"],
        _event_timestamp_is_precise=metadata["_event_timestamp_is_precise"],
        _periods_changed_playing_direction=metadata[
            "_periods_changed_playing_direction"
        ],
        _is_synchronised=metadata["_is_synchronised"],
        _check_inputs_=False,
    )


@logging_wrapper(__file__)
def load_tracking_data(
    *,
    tracking_data_loc: str,
    tracking_metadata_loc: str,
    tracking_data_provider: str,
    verbose: bool = True,
) -> tuple[pd.DataFrame, Metadata]:
    """Function to load the tracking data of a game

    Args:
        tracking_data_loc (str): location of the tracking data file
        tracking_metadata_loc (str): location of the tracking metadata file
        tracking_data_provider (str): provider of the tracking data
        verbose (bool, optional): whether or not to print info about progress

    Returns:
        Tuple[pd.DataFrame, Metadata]: tracking data and metadata of the game
    """

    if tracking_data_provider not in ["tracab", "metrica", "inmotio", "sportec", "dfl"]:
        raise ValueError(
            f"We do not support '{tracking_data_provider}' as tracking data provider"
            " yet, please open an issue in our Github repository."
        )

    # Get tracking data and tracking metadata
    if tracking_data_provider in ["tracab", "sportec", "dfl"]:
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


@logging_wrapper(__file__)
def load_event_data(
    *,
    event_data_loc: str,
    event_metadata_loc: str,
    event_data_provider: str,
    event_match_loc: str,
    event_lineup_loc: str,
) -> tuple[pd.DataFrame, Metadata]:
    """Function to load the event data of a game

    Args:
        event_data_loc (str): location of the event data file
        event_metadata_loc (str): location of the event metadata file
        event_data_provider (str): provider of the event data
        event_match_loc (str): location of match file (specific to statsbomb)
        event_lineup_loc (str): location of lineup file (specific to statsbomb)

    Returns:
        Tuple[pd.DataFrame, Metadata]: event data and metadata of the game
    """

    if event_data_provider not in [
        "opta",
        "metrica",
        "instat",
        "scisports",
        "statsbomb",
        "sportec",
        "dfl",
    ]:
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
    elif event_data_provider == "statsbomb":
        event_data, event_metadata, databallpy_events = load_statsbomb_event_data(
            events_loc=event_data_loc,
            match_loc=event_match_loc,
            lineup_loc=event_lineup_loc,
        )
    elif event_data_provider in ["sportec", "dfl"]:
        event_data, event_metadata, databallpy_events = load_sportec_event_data(
            event_data_loc=event_data_loc, metadata_loc=event_metadata_loc
        )
    return event_data, event_metadata, databallpy_events


@logging_wrapper(__file__)
def get_open_game(
    provider: str = "sportec",
    game_id: str = "J03WMX",
    verbose: bool = True,
    use_cache: bool = True,
) -> Game:
    """Function to load a game object from an open datasource

    Args:
        provider (str, optional): What provider to get the open data from. Defaults to "dfl". Options are ["metrica", "dfl", "sportec", "tracab"]
        game_id (str, optional): The Game id of the open game. Defaults to 'J03WMX',
        verbose (bool, optional): Whether or not to print info about progress
        in the terminal, Defaults to True.
        use_cache (bool, optional): Use cached version of match if available.

    Returns:
        Game: All information about the game
    """
    provider_options = ["metrica", "dfl", "sportec", "tracab"]
    if provider not in provider_options:
        raise ValueError(
            f"Open game provider should be in {provider_options}, not {provider}."
        )

    if provider == "metrica":
        save_path = os.path.join("datasets", "metrica")
        if use_cache and os.path.exists(save_path):
            return get_saved_game(save_path)
        tracking_data, metadata = load_metrica_open_tracking_data(verbose=verbose)
        event_data, ed_metadata, databallpy_events = load_metrica_open_event_data()

    elif provider in ["dfl", "tracab", "sportec"]:
        save_path = os.path.join("datasets", "IDSSE", game_id)
        if use_cache and os.path.exists(save_path):
            return get_saved_game(save_path)

        tracking_data, metadata = load_sportec_open_tracking_data(
            game_id=game_id,
            verbose=verbose,
        )
        event_data, ed_metadata, databallpy_events = load_sportec_open_event_data(
            game_id=game_id
        )
        os.remove(os.path.join("datasets", "IDSSE", game_id, "tracking_data_temp.xml"))
        os.remove(os.path.join("datasets", "IDSSE", game_id, "metadata_temp.xml"))

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

    shot_events = (
        create_event_attributes_dataframe(databallpy_events["shot_events"])
        if "shot_events" in databallpy_events.keys()
        else pd.DataFrame()
    )
    pass_events = (
        create_event_attributes_dataframe(databallpy_events["pass_events"])
        if "pass_events" in databallpy_events.keys()
        else pd.DataFrame()
    )
    dribble_events = (
        create_event_attributes_dataframe(databallpy_events["dribble_events"])
        if "dribble_events" in databallpy_events.keys()
        else pd.DataFrame()
    )

    game = Game(
        tracking_data=TrackingData(
            tracking_data, provider=provider, frame_rate=metadata.frame_rate
        ),
        event_data=EventData(event_data, provider=provider),
        pitch_dimensions=metadata.pitch_dimensions,
        periods=merged_periods,
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
        country=ed_metadata.country,
        allow_synchronise_tracking_and_event_data=True,
        shot_events=shot_events,
        dribble_events=dribble_events,
        pass_events=pass_events,
        _tracking_timestamp_is_precise=True,
        _event_timestamp_is_precise=True,
        _periods_changed_playing_direction=(metadata.periods_changed_playing_direction),
    )

    game.save_game(save_path, verbose=False, allow_overwrite=True)
    return game


@logging_wrapper(__file__)
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


@logging_wrapper(__file__)
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

    return event_data, databallpy_events


@logging_wrapper(__file__)
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

    home_eq = (
        tracking_metadata.home_players["id"]
        .isin(event_metadata.home_players["id"])
        .sum()
        > 11
    )
    away_eq = (
        tracking_metadata.away_players["id"]
        .isin(event_metadata.away_players["id"])
        .sum()
        > 11
    )

    if not home_eq or not away_eq:
        tracking_metadata = align_player_ids(tracking_metadata, event_metadata)

    # Align team id's
    tracking_metadata.home_team_id = event_metadata.home_team_id
    tracking_metadata.away_team_id = event_metadata.away_team_id
    tracking_metadata.home_team_name = event_metadata.home_team_name
    tracking_metadata.away_team_name = event_metadata.away_team_name

    return tracking_metadata


@logging_wrapper(__file__)
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
    away_players["position"] = event_metadata.away_players["position"]
    home_players["position"] = event_metadata.home_players["position"]
    return home_players, away_players


@deprecated(
    "`get_match` is deprecated and will be removed in version 0.8.0. Please use `get_game` instead"
)
def get_match(*args, **kwargs):
    return get_game(*args, **kwargs)


@deprecated(
    "`get_saved_match` is deprecated and will be removed in version 0.8.0. Please use `get_saved_game` instead"
)
def get_open_match(*args, **kwargs):
    return get_open_game(*args, **kwargs)


@deprecated(
    "`get_saved_match` is deprecated and will be removed in version 0.8.0. Please use `get_saved_game` instead"
)
def get_saved_match(*args, **kwargs):
    return get_saved_game(*args, **kwargs)
