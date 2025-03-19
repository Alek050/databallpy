import os

import bs4
import chardet
import numpy as np
import pandas as pd
import requests

from databallpy.data_parsers.metadata import Metadata
from databallpy.data_parsers.sportec_metadata_parser import (
    _get_sportec_metadata,
    _get_sportec_open_data_url,
)
from databallpy.events import DribbleEvent, PassEvent, ShotEvent
from databallpy.utils.constants import MISSING_INT
from databallpy.utils.logging import logging_wrapper

SPORTEC_SET_PIECES_MAP = {
    "ThrowIn": "throw_in",
    "GoalKick": "goal_kick",
    "FreeKick": "free_kick",
    "Penalty": "penalty",
    "CornerKick": "corner_kick",
    "KickOff": "kick_off",
}
SPORTEC_ON_BALL_EVENTS_MAP = {
    "ShotAtGoal": "shot",
    "Play": "pass",
    "Pass": "pass",
    "Cross": "pass",
}
SPORTEC_SHOT_OUTCOMES = {
    "SavedShot": "miss_on_target",
    "BlockedShot": "blocked",
    "SuccessfulShot": "goal",
    "ShotWide": "miss_off_target",
    "ShotWoodWork": "miss_hit_post",
    "OtherShot": "unspecified",
}
SPORTEC_BODY_PARTS = {
    "head": "head",
    "leftLeg": "left_foot",
    "rightLeg": "right_foot",
}
SPORTEC_ASSISTS = {
    "freeKick": "free_kick",
    "shot": "rebound",
    "header": "open_play",
    "otherPassFromOpenPlay": "open_play",
    "throwIn": "throw_in",
    "cornerKick": "corner_kick",
    "longPassFromOpenPlay": "open_play",
    "crossFromOpenPlay": "open_play",
}

SPORTEC_UNCLEAR_EVENTS = [
    "TacklingGame",
    "OtherBallAction",
]
ALL_SPORTEC_EVENTS = SPORTEC_UNCLEAR_EVENTS + list(SPORTEC_ON_BALL_EVENTS_MAP.keys())


@logging_wrapper(__file__)
def load_sportec_event_data(
    event_data_loc: str, metadata_loc: str
) -> tuple[pd.DataFrame, Metadata, dict[str, dict]]:
    """Base function to load the sportec/DFL event data.

    Args:
        event_data_loc (str): the location of the event data xml
        metadata_loc (str): the location of the tracking data xml

    Raises:
        FileNotFoundError: If the event or metadata location is not found

    Returns:
        tuple[pd.DataFrame, Metadata, dict[str, dict]]: The event data, the event
            metadata, and the databallpy events dictionary.
    """

    metadata = _get_sportec_metadata(metadata_loc, only_event_data=True)
    event_data, databallpy_events = _get_sportec_event_data(event_data_loc, metadata)

    metadata.periods_frames["start_datetime_ed"] = pd.to_datetime(
        metadata.periods_frames["start_datetime_ed"]
    ).dt.tz_localize("Europe/Berlin")
    metadata.periods_frames["end_datetime_ed"] = pd.to_datetime(
        metadata.periods_frames["end_datetime_ed"]
    ).dt.tz_localize("Europe/Berlin")

    metadata.periods_frames.loc[0, "start_datetime_ed"] = event_data.iloc[0]["datetime"]
    metadata.periods_frames.loc[0, "end_datetime_ed"] = event_data.loc[
        event_data["period_id"] == 1, "datetime"
    ].iloc[-1]
    metadata.periods_frames.loc[1, "end_datetime_ed"] = event_data.iloc[-1]["datetime"]
    metadata.periods_frames.loc[1, "start_datetime_ed"] = event_data.loc[
        event_data["period_id"] == 2, "datetime"
    ].iloc[0]

    return event_data, metadata, databallpy_events


@logging_wrapper(__file__)
def load_sportec_open_event_data(
    game_id: str,
) -> tuple[pd.DataFrame, Metadata, dict[str, dict]]:
    """Function to (down)load on open game from Sportec/Tracab

    Args:
        game_id (str): The id of the open game

    Returns:
        tuple[pd.DataFrame, Metadata, dict[str, dict]]: The event data, the event
            metadata, and the databallpy events dictionary.

    Reference:
        Bassek, M., Weber, H., Rein, R., & Memmert,D. (2024). An integrated
        dataset of synchronized spatiotemporal and event data in elite soccer.

    """
    metadata_url = _get_sportec_open_data_url(game_id, "metadata")
    save_path = os.path.join(os.getcwd(), "datasets", "IDSSE", game_id)
    os.makedirs(save_path, exist_ok=True)
    if not os.path.exists(os.path.join(save_path, "metadata.xml")):
        metadata = requests.get(metadata_url)
        with open(os.path.join(save_path, "metadata.xml"), "wb") as f:
            f.write(metadata.content)
    if not os.path.exists(os.path.join(save_path, "event_data.xml")):
        event_data = requests.get(_get_sportec_open_data_url(game_id, "event_data"))
        with open(os.path.join(save_path, "event_data.xml"), "wb") as f:
            f.write(event_data.content)

    return load_sportec_event_data(
        os.path.join(save_path, "event_data.xml"),
        os.path.join(save_path, "metadata.xml"),
    )


@logging_wrapper(__file__)
def _get_sportec_event_data(
    event_data_loc: str, metadata: Metadata
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Functionto get the event data and the metadata for sportec/DFL
    based data.

    Args:
        event_data_loc (str): location of the event data xml file
        metadata (Metadata): Metadata object of the game

    Returns:
        tuple[pd.DataFrame, dict[str, dict]]: the event data and the databallpy events
    """
    with open(event_data_loc, "rb") as f:
        encoding = chardet.detect(f.read())["encoding"]
    with open(event_data_loc, "r", encoding=encoding) as file:
        lines = file.read()
    soup = bs4.BeautifulSoup(lines, "xml")

    all_events = soup.find_all("Event", {"X-Position": True})

    def update_results_dict(res_dict, i, **kwargs):
        for key in res_dict.keys():
            if key in kwargs:
                res_dict[key][i] = kwargs[key]
        return res_dict

    result_dict = {
        "event_id": [MISSING_INT] * len(all_events),
        "databallpy_event": [None] * len(all_events),
        "period_id": [MISSING_INT] * len(all_events),
        "minutes": [MISSING_INT] * len(all_events),
        "seconds": [np.nan] * len(all_events),
        "player_id": [None] * len(all_events),
        "team_id": [None] * len(all_events),
        "is_successful": [None] * len(all_events),
        "start_x": [np.nan] * len(all_events),
        "start_y": [np.nan] * len(all_events),
        "datetime": ["NaT"] * len(all_events),
        "original_event_id": [MISSING_INT] * len(all_events),
        "original_event": [None] * len(all_events),
    }

    pitch_center, period_start_times, swap_half = _initialize_search_variables(
        soup, metadata.home_team_id
    )
    metadata.periods_changed_playing_direction = [swap_half]

    databallpy_events = {
        "pass_events": {},
        "shot_events": {},
        "dribble_events": {},
        "other_events": {},
    }

    for idx, event in enumerate(all_events):
        kwargs = {}
        kwargs["set_piece"] = SPORTEC_SET_PIECES_MAP.get(
            event.find_next().name, "no_set_piece"
        )

        kwargs["datetime"] = pd.to_datetime(event["EventTime"]).tz_convert(
            "Europe/Berlin"
        )
        dt_idx = int(kwargs["datetime"] >= period_start_times[1])
        kwargs["period_id"] = dt_idx + 1
        time_diff_s = (
            kwargs["datetime"].timestamp() - period_start_times[dt_idx].timestamp()
        )
        kwargs["minutes"] = int((45 * dt_idx) + time_diff_s // 60)
        kwargs["seconds"] = time_diff_s % 60
        kwargs["event_id"] = idx
        kwargs["original_event_id"] = int(event["EventId"])

        kwargs["start_x"] = float(event["X-Position"]) - pitch_center[0]
        kwargs["start_y"] = float(event["Y-Position"]) - pitch_center[1]

        if kwargs["period_id"] == swap_half:
            kwargs["start_x"] *= -1
            kwargs["start_y"] *= -1

        next_tag = event
        while next_tag is not None and next_tag.name not in ALL_SPORTEC_EVENTS:
            next_tag = next_tag.find_next()

        event = next_tag if next_tag is not None else event.find_next()

        kwargs["original_event"] = event.name
        kwargs["player_id"] = event.get("Player", event.get("Winner"))
        kwargs["team_id"] = event.get("Team", event.get("WinnerTeam"))

        if event.name == "ShotAtGoal":
            kwargs, shot_event = _handle_shot_event(event, metadata, kwargs)
            databallpy_events["shot_events"][shot_event.event_id] = shot_event
        elif event.name == "Play":
            kwargs, pass_event = _handle_play_event(event, metadata, kwargs)
            databallpy_events["pass_events"][pass_event.event_id] = pass_event
        elif event.name == "TacklingGame":
            kwargs, dbp_event = _handle_tackling_game_event(event, metadata, kwargs)
            if isinstance(dbp_event, DribbleEvent):
                databallpy_events["dribble_events"][dbp_event.event_id] = dbp_event

        if "outcome" in kwargs.keys():
            kwargs["is_successful"] = kwargs.pop("outcome")
        result_dict = update_results_dict(result_dict, idx, **kwargs)

    event_data = pd.DataFrame(result_dict)
    # event_data["datetime"] = pd.to_datetime(event_data["datetime"]).dt.tz_convert(
    #     "Europe/Berlin"
    # )
    event_data = event_data.sort_values("datetime").reset_index(drop=True)

    all_players = pd.concat([metadata.home_players, metadata.away_players])[
        ["full_name", "id"]
    ]
    player_name_series = pd.Series(event_data.index, None, dtype=str)
    player_name_series[~pd.isnull(event_data["player_id"])] = event_data.loc[
        ~pd.isnull(event_data["player_id"]), "player_id"
    ].apply(lambda x: all_players.loc[all_players["id"] == x, "full_name"].iloc[0])
    event_data.insert(6, "player_name", player_name_series)
    event_data["is_successful"] = event_data["is_successful"].astype("boolean")

    return event_data, databallpy_events


def _initialize_search_variables(
    soup: bs4.element.Tag, home_team_id: str
) -> tuple[list[float], list]:
    """Function to get the base variables for the event data.
    The function calculates the center of the pitch, and the start
    datetimes of the first and second half, needed later for calculating
    the minutes/seconds and period of the game.

    Args:
        soup (bs4.element.Tag): The soup of an event
        home_team_id (str): The id of the home team

    Returns:
        tuple[list[float], list[dt.DateTime]]: the x,y location of the center of the
            pitch and the start datetime of the first and second half.
    """
    first_half_kick_off = soup.find(
        lambda tag: tag.name == "Event"
        and tag.find("KickOff", {"GameSection": "firstHalf"})
    )
    second_half_kick_off = soup.find(
        lambda tag: tag.name == "Event"
        and tag.find("KickOff", {"GameSection": "secondHalf"})
    )
    pitch_center = [
        float(first_half_kick_off["X-Position"]),
        float(first_half_kick_off["Y-Position"]),
    ]
    period_start_times = pd.to_datetime(
        [first_half_kick_off["EventTime"], second_half_kick_off["EventTime"]]
    )
    swap_period = (
        1 if first_half_kick_off.find("KickOff")["TeamRight"] == home_team_id else 2
    )

    return pitch_center, period_start_times, swap_period


def _handle_tackling_game_event(
    event: bs4.element.Tag, metadata: Metadata, kwargs_dict: dict
) -> tuple[dict, DribbleEvent | None]:
    """Funtion to handle tackling game events. Only dribbles
    are now  considered since it is not clear when a tackle was performed.

    Args:
        event (bs4.element.Tag): The TacklingGame event
        metadata (Metadata): The metadata of the event data
        kwargs_dict (dict): The kwargs for event_data and databallpy events

    Returns:
        tuple[dict, DribbleEvent | None]: The updated kwargs for the
        event data, and the dribble event or None if it was not a dribble event
    """
    kwargs_dict["original_event"] = event.get("WinnerResult", event.name)
    if not event["WinnerResult"] == "dribbledAround":
        return kwargs_dict, None

    kwargs_dict = _get_base_on_ball_event_kwargs(metadata, kwargs_dict)
    kwargs_dict["outcome"] = event.get("DribbleEvaluation") == "successful"
    kwargs_dict["body_part"] = "foot"
    kwargs_dict["possession_type"] = "open_play"
    kwargs_dict["duel_type"] = "unspecified"
    kwargs_dict["with_opponent"] = True
    kwargs_dict["related_event_id"] = None
    kwargs_dict["databallpy_event"] = "dribble"

    temp_exclude = ["original_event", "original_event_id", "databallpy_event"]

    dribble_event = DribbleEvent(
        **{k: v for k, v in kwargs_dict.items() if k not in temp_exclude} | {"_xt": -1}
    )
    return kwargs_dict, dribble_event


def _handle_shot_event(
    event: bs4.element.Tag, metadata: Metadata, kwargs_dict: dict
) -> tuple[dict, ShotEvent]:
    """Funtion to handle ShotAtGoal events from sportec

    Args:
        event (bs4.element.Tag): The ShotAtGoal event
        metadata (Metadata): The metadata of the event data
        kwargs_dict (dict): The kwargs for event_data and databallpy events

    Returns:
        tuple[dict, ShotEvent]: The updated kwargs for the event data, and
        the databallpy shot event
    """
    kwargs_dict = _get_base_on_ball_event_kwargs(metadata, kwargs_dict)

    kwargs_dict["original_event"] = event.find_next().name
    kwargs_dict["databallpy_event"] = "shot"
    kwargs_dict["related_event_id"] = None
    kwargs_dict["body_part"] = SPORTEC_BODY_PARTS.get(
        event.get("TypeOfShot"), "unspecified"
    )
    kwargs_dict["possession_type"] = SPORTEC_ASSISTS.get(
        event.get("AssistAction"), "unspecified"
    )
    kwargs_dict["outcome"] = SPORTEC_SHOT_OUTCOMES[event.find_next().name] == "goal"
    kwargs_dict["outcome_str"] = SPORTEC_SHOT_OUTCOMES[event.find_next().name]

    temp_exclude = ["original_event", "original_event_id", "databallpy_event"]

    shot_event = ShotEvent(
        **{k: v for k, v in kwargs_dict.items() if k not in temp_exclude} | {"_xt": -1}
    )
    return kwargs_dict, shot_event


def _handle_play_event(
    event: bs4.element.Tag, metadata: Metadata, kwargs_dict: dict
) -> tuple[dict, PassEvent]:
    """Funtion to handle Play events from sportec

    Args:
        event (bs4.element.Tag): The Play event
        metadata (Metadata): The metadata of the event data
        kwargs_dict (dict): The kwargs for event_data and databallpy events

    Returns:
        tuple[dict, PassEvent]: The updated kwargs for the event data, and
        the databallpy pass event
    """
    kwargs_dict = _get_base_on_ball_event_kwargs(metadata, kwargs_dict)
    kwargs_dict["original_event"] = event.find_next().name
    kwargs_dict["databallpy_event"] = "pass"
    kwargs_dict["outcome"] = event["Evaluation"] == "successfullyCompleted"
    kwargs_dict["related_event_id"] = None
    kwargs_dict["body_part"] = "unspecified"
    kwargs_dict["possession_type"] = (
        "open_play" if event["FromOpenPlay"] == "true" else "unspecified"
    )
    kwargs_dict["outcome_str"] = "unspecified"
    kwargs_dict["end_x"] = np.nan
    kwargs_dict["end_y"] = np.nan
    kwargs_dict["pass_type"] = (
        "cross" if event.find_next().name == "Cross" else "unspecified"
    )
    kwargs_dict["receiver_player_id"] = event.get("Recipient", None)

    temp_exclude = ["original_event", "original_event_id", "databallpy_event"]

    pass_event = PassEvent(
        **{k: v for k, v in kwargs_dict.items() if k not in temp_exclude} | {"_xt": -1}
    )
    return kwargs_dict, pass_event


def _get_base_on_ball_event_kwargs(metadata: Metadata, kwargs_dict: dict) -> dict:
    """Function to get the base on ball event info.
    "team_side", "pitch_size", and "jersey"

    Args:
        metadata (Metadata): metadata of the event data
        kwargs_dict (dict): The kwargs for event_data and databallpy events

    Returns:
        dict: The updated kwargs for event_data and databallpy events
    """
    kwargs_dict["team_side"] = (
        "home" if kwargs_dict["team_id"] == metadata.home_team_id else "away"
    )
    kwargs_dict["pitch_size"] = metadata.pitch_dimensions
    players = (
        metadata.home_players
        if kwargs_dict["team_side"] == "home"
        else metadata.away_players
    )
    kwargs_dict["jersey"] = players.loc[
        players["id"] == kwargs_dict["player_id"], "shirt_num"
    ].iloc[0]

    return kwargs_dict
