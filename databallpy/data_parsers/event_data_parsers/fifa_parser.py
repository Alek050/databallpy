import json

import numpy as np
import pandas as pd

from databallpy.data_parsers import Metadata
from databallpy.data_parsers.event_data_parsers.utils import _add_spadl_columns
from databallpy.utils.constants import MISSING_INT
from databallpy.utils.event_extras import create_all_event_extras
from databallpy.utils.logging import logging_wrapper
from databallpy.utils.tz_modification import utc_to_local_datetime

FIFA_TO_DATABALLPY_MAP = {
    "pass": "pass",
    "assist": "pass",
    "cross": "pass",
    "attempt_at_goal": "shot",
    "goal": "shot",
    "own_goal": "own_goal",
}

SHOT_OUTCOMES = {
    "incomplete": "miss_off_target",
    "off_target": "miss_off_target",
    "on_target": "miss_on_target",
    "complete": "goal",
    "own_goal": "own_goal",
}

BODY_PART_MAP = {
    "right_foot": "right_foot",
    "left_foot": "left_foot",
    "head": "head",
    "hands": "hands",
    "body": "other",
    "feet": "other",
}

SET_PIECE_MAP = {
    "freekick": "free_kick",
    "throwin": "throw_in",
    "goalkick": "goal_kick",
    "corner": "corner_kick",
    "kickoff": "kick_off",
    "penalty": "penalty",
}

FIFA_LINE_BREAK_TO_PASS_TYPE = {
    "over": "long_ball",
    "through": "through_ball",
    "around": "through_ball",
}

SPADL_BODY_PART_MAPPING = {
    "right_foot": "foot_right",
    "left_foot": "foot_left",
    "head": "head",
    "hands": "other",
    "body": "other",
    "feet": "foot",
}

SPADL_SET_PIECE_TYPES = {
    "freekick": "freekick_short",
    "corner": "corner_short",
    "throwin": "throw_in",
    "goalkick": "goalkick",
    "kickoff": "pass",
}


@logging_wrapper(__file__)
def load_fifa_event_data(
    metadata_loc: str,
    events_loc: str,
    pitch_dimensions: list = [105.0, 68.0],
) -> tuple[pd.DataFrame, Metadata, dict[str, pd.DataFrame]]:
    """This function retrieves the metadata and event data of a FIFA match. The x
    and y coordinates provided have been scaled to the dimensions of the pitch, with
    (0, 0) being the center. Additionally, the coordinates have been standardized so
    that the home team is represented as playing from left to right for the entire
    game, and the away team is represented as playing from right to left.

    Args:
        metadata_loc (str): location of the metadata JSON file.
        events_loc (str): location of the events JSON file.
        pitch_dimensions (list, optional): the length and width of the pitch in meters.

    Returns:
        Tuple[pd.DataFrame, Metadata, dict]: the event data of the game, the metadata,
        and the extra information of the shot and pass events.
    """
    if not isinstance(metadata_loc, str):
        raise TypeError(f"metadata_loc should be a string, not a {type(metadata_loc)}")
    if not isinstance(events_loc, str):
        raise TypeError(f"events_loc should be a string, not a {type(events_loc)}")
    if not metadata_loc.endswith(".json"):
        raise ValueError(
            f"metadata file should be of .json format, not {metadata_loc.split('.')[-1]}"
        )
    if not events_loc.endswith(".json"):
        raise ValueError(
            f"events file should be of .json format, not {events_loc.split('.')[-1]}"
        )

    metadata = _load_metadata(metadata_loc, pitch_dimensions=pitch_dimensions)
    kickoff_time = metadata.periods_frames.loc[
        metadata.periods_frames["period_id"] == 1, "start_datetime_ed"
    ].iloc[0]

    kickoff_time_utc = kickoff_time.tz_convert("UTC")

    event_data, databallpy_events = _load_event_data(
        events_loc,
        metadata.home_team_id,
        metadata.away_team_id,
        pitch_dimensions=pitch_dimensions,
        kickoff_time=kickoff_time_utc,
    )

    event_data["datetime"] = utc_to_local_datetime(
        event_data["datetime"], metadata.country
    )

    home_score, away_score = _get_game_score(
        event_data,
        metadata.home_team_id,
        metadata.away_team_id,
    )
    metadata.home_score = home_score
    metadata.away_score = away_score

    home_players = dict(
        zip(metadata.home_players["id"], metadata.home_players["full_name"])
    )
    away_players = dict(
        zip(metadata.away_players["id"], metadata.away_players["full_name"])
    )

    home_mask = (event_data["team_id"] == metadata.home_team_id) & ~pd.isnull(
        event_data["player_id"]
    )
    away_mask = (event_data["team_id"] == metadata.away_team_id) & ~pd.isnull(
        event_data["player_id"]
    )

    event_data.insert(event_data.columns.get_loc("player_id") + 1, "player_name", None)
    event_data.loc[home_mask, "player_name"] = event_data.loc[
        home_mask, "player_id"
    ].map(home_players)
    event_data.loc[away_mask, "player_name"] = event_data.loc[
        away_mask, "player_id"
    ].map(away_players)
    event_data["player_name"] = event_data["player_name"].replace({np.nan: None})

    return event_data, metadata, databallpy_events


@logging_wrapper(__file__)
def _load_metadata(metadata_loc: str, pitch_dimensions: list) -> Metadata:
    """Function to load metadata from the FIFA metadata JSON file.

    Args:
        metadata_loc (str): location of the metadata JSON file.
        pitch_dimensions (list): the length and width of the pitch in meters.

    Returns:
        Metadata: all metadata information of the current game.
    """
    with open(metadata_loc, "r", encoding="utf-8") as f:
        metadata_json = json.load(f)

    match_id = metadata_json["match_id"]
    country = metadata_json.get("country", "UNKNOWN")

    kickoff_time = pd.to_datetime(metadata_json["kickoff_utc"], utc=True)

    periods = {
        "period_id": [1, 2, 3, 4, 5],
        "start_datetime_ed": [],
        "end_datetime_ed": [],
    }

    for phase in metadata_json["phases"][:2]:
        start_time = kickoff_time + pd.to_timedelta(phase["phase_start"])
        end_time = kickoff_time + pd.to_timedelta(phase["phase_end"])
        periods["start_datetime_ed"].append(start_time)
        periods["end_datetime_ed"].append(end_time)

    for _ in range(3):
        periods["start_datetime_ed"].append(pd.to_datetime("NaT", utc=True))
        periods["end_datetime_ed"].append(pd.to_datetime("NaT", utc=True))

    periods_df = pd.DataFrame(periods)
    periods_df["start_datetime_ed"] = utc_to_local_datetime(
        periods_df["start_datetime_ed"], country
    )
    periods_df["end_datetime_ed"] = utc_to_local_datetime(
        periods_df["end_datetime_ed"], country
    )

    home_players = _get_player_info(metadata_json["home_team_players"])
    away_players = _get_player_info(metadata_json["away_team_players"])

    home_team_id = metadata_json["home_team_id"]
    away_team_id = metadata_json["away_team_id"]
    home_score, away_score = np.nan, np.nan

    home_formation = metadata_json.get("home_formation", "")
    away_formation = metadata_json.get("away_formation", "")

    metadata = Metadata(
        game_id=match_id,
        pitch_dimensions=pitch_dimensions,
        periods_frames=periods_df,
        frame_rate=MISSING_INT,
        home_team_id=home_team_id,
        home_team_name=metadata_json["home_team_name"],
        home_players=home_players,
        home_score=home_score,
        home_formation=home_formation,
        away_team_id=away_team_id,
        away_team_name=metadata_json["away_team_name"],
        away_players=away_players,
        away_score=away_score,
        away_formation=away_formation,
        country=country,
    )

    return metadata


def _get_player_info(players_data: list) -> pd.DataFrame:
    """Function to loop over all players and save data in a pd.DataFrame.

    Args:
        players_data (list): for every player a dictionary with info about the player.

    Returns:
        pd.DataFrame: all information of the players.
    """
    n = len(players_data)
    result_dict = {
        "id": [MISSING_INT] * n,
        "full_name": [""] * n,
        "formation_place": [MISSING_INT] * n,
        "position": ["unspecified"] * n,
        "starter": [False] * n,
        "shirt_num": [MISSING_INT] * n,
    }

    for idx, player in enumerate(players_data):
        result_dict["id"][idx] = player["player_id"]
        result_dict["full_name"][idx] = player["player_name"]
        result_dict["shirt_num"][idx] = player["player_shirt_number"]

    players_df = pd.DataFrame(result_dict)
    players_df["full_name"] = players_df["full_name"].str.title()
    return players_df


@logging_wrapper(__file__)
def _load_event_data(
    events_loc: str,
    home_team_id: int,
    away_team_id: int,
    pitch_dimensions: list = [105.0, 68.0],
    kickoff_time: pd.Timestamp = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Function to load FIFA event data from a newline-delimited JSON file.

    Args:
        events_loc (str): location of the events JSON file.
        home_team_id (int): id of the home team.
        away_team_id (int): id of the away team.
        pitch_dimensions (list, optional): dimensions of the pitch.
            Defaults to [105.0, 68.0].
        kickoff_time (pd.Timestamp): kickoff time of the match.

    Returns:
        pd.DataFrame: all events of the game in a pd.DataFrame.
        dict: dict with "shot_events", "pass_events" and "dribble_events" as keys,
              each with a dataframe with the extra information of the events.
    """
    shot_events = {}
    pass_events = {}

    events_list = []
    with open(events_loc, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events_list.append(json.loads(line))

    flip_first_half, flip_second_half = _determine_period_flips(
        events_list, home_team_id, away_team_id
    )

    result_dict = {
        "event_id": [],
        "databallpy_event": [],
        "period_id": [],
        "minutes": [],
        "seconds": [],
        "player_id": [],
        "team_id": [],
        "is_successful": [],
        "start_x": [],
        "start_y": [],
        "end_x": [],
        "end_y": [],
        "datetime": [],
        "phase": [],
        "original_event_id": [],
        "original_event": [],
        "original_event_type": [],
        "action_type": [],
        "action_result": [],
        "action_bodypart": [],
    }

    for i_event, event in enumerate(events_list):
        if event["team_id"] == 0 or event["from_player_id"] == 0:
            continue

        result_dict["event_id"].append(i_event)
        result_dict["original_event_id"].append(event["event_id"])
        result_dict["original_event_type"].append(event["event_type"])
        result_dict["phase"].append(event["category"])

        event_name = event["event"]
        result_dict["original_event"].append(event_name)

        period_id = event["half_time"]
        result_dict["period_id"].append(period_id)

        total_seconds = event["match_time_in_ms"] / 1000.0
        result_dict["minutes"].append(int(total_seconds // 60))
        result_dict["seconds"].append(total_seconds % 60)

        if kickoff_time is not None:
            event_datetime = kickoff_time + pd.to_timedelta(total_seconds, unit="s")
        else:
            event_datetime = pd.NaT
        result_dict["datetime"].append(event_datetime)

        result_dict["player_id"].append(
            event["from_player_id"] if event["from_player_id"] != 0 else MISSING_INT
        )
        result_dict["team_id"].append(event["team_id"])

        # Determine is_successful per event type
        if event_name in ["pass", "tackle"]:
            outcome = (event.get("outcome") or "").lower()
            result_dict["is_successful"].append(
                1
                if ("possession_complete" in outcome or "possession_won" in outcome)
                else 0
            )
        elif event_name == "attempt_at_goal":
            shot_outcome_str = (event.get("outcome") or "").lower()
            result_dict["is_successful"].append(
                1 if SHOT_OUTCOMES.get(shot_outcome_str) == "goal" else 0
            )
        elif event_name == "goal":
            result_dict["is_successful"].append(1)
        elif event_name == "own_goal":
            result_dict["is_successful"].append(1)
        else:
            result_dict["is_successful"].append(None)

        x_raw = (
            event.get("x_location_start")
            if event.get("x_location_start") is not None
            else event.get("x")
        )
        y_raw = (
            event.get("y_location_start")
            if event.get("y_location_start") is not None
            else event.get("y")
        )

        x_start, y_start = _get_transformed_coordinates(
            x_raw, y_raw, pitch_dimensions, period_id, flip_first_half, flip_second_half
        )
        result_dict["start_x"].append(x_start)
        result_dict["start_y"].append(y_start)

        action_type, action_result, action_bodypart = _get_fifa_spadl_action(event)
        result_dict["action_type"].append(action_type)
        result_dict["action_result"].append(action_result)
        result_dict["action_bodypart"].append(action_bodypart)
        if action_result == "owngoal":
            # the own goal ends in the own goal of the team, the home team plays
            # from left to right
            x_end = (
                pitch_dimensions[0]
                / 2.0
                * (-1 if event["team_id"] == home_team_id else 1)
            )
            y_end = 0.0
        else:
            x_end, y_end = _get_transformed_coordinates(
                event.get("x_location_end"),
                event.get("y_location_end"),
                pitch_dimensions,
                period_id,
                flip_first_half,
                flip_second_half,
            )
        result_dict["end_x"].append(x_end)
        result_dict["end_y"].append(y_end)

        if event_name in ["pass", "cross", "assist"]:
            pass_events[i_event] = _get_pass_extras(event)

        if event_name in ["attempt_at_goal", "goal", "own_goal"]:
            shot_events[i_event] = _get_shot_extras(event)

    result_dict["databallpy_event"] = [None] * len(result_dict["event_id"])
    event_data = pd.DataFrame(result_dict)
    event_data["databallpy_event"] = (
        event_data["original_event"]
        .map(FIFA_TO_DATABALLPY_MAP)
        .replace([np.nan], [None])
    )

    event_data["is_successful"] = event_data["is_successful"].astype("boolean")
    event_data.loc[event_data["period_id"] > 5, "period_id"] = -1
    event_data = _add_spadl_columns(event_data)

    return event_data, create_all_event_extras(
        {
            "shot_events": shot_events,
            "pass_events": pass_events,
        }
    )


def _get_fifa_spadl_action(
    event: dict,
) -> tuple[str | None, str | None, str | None]:
    """Function to map a FIFA event to the SPADL vocabulary. The set piece type of
    passes and shots is taken from the origin of the event, kickoff events themselves
    are no actions.

    Args:
        event (dict): event from the FIFA json

    Returns:
        tuple[str | None, str | None, str | None]: The SPADL action type, result and
            bodypart, all None for events that are not SPADL actions
    """
    event_name = event["event"]
    outcome = (event.get("outcome") or "").lower()
    origin = event.get("origin")
    bodypart = SPADL_BODY_PART_MAPPING.get(event.get("body_type"), "foot")

    if event_name in ["pass", "assist", "cross"]:
        action_type = "cross" if event_name == "cross" else "pass"
        if origin in SPADL_SET_PIECE_TYPES:
            action_type = SPADL_SET_PIECE_TYPES[origin]
            if action_type in ["freekick_short", "corner_short"] and (
                event_name == "cross"
            ):
                action_type = action_type.replace("short", "crossed")
            elif action_type == "throw_in":
                bodypart = "other"
        result = "success" if "possession_complete" in outcome else "fail"
        return action_type, result, bodypart
    elif event_name in ["attempt_at_goal", "goal"]:
        # NOTE: it is unknown whether a goal is also stored as an attempt_at_goal
        # with outcome complete, in that case the goal is counted as two shots
        action_type = "shot"
        if origin == "penalty":
            action_type = "shot_penalty"
        elif origin == "freekick":
            action_type = "shot_freekick"
        is_goal = event_name == "goal" or SHOT_OUTCOMES.get(outcome) == "goal"
        return action_type, "success" if is_goal else "fail", bodypart
    elif event_name == "own_goal":
        return "bad_touch", "owngoal", bodypart
    elif event_name == "tackle":
        return "tackle", "success" if "possession_won" in outcome else "fail", bodypart

    return None, None, None


def _get_pass_extras(event: dict) -> dict:
    """Function to get the extra information of a pass event from FIFA event data.

    Args:
        event (dict): event from FIFA data.

    Returns:
        dict: the extra information of the pass event.
    """
    extras = _get_on_ball_event_info(event)
    extras["related_event_id"] = MISSING_INT
    extras["outcome"] = (event.get("outcome") or "").lower() == "possession_complete"
    extras["outcome_str"] = "successful" if extras["outcome"] else "unsuccessful"

    event_name = event.get("event", "")
    if event_name == "cross":
        extras["pass_type"] = "cross"
    else:
        line_break = event.get("line_break_direction")
        extras["pass_type"] = FIFA_LINE_BREAK_TO_PASS_TYPE.get(line_break, "unspecified")

    return extras


def _get_shot_extras(event: dict) -> dict:
    """Function to get the extra information of a shot event from FIFA event data.

    Args:
        event (dict): event from FIFA data.

    Returns:
        dict: the extra information of the shot event.
    """
    extras = _get_on_ball_event_info(event)
    extras["related_event_id"] = MISSING_INT

    event_name = (event.get("event") or "").lower()
    if event_name == "own_goal":
        shot_outcome = "own_goal"
    elif event_name == "goal":
        shot_outcome = "goal"
    else:
        raw_outcome = (event.get("outcome") or "").lower()
        shot_outcome = SHOT_OUTCOMES.get(raw_outcome, "miss_off_target")

    extras["outcome"] = shot_outcome == "goal"
    extras["outcome_str"] = shot_outcome
    return extras


def _get_on_ball_event_info(event: dict) -> dict:
    """Function to get the on-ball event information of an event.

    Args:
        event (dict): event from FIFA data.

    Returns:
        dict: dictionary with body_part, set_piece, and possession_type.
    """
    body_type = event.get("body_type")
    body_part = (
        BODY_PART_MAP.get(body_type, "unspecified") if body_type else "unspecified"
    )

    origin = event.get("origin") or ""
    set_piece = SET_PIECE_MAP.get(origin, "no_set_piece")

    return {
        "body_part": body_part,
        "set_piece": set_piece,
        "possession_type": "open_play",
    }


def _get_game_score(
    events: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
) -> tuple[int, int]:
    """Function to extract game scores by counting goals from event data.

    Args:
        events (pd.DataFrame): DataFrame of event data.
        home_team_id (int): ID of the home team.
        away_team_id (int): ID of the away team.

    Returns:
        tuple[int, int]: (home_score, away_score).
    """
    home_score = 0
    away_score = 0

    shot_events = events[
        events["original_event"].isin(["attempt_at_goal", "goal", "own_goal"])
    ]

    for _, row in shot_events.iterrows():
        event_name = (row.get("original_event") or "").lower()
        team_id = row.get("team_id")

        is_goal = False
        is_own_goal = False

        if event_name == "own_goal":
            is_own_goal = True
            is_goal = True
        elif event_name == "goal":
            is_goal = True
        elif event_name == "attempt_at_goal":
            if row.get("is_successful"):
                is_goal = True

        if is_goal:
            if is_own_goal:
                if team_id == home_team_id:
                    away_score += 1
                elif team_id == away_team_id:
                    home_score += 1
            else:
                if team_id == home_team_id:
                    home_score += 1
                elif team_id == away_team_id:
                    away_score += 1

    return home_score, away_score


def _determine_period_flips(
    events_list: list,
    home_team_id: int,
    away_team_id: int,
) -> tuple[bool, bool]:
    """Determine whether to flip coordinates for first and second half.

    FIFA uses a global coordinate frame (not team-relative). The kickoff event
    tells us which side the kicking team plays from. If the home team attacks
    toward x=0 (right side = side "r") in the first half, all first-half
    coordinates need to be mirrored so the home team consistently attacks
    toward positive x. The second half is always the opposite of the first.

    Args:
        events_list (list): list of all events.
        home_team_id (int): home team ID.
        away_team_id (int): away team ID.

    Returns:
        tuple[bool, bool]: (flip_first_half, flip_second_half).
    """
    flip_first_half = False
    flip_second_half = False

    for event in events_list:
        if (
            event.get("half_time") == 1
            and event.get("event", "").lower() == "kickoff"
            and event.get("from_player_id", 0) != 0
        ):
            team_id = event.get("team_id")
            side = (event.get("side") or "").lower()

            if team_id == home_team_id:
                flip_first_half = side == "r"
            elif team_id == away_team_id:
                flip_first_half = side == "l"

            flip_second_half = not flip_first_half
            break

    return flip_first_half, flip_second_half


def _get_transformed_coordinates(
    x_val: float,
    y_val: float,
    pitch_dimensions: list,
    period_id: int,
    flip_first_half: bool,
    flip_second_half: bool,
) -> tuple[float, float]:
    """Transform raw FIFA coordinates to databallpy pitch coordinates.

    FIFA data uses a consistent global coordinate frame (0–1 normalized, same
    origin for both teams). The period flip is the only normalization needed:
    it ensures the home team always attacks toward positive x regardless of
    which end they start from. No team-specific flip is applied because the
    global frame already places the away team's attacks in negative-x territory
    after the period is normalized.

    Missing coordinates are returned as NaN rather than silently placed at
    the pitch center.

    Args:
        x_val (float): raw x coordinate in [0, 1].
        y_val (float): raw y coordinate in [0, 1].
        pitch_dimensions (list): [length, width] of the pitch in meters.
        period_id (int): match period (1 or 2).
        flip_first_half (bool): mirror all first-half coordinates.
        flip_second_half (bool): mirror all second-half coordinates.

    Returns:
        tuple[float, float]: (x, y) in metres centred at (0, 0).
    """
    if x_val is None or pd.isna(x_val):
        return np.nan, np.nan
    if y_val is None or pd.isna(y_val):
        return np.nan, np.nan

    x_norm = max(0.0, min(1.0, float(x_val)))
    y_norm = max(0.0, min(1.0, float(y_val)))

    if period_id == 1 and flip_first_half:
        x_norm = 1.0 - x_norm
        y_norm = 1.0 - y_norm
    elif period_id == 2 and flip_second_half:
        x_norm = 1.0 - x_norm
        y_norm = 1.0 - y_norm

    x_trans = (x_norm * pitch_dimensions[0]) - (pitch_dimensions[0] / 2.0)
    y_trans = (y_norm * pitch_dimensions[1]) - (pitch_dimensions[1] / 2.0)

    return x_trans, y_trans
