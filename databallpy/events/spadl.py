"""Provider independent conversion of databallpy event data to SPADL actions.

The conversion follows socceraction (Decroos et al., 2019) as closely as possible:
same columns, ids, vocabularies and the same rules to fix clearances and add
dribbles. The provider specific mapping to the SPADL vocabulary is done by the
parsers in the ``action_type``, ``action_result`` and ``action_bodypart`` columns.
"""

import warnings

import numpy as np
import pandas as pd
from pandera.errors import SchemaError, SchemaErrors

from databallpy.schemas.spadl import SPADLSchema
from databallpy.utils.constants import (
    MISSING_INT,
    PERIOD_START_MINUTES,
    SPADL_ACTION_TYPES,
    SPADL_BODYPARTS,
    SPADL_FIELD_LENGTH,
    SPADL_FIELD_WIDTH,
    SPADL_RESULTS,
    SPADL_TO_DATABALLPY_EVENTS,
)
from databallpy.utils.warnings import DataBallPyWarning

SPADL_ACTION_COLUMNS = [
    "game_id",
    "original_event_id",
    "event_id",
    "action_id",
    "period_id",
    "time_seconds",
    "team_id",
    "player_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "type_id",
    "result_id",
    "bodypart_id",
    "type_name",
    "result_name",
    "bodypart_name",
]

PLAYING_DIRECTIONS = [
    "home_oriented",
    "away_oriented",
    "possession_oriented",
    "out_of_possession_oriented",
]
COORDINATE_SYSTEMS = ["databallpy", "spadl"]

MIN_DRIBBLE_LENGTH = 3.0
MAX_DRIBBLE_LENGTH = 60.0
MAX_DRIBBLE_DURATION = 10.0


def event_data_to_spadl(
    event_data: pd.DataFrame,
    home_team_id: int | str,
    pitch_dimensions: list[float],
    game_id: int | str | None = None,
    playing_direction: str = "home_oriented",
    coordinate_system: str = "databallpy",
) -> pd.DataFrame:
    """Function to convert event data to SPADL actions.

    Args:
        event_data (pd.DataFrame): The event data with the SPADL vocabulary
            columns (action_type, action_result and action_bodypart) filled in.
        home_team_id (int | str): The id of the home team.
        pitch_dimensions (list[float]): The length and width of the pitch in meters.
        game_id (int | str | None, optional): The value of the game_id column.
            Defaults to None.
        playing_direction (str, optional): The playing direction of the actions.
            Defaults to "home_oriented", options are {home_oriented, away_oriented,
            possession_oriented, out_of_possession_oriented}. Possession oriented
            means the team performing the action plays from left to right.
        coordinate_system (str, optional): The coordinate system of the actions.
            Defaults to "databallpy", meters on the actual pitch with (0, 0) at
            the centre spot. With "spadl" the coordinates are rescaled to the
            105x68 SPADL field with (0, 0) in the bottom left corner.

    Returns:
        pd.DataFrame: The SPADL actions. The original_event_id and event_id columns
            link the actions to the event data and are null for added dribbles.
    """
    playing_direction = _check_spadl_options(playing_direction, coordinate_system)
    actions = _get_home_oriented_spadl_actions(event_data, pitch_dimensions, game_id)
    actions = _apply_playing_direction(actions, home_team_id, playing_direction)
    if coordinate_system == "databallpy":
        actions = _spadl_to_databallpy_coordinates(actions, pitch_dimensions)
    return actions


def _check_spadl_options(playing_direction: str, coordinate_system: str) -> str:
    """Function to check the playing direction and coordinate system of the SPADL
    conversion, returns the playing direction with deprecated options replaced."""
    if playing_direction == "team_oriented":
        warnings.warn(
            "'team_oriented' is deprecated and will be removed in a future "
            "version. Use 'home_oriented' instead",
            category=DataBallPyWarning,
        )
        playing_direction = "home_oriented"
    if playing_direction not in PLAYING_DIRECTIONS:
        raise ValueError(
            f"Playing direction {playing_direction} is not supported, options are "
            f"{PLAYING_DIRECTIONS}."
        )
    if coordinate_system not in COORDINATE_SYSTEMS:
        raise ValueError(
            f"Coordinate system {coordinate_system} is not supported, options are "
            f"{COORDINATE_SYSTEMS}."
        )
    return playing_direction


def _get_home_oriented_spadl_actions(
    event_data: pd.DataFrame,
    pitch_dimensions: list[float],
    game_id: int | str | None,
) -> pd.DataFrame:
    """Function to get the validated SPADL actions in the SPADL coordinate system,
    with the home team playing from left to right."""
    actions = _get_actions(event_data, pitch_dimensions, game_id)
    actions = actions.sort_values(
        ["period_id", "time_seconds"], kind="mergesort"
    ).reset_index(drop=True)
    actions = _fix_clearances(actions)
    actions = _fix_interceptions(actions)
    actions["action_id"] = range(len(actions))
    actions = _add_dribbles(actions)
    actions = _add_names(actions)
    return SPADLSchema.validate(actions[SPADL_ACTION_COLUMNS])


def _add_names(actions: pd.DataFrame) -> pd.DataFrame:
    """Function to add the type, result and bodypart names of the ids."""
    return actions.assign(
        type_name=actions["type_id"].map(dict(enumerate(SPADL_ACTION_TYPES))),
        result_name=actions["result_id"].map(dict(enumerate(SPADL_RESULTS))),
        bodypart_name=actions["bodypart_id"].map(dict(enumerate(SPADL_BODYPARTS))),
    )


def _get_actions(
    event_data: pd.DataFrame,
    pitch_dimensions: list[float],
    game_id: int | str | None,
) -> pd.DataFrame:
    """Function to select the on-ball actions from the event data and convert them
    to the SPADL columns, ids and coordinate system (home team plays left to right).
    """
    is_action = (
        event_data["action_type"].notna()
        & (event_data["action_type"] != "non_action")
        & event_data["period_id"].between(1, 5)
    )
    events = event_data.loc[is_action]

    # SPADL actions always have a start location, a team and a player
    has_location = events["start_x"].notna() & events["start_y"].notna()
    has_team = events["team_id"].notna() & (events["team_id"] != MISSING_INT)
    has_player = events["player_id"].notna() & (events["player_id"] != MISSING_INT)
    for is_valid, missing in [
        (has_location, "a start location"),
        (has_team, "a team_id"),
        (has_player, "a player_id"),
    ]:
        if not is_valid.all():
            warnings.warn(
                f"{(~is_valid).sum()} action(s) without {missing} are not "
                "converted to SPADL.",
                category=DataBallPyWarning,
            )
    events = events.loc[has_location & has_team & has_player]

    period_start = events["period_id"].map(PERIOD_START_MINUTES)
    start_x, start_y = _databallpy_to_spadl(
        events["start_x"], events["start_y"], pitch_dimensions
    )
    # actions without an end location end where they start
    end_x, end_y = _databallpy_to_spadl(
        events.get("end_x", events["start_x"]).fillna(events["start_x"]),
        events.get("end_y", events["start_y"]).fillna(events["start_y"]),
        pitch_dimensions,
    )

    return pd.DataFrame(
        {
            "game_id": pd.Series([game_id] * len(events), dtype=object),
            "original_event_id": events["original_event_id"].astype(object).values,
            "event_id": pd.array(events["event_id"].values, dtype="Int64"),
            "period_id": events["period_id"].astype(int).values,
            "time_seconds": (
                events["minutes"] * 60 + events["seconds"] - period_start * 60
            )
            .astype(float)
            .values,
            "team_id": events["team_id"].astype(object).values,
            "player_id": events["player_id"].astype(object).values,
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
            "type_id": events["action_type"].map(SPADL_ACTION_TYPES.index).values,
            "result_id": events["action_result"]
            .map(SPADL_RESULTS.index, na_action="ignore")
            .values,
            "bodypart_id": events["action_bodypart"]
            .map(SPADL_BODYPARTS.index, na_action="ignore")
            .values,
        }
    )


def _databallpy_to_spadl(
    x: pd.Series, y: pd.Series, pitch_dimensions: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Function to rescale databallpy coordinates to the SPADL field, positions
    outside the pitch are clipped to the field like socceraction does."""
    length, width = pitch_dimensions
    spadl_x = (x.to_numpy(dtype=float) / length + 0.5) * SPADL_FIELD_LENGTH
    spadl_y = (y.to_numpy(dtype=float) / width + 0.5) * SPADL_FIELD_WIDTH
    return (
        np.clip(spadl_x, 0, SPADL_FIELD_LENGTH),
        np.clip(spadl_y, 0, SPADL_FIELD_WIDTH),
    )


def _spadl_to_databallpy_coordinates(
    actions: pd.DataFrame, pitch_dimensions: list[float]
) -> pd.DataFrame:
    """Function to rescale SPADL coordinates back to databallpy coordinates."""
    length, width = pitch_dimensions
    for col in ["start_x", "end_x"]:
        actions[col] = (actions[col] / SPADL_FIELD_LENGTH - 0.5) * length
    for col in ["start_y", "end_y"]:
        actions[col] = (actions[col] / SPADL_FIELD_WIDTH - 0.5) * width
    return actions


def _fix_clearances(actions: pd.DataFrame) -> pd.DataFrame:
    """Function to set the end location of clearances to the start location of the
    next action, as in socceraction."""
    next_actions = actions.shift(-1)
    next_actions.iloc[-1:] = actions.iloc[-1:]
    clearance_idx = actions["type_id"] == SPADL_ACTION_TYPES.index("clearance")
    actions.loc[clearance_idx, "end_x"] = next_actions.loc[
        clearance_idx, "start_x"
    ].values
    actions.loc[clearance_idx, "end_y"] = next_actions.loc[
        clearance_idx, "start_y"
    ].values
    return actions


def _fix_interceptions(actions: pd.DataFrame) -> pd.DataFrame:
    """Function to set the result of interceptions to fail when the next action in the
    same period is of the other team, since the ball was not regained."""
    next_actions = actions.shift(-1)
    lost = (
        (actions["type_id"] == SPADL_ACTION_TYPES.index("interception"))
        & (next_actions["period_id"] == actions["period_id"])
        & (next_actions["team_id"] != actions["team_id"])
    )
    actions.loc[lost, "result_id"] = SPADL_RESULTS.index("fail")
    return actions


def _add_dribbles(actions: pd.DataFrame) -> pd.DataFrame:
    """Function to add dribbles between consecutive actions of the same team when the
    ball moved between the end of one action and the start of the next, as in
    socceraction."""
    next_actions = actions.shift(-1, fill_value=0)

    same_team = actions["team_id"] == next_actions["team_id"]
    not_offensive_foul = same_team & (
        next_actions["type_id"] != SPADL_ACTION_TYPES.index("foul")
    )
    # this mirrors socceraction, which excludes all shots and all headers
    not_headed_shot = (next_actions["type_id"] != SPADL_ACTION_TYPES.index("shot")) & (
        next_actions["bodypart_id"] != SPADL_BODYPARTS.index("head")
    )

    dx = actions["end_x"] - next_actions["start_x"]
    dy = actions["end_y"] - next_actions["start_y"]
    far_enough = dx**2 + dy**2 >= MIN_DRIBBLE_LENGTH**2
    not_too_far = dx**2 + dy**2 <= MAX_DRIBBLE_LENGTH**2

    dt = next_actions["time_seconds"] - actions["time_seconds"]
    same_phase = dt < MAX_DRIBBLE_DURATION
    same_period = actions["period_id"] == next_actions["period_id"]

    dribble_idx = (
        same_team
        & far_enough
        & not_too_far
        & same_phase
        & same_period
        & not_offensive_foul
        & not_headed_shot
    )

    prev = actions[dribble_idx]
    nex = next_actions[dribble_idx]
    dribbles = pd.DataFrame(
        {
            "game_id": nex["game_id"].values,
            "original_event_id": pd.Series([None] * len(prev), dtype=object),
            "event_id": pd.array([pd.NA] * len(prev), dtype="Int64"),
            "period_id": nex["period_id"].values,
            "action_id": prev["action_id"].values + 0.1,
            "time_seconds": (prev["time_seconds"].values + nex["time_seconds"].values)
            / 2,
            "team_id": nex["team_id"].values,
            "player_id": nex["player_id"].values,
            "start_x": prev["end_x"].values,
            "start_y": prev["end_y"].values,
            "end_x": nex["start_x"].values,
            "end_y": nex["start_y"].values,
            "bodypart_id": SPADL_BODYPARTS.index("foot"),
            "type_id": SPADL_ACTION_TYPES.index("dribble"),
            "result_id": SPADL_RESULTS.index("success"),
        }
    )

    if len(dribbles) > 0:
        actions = pd.concat([actions, dribbles], ignore_index=True, sort=False)
    actions = actions.sort_values(["period_id", "action_id"]).reset_index(drop=True)
    actions["action_id"] = range(len(actions))
    return actions


def add_carries_to_event_data(
    event_data: pd.DataFrame, pitch_dimensions: list[float]
) -> tuple[pd.DataFrame, dict]:
    """Function to add the dribbles (carries) that SPADL adds between consecutive
    actions of the same team as events to the event data. The carries are inserted
    just before the action that follows them and all event ids are renumbered.

    Args:
        event_data (pd.DataFrame): The event data with the SPADL vocabulary columns.
        pitch_dimensions (list[float]): The length and width of the pitch in meters.

    Returns:
        tuple[pd.DataFrame, dict]: The event data with the carries and the mapping
            from the old to the new event ids.
    """
    identity = dict(zip(event_data["event_id"], event_data["event_id"]))
    if "action_type" not in event_data.columns or event_data["action_type"].isna().all():
        return event_data, identity

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DataBallPyWarning)
            actions = _get_home_oriented_spadl_actions(
                event_data, pitch_dimensions, None
            )
    except (SchemaError, SchemaErrors) as e:
        warnings.warn(
            f"Could not convert the event data to SPADL, no carries are added: {e}",
            category=DataBallPyWarning,
        )
        return event_data, identity

    is_carry = actions["event_id"].isna()
    if not is_carry.any():
        return event_data, identity

    next_ids = actions["event_id"].shift(-1)[is_carry].astype(int).values
    coordinates = _spadl_to_databallpy_coordinates(
        actions.loc[is_carry, ["start_x", "start_y", "end_x", "end_y"]].copy(),
        pitch_dimensions,
    )

    # the carry is timed between the event just before it (the previous action or
    # an event without action, like a ball recovery) and the next action
    positions = pd.Series(np.arange(len(event_data)), index=event_data["event_id"])
    prev = event_data.iloc[positions[next_ids].values - 1].reset_index(drop=True)
    nex = event_data.iloc[positions[next_ids].values].reset_index(drop=True)
    carries = nex.copy()

    # the carry is performed by the player of the next action, all other columns
    # are emptied before the carry specific values are filled in
    keep = ["event_id", "period_id", "team_id", "team_name", "player_id", "player_name"]
    for col, dtype in carries.dtypes.items():
        if col in keep:
            continue
        if pd.api.types.is_extension_array_dtype(dtype):
            carries[col] = pd.NA
        elif pd.api.types.is_bool_dtype(dtype):
            carries[col] = False
        elif pd.api.types.is_integer_dtype(dtype):
            carries[col] = -1 if col == "event_type_id" else MISSING_INT
        elif pd.api.types.is_float_dtype(dtype):
            carries[col] = np.nan
        elif not pd.api.types.is_datetime64_any_dtype(dtype):
            carries[col] = None
        carries[col] = carries[col].astype(dtype)

    total_seconds = (
        prev["minutes"] * 60 + prev["seconds"] + nex["minutes"] * 60 + nex["seconds"]
    ) / 2
    carries["minutes"] = (total_seconds // 60).astype(int)
    carries["seconds"] = (total_seconds - carries["minutes"] * 60).astype(float)
    if "datetime" in carries.columns:
        carries["datetime"] = prev["datetime"] + (nex["datetime"] - prev["datetime"]) / 2
    for col in ["start_x", "start_y", "end_x", "end_y"]:
        carries[col] = coordinates[col].values
    carries["is_successful"] = pd.array([True] * len(carries), dtype="boolean")
    carries["original_event_id"] = MISSING_INT
    carries["original_event"] = "carry"
    carries["action_type"] = "dribble"
    carries["action_result"] = "success"
    carries["action_bodypart"] = "foot"
    if "databallpy_event" in carries.columns:
        carries["databallpy_event"] = SPADL_TO_DATABALLPY_EVENTS["dribble"]

    # insert the carries just before the next action and renumber all events
    order = np.concatenate(
        [np.arange(len(event_data)), positions[next_ids].values - 0.5]
    )
    sorter = np.argsort(order, kind="mergesort")
    is_original = sorter < len(event_data)
    event_data = pd.concat([event_data, carries], ignore_index=True).iloc[sorter]
    old_ids = event_data["event_id"].values[is_original]
    event_data = event_data.reset_index(drop=True)
    event_data["event_id"] = np.arange(len(event_data))
    return event_data, dict(zip(old_ids, np.flatnonzero(is_original)))


def _apply_playing_direction(
    actions: pd.DataFrame, home_team_id: int | str, playing_direction: str
) -> pd.DataFrame:
    """Function to mirror the actions to the requested playing direction. SPADL
    actions are home oriented, so the home team plays from left to right."""
    is_home = actions["team_id"] == home_team_id
    if playing_direction == "away_oriented":
        to_flip = pd.Series(True, index=actions.index)
    elif playing_direction == "possession_oriented":
        to_flip = ~is_home
    elif playing_direction == "out_of_possession_oriented":
        to_flip = is_home
    else:
        return actions

    for col in ["start_x", "end_x"]:
        actions.loc[to_flip, col] = SPADL_FIELD_LENGTH - actions.loc[to_flip, col]
    for col in ["start_y", "end_y"]:
        actions.loc[to_flip, col] = SPADL_FIELD_WIDTH - actions.loc[to_flip, col]
    return actions
