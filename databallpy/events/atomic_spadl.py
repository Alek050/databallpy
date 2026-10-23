"""Conversion of databallpy event data to Atomic-SPADL actions.

Atomic-SPADL (Decroos et al., 2020) splits the SPADL actions into atomic actions
without a result: a pass is followed by a receival, interception, out or offside
and a shot by a goal or out. The conversion follows socceraction, with a few
fixes: there is a single interception type, the corner and freekick types that are
simplified to corner and freekick are not in the vocabulary, a pass followed by a
corner of the other team is out, shot_freekick is kept as a shot, and no dribble
is added from an out to the throw-in that follows it.
"""

import pandas as pd

from databallpy.events.spadl import (
    _add_dribbles,
    _apply_playing_direction,
    _check_spadl_options,
    _get_home_oriented_spadl_actions,
)
from databallpy.schemas.spadl import AtomicSPADLSchema
from databallpy.utils.constants import (
    ATOMIC_SPADL_ACTION_TYPES,
    SPADL_ACTION_TYPES,
    SPADL_BODYPARTS,
    SPADL_FIELD_LENGTH,
    SPADL_FIELD_WIDTH,
    SPADL_RESULTS,
)

ATOMIC_SPADL_ACTION_COLUMNS = [
    "game_id",
    "original_event_id",
    "event_id",
    "action_id",
    "period_id",
    "time_seconds",
    "team_id",
    "player_id",
    "x",
    "y",
    "dx",
    "dy",
    "type_id",
    "bodypart_id",
]

PASS_LIKE_TYPES = [
    "pass",
    "cross",
    "throw_in",
    "freekick_short",
    "freekick_crossed",
    "corner_crossed",
    "corner_short",
    "clearance",
    "goalkick",
]
INTERCEPTION_LIKE_TYPES = [
    "interception",
    "tackle",
    "keeper_punch",
    "keeper_save",
    "keeper_claim",
    "keeper_pick_up",
]
SHOT_LIKE_TYPES = ["shot", "shot_freekick", "shot_penalty"]
RESTART_TYPES = ["corner_crossed", "corner_short", "goalkick"]
SIMPLIFIED_TYPES = {
    "corner_crossed": "corner",
    "corner_short": "corner",
    "freekick_crossed": "freekick",
    "freekick_short": "freekick",
}


def event_data_to_atomic_spadl(
    event_data: pd.DataFrame,
    home_team_id: int | str,
    pitch_dimensions: list[float],
    game_id: int | str | None = None,
    playing_direction: str = "home_oriented",
    coordinate_system: str = "databallpy",
) -> pd.DataFrame:
    """Function to convert event data to Atomic-SPADL actions.

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
        pd.DataFrame: The Atomic-SPADL actions. The original_event_id and event_id
            columns link the actions to the event data and are null for added
            dribbles.
    """
    playing_direction = _check_spadl_options(playing_direction, coordinate_system)
    actions = _get_home_oriented_spadl_actions(event_data, pitch_dimensions, game_id)
    actions["type_name"] = actions["type_id"].map(dict(enumerate(SPADL_ACTION_TYPES)))

    actions = _add_from_passes(actions)
    actions = _add_dribbles_after_receivals(actions)
    actions = _add_from_shots(actions)
    actions = _add_from_cards(actions)

    actions = _apply_playing_direction(actions, home_team_id, playing_direction)
    actions = AtomicSPADLSchema.validate(_to_atomic_columns(actions))
    if coordinate_system == "databallpy":
        actions = _atomic_spadl_to_databallpy_coordinates(actions, pitch_dimensions)
    return actions


def _get_extra_actions(prev: pd.DataFrame) -> pd.DataFrame:
    """Function to create atomic actions at the end location of the previous
    actions, the team, player, type and bodypart are set by the caller."""
    return pd.DataFrame(
        {
            "game_id": prev["game_id"],
            "original_event_id": prev["original_event_id"],
            "event_id": prev["event_id"],
            "period_id": prev["period_id"],
            "action_id": prev["action_id"] + 0.1,
            "time_seconds": prev["time_seconds"],
            "team_id": prev["team_id"],
            "player_id": prev["player_id"],
            "start_x": prev["end_x"],
            "start_y": prev["end_y"],
            "end_x": prev["end_x"],
            "end_y": prev["end_y"],
            "bodypart_id": prev["bodypart_id"],
            # the type_id is set to the atomic type at the end, and the extra
            # actions are never fouls or shots in _add_dribbles
            "type_id": -1,
        }
    )


def _insert_actions(actions: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Function to insert the extra actions after the actions they are based on."""
    if len(extra) > 0:
        actions = pd.concat([actions, extra], ignore_index=True, sort=False)
    actions = actions.sort_values(
        ["period_id", "action_id"], kind="mergesort"
    ).reset_index(drop=True)
    actions["action_id"] = range(len(actions))
    return actions


def _add_from_passes(actions: pd.DataFrame) -> pd.DataFrame:
    """Function to add the receival, interception, out or offside after pass-like
    actions, unless the next action is an interception-like action."""
    next_actions = actions.shift(-1)
    extra_idx = (
        actions["type_name"].isin(PASS_LIKE_TYPES)
        & (actions["period_id"] == next_actions["period_id"])
        & ~next_actions["type_name"].isin(INTERCEPTION_LIKE_TYPES)
    )
    prev = actions[extra_idx]
    nex = next_actions[extra_idx]

    same_team = prev["team_id"] == nex["team_id"]
    offside = prev["result_id"] == SPADL_RESULTS.index("offside")
    out = (nex["type_name"].isin(RESTART_TYPES) & ~same_team) | (
        nex["type_name"] == "throw_in"
    )
    interception = ~same_team & ~out & ~offside

    extra = _get_extra_actions(prev)
    extra["time_seconds"] = (prev["time_seconds"] + nex["time_seconds"]) / 2
    extra["bodypart_id"] = SPADL_BODYPARTS.index("foot")
    extra["type_name"] = (
        pd.Series("receival", index=prev.index)
        .mask(~same_team, "interception")
        .mask(out, "out")
        .mask(offside, "offside")
    )
    extra["team_id"] = prev["team_id"].mask(interception, nex["team_id"])
    extra["player_id"] = nex["player_id"].mask(out | offside, prev["player_id"])
    return _insert_actions(actions, extra)


def _add_dribbles_after_receivals(actions: pd.DataFrame) -> pd.DataFrame:
    """Function to add dribbles after the added receivals and interceptions when
    the player moved the ball before the next action, the other dribbles are
    already added in the SPADL conversion."""
    actions = _add_dribbles(actions)
    is_added = actions["type_name"].isna()
    after_receival = actions["type_name"].shift(1).isin(["receival", "interception"])
    actions = actions[~is_added | after_receival].copy()
    actions["type_name"] = actions["type_name"].fillna("dribble")
    actions["action_id"] = range(len(actions))
    return actions.reset_index(drop=True)


def _add_from_shots(actions: pd.DataFrame) -> pd.DataFrame:
    """Function to add a goal after scored shots, an owngoal after own goals and an
    out after shots that are followed by a corner or goalkick."""
    next_actions = actions.shift(-1)
    shot = actions["type_name"].isin(SHOT_LIKE_TYPES)
    goal = shot & (actions["result_id"] == SPADL_RESULTS.index("success"))
    owngoal = actions["result_id"] == SPADL_RESULTS.index("owngoal")
    out = (
        shot
        & next_actions["type_name"].isin(RESTART_TYPES)
        & (actions["period_id"] == next_actions["period_id"])
    )
    extra_idx = goal | owngoal | out

    extra = _get_extra_actions(actions[extra_idx])
    extra["type_name"] = (
        pd.Series("out", index=extra.index)
        .mask(goal[extra_idx], "goal")
        .mask(owngoal[extra_idx], "owngoal")
    )
    return _insert_actions(actions, extra)


def _add_from_cards(actions: pd.DataFrame) -> pd.DataFrame:
    """Function to add a yellow or red card after fouls with a card."""
    yellow = actions["result_id"] == SPADL_RESULTS.index("yellow_card")
    red = actions["result_id"] == SPADL_RESULTS.index("red_card")
    extra_idx = yellow | red

    extra = _get_extra_actions(actions[extra_idx])
    extra["type_name"] = pd.Series("yellow_card", index=extra.index).mask(
        red[extra_idx], "red_card"
    )
    return _insert_actions(actions, extra)


def _to_atomic_columns(actions: pd.DataFrame) -> pd.DataFrame:
    """Function to convert the start and end locations to a location and
    displacement and to set the Atomic-SPADL type_id."""
    type_names = actions["type_name"].replace(SIMPLIFIED_TYPES)
    return actions.assign(
        x=actions["start_x"],
        y=actions["start_y"],
        dx=actions["end_x"] - actions["start_x"],
        dy=actions["end_y"] - actions["start_y"],
        type_id=type_names.map(ATOMIC_SPADL_ACTION_TYPES.index),
    )[ATOMIC_SPADL_ACTION_COLUMNS]


def _atomic_spadl_to_databallpy_coordinates(
    actions: pd.DataFrame, pitch_dimensions: list[float]
) -> pd.DataFrame:
    """Function to rescale Atomic-SPADL coordinates to databallpy coordinates."""
    length, width = pitch_dimensions
    actions["x"] = (actions["x"] / SPADL_FIELD_LENGTH - 0.5) * length
    actions["y"] = (actions["y"] / SPADL_FIELD_WIDTH - 0.5) * width
    actions["dx"] = actions["dx"] / SPADL_FIELD_LENGTH * length
    actions["dy"] = actions["dy"] / SPADL_FIELD_WIDTH * width
    return actions
