import numpy as np
import pandas as pd

from databallpy.features.expected_values import get_shot_features, get_xt
from databallpy.utils.constants import MISSING_INT

EVENT_TYPES = ["shot", "pass", "dribble"]
"""The event types that have extra information on top of the event data."""

_COMMON_EXTRA_COLUMNS = [
    "event_id",
    "outcome",
    "related_event_id",
    "body_part",
    "possession_type",
    "set_piece",
]

EXTRA_COLUMNS = {
    "shot": _COMMON_EXTRA_COLUMNS
    + ["outcome_str", "y_target", "z_target", "first_touch"],
    "pass": _COMMON_EXTRA_COLUMNS + ["outcome_str", "pass_type", "receiver_player_id"],
    "dribble": _COMMON_EXTRA_COLUMNS + ["duel_type", "with_opponent"],
}
"""The columns of the extra information per event type. The extra information is
stored apart from the event data, and is joined with the event data on the
``event_id`` to get the per event type dataframes (like ``game.shot_events``)."""

_EXTRA_DEFAULTS = {
    "y_target": np.nan,
    "z_target": np.nan,
    "first_touch": False,
    "receiver_player_id": MISSING_INT,
}

_EVENT_DATA_COLUMNS = [
    "event_id",
    "period_id",
    "minutes",
    "seconds",
    "datetime",
    "start_x",
    "start_y",
    "team_id",
    "team_side",
    "player_id",
    "jersey",
]


def create_event_extras(events: dict[int, dict], event_type: str) -> pd.DataFrame:
    """Create the dataframe with the extra information of one event type from the
    parsed events.

    Args:
        events (dict[int, dict]): The parsed events, the key is the event id and the
            value a dictionary with (at least) the extra information of the event.
        event_type (str): The event type, one of "shot", "pass" or "dribble".

    Returns:
        pd.DataFrame: The extra information, with the columns of
            ``EXTRA_COLUMNS[event_type]``.
    """
    columns = EXTRA_COLUMNS[event_type]
    return pd.DataFrame(
        [
            {
                "event_id": event_id,
                **{col: event.get(col, _EXTRA_DEFAULTS.get(col)) for col in columns[1:]},
            }
            for event_id, event in events.items()
        ],
        columns=columns,
    ).astype({"event_id": "int64"})


def create_all_event_extras(
    events: dict[str, dict[int, dict]],
) -> dict[str, pd.DataFrame]:
    """Create the dataframes with the extra information of all event types.

    Args:
        events (dict[str, dict[int, dict]]): The parsed events per event type, with
            the keys "shot_events", "pass_events" and "dribble_events".

    Returns:
        dict[str, pd.DataFrame]: The extra information per event type, with the keys
            "shot_events", "pass_events" and "dribble_events".
    """
    return {
        f"{event_type}_events": create_event_extras(
            events.get(f"{event_type}_events", {}), event_type
        )
        for event_type in EVENT_TYPES
    }


def get_event_extras_from_dataframe(
    event_df: pd.DataFrame, event_type: str
) -> pd.DataFrame:
    """Get the extra information of one event type from a dataframe with all
    attributes of the events, like the ``shot_events`` of games saved with databallpy
    versions before v0.8.2.

    Args:
        event_df (pd.DataFrame): The dataframe with the attributes of the events.
        event_type (str): The event type, one of "shot", "pass" or "dribble".

    Returns:
        pd.DataFrame: The extra information, with the columns of
            ``EXTRA_COLUMNS[event_type]``.
    """
    columns = EXTRA_COLUMNS[event_type]
    if event_df is None or len(event_df) == 0:
        return pd.DataFrame(columns=columns).astype({"event_id": "int64"})
    return pd.DataFrame(event_df.reindex(columns=columns).reset_index(drop=True))


def get_event_type_dataframe(
    event_data: pd.DataFrame,
    extras: pd.DataFrame,
    event_type: str,
    home_team_id: int | str,
    players: pd.DataFrame,
    pitch_dimensions: list[float] | tuple[float, float],
) -> pd.DataFrame:
    """Join the event data with the extra information of one event type.

    Args:
        event_data (pd.DataFrame): The event data of the game.
        extras (pd.DataFrame): The extra information of the event type.
        event_type (str): The event type, one of "shot", "pass" or "dribble".
        home_team_id (int | str): The id of the home team.
        players (pd.DataFrame): The players of both teams, with the columns ``id`` and
            ``shirt_num``.
        pitch_dimensions (list[float] | tuple[float, float]): The length and width of
            the pitch in meters.

    Returns:
        pd.DataFrame: The events of the event type with the columns of the event data
            that belong to the event, the extra information and the expected threat.
            Passes also get the end location, and shots the distance and angle to goal
            and the expected goals.
    """
    if extras is None or len(extras) == 0 or len(event_data) == 0:
        return pd.DataFrame()

    ed = pd.DataFrame(event_data).copy()
    ed["team_side"] = np.where(ed["team_id"] == home_team_id, "home", "away")
    jerseys = dict(zip(players["id"], players["shirt_num"]))
    ed["jersey"] = ed["player_id"].map(jerseys)
    if "xt" not in ed.columns:
        ed["xt"] = get_xt(ed, home_team_id)

    extra_cols = [c for c in extras.columns if c != "event_id"]
    common_cols = extra_cols[: len(_COMMON_EXTRA_COLUMNS) - 1]
    specific_cols = extra_cols[len(_COMMON_EXTRA_COLUMNS) - 1 :]
    ed_cols = _EVENT_DATA_COLUMNS + (["end_x", "end_y"] if event_type == "pass" else [])
    ed_cols = [c for c in ed_cols if c in ed.columns]
    action_cols = ["action_type", "action_result", "action_bodypart"]

    df = ed[ed_cols + ["xt"] + action_cols].merge(extras, on="event_id", how="inner")
    if event_type == "shot":
        df = pd.concat(
            [df, get_shot_features(df, pitch_dimensions, home_team_id)], axis=1
        )
        specific_cols += ["ball_goal_distance", "shot_angle", "xg"]

    columns = (
        [c for c in _EVENT_DATA_COLUMNS if c in df.columns]
        + common_cols[:2]
        + ["xt"]
        + common_cols[2:]
        + specific_cols
    )
    if event_type == "pass":
        idx = columns.index("outcome_str") + 1
        end_cols = [c for c in ["end_x", "end_y"] if c in df.columns]
        columns = columns[:idx] + end_cols + columns[idx:]
    return df[columns].reset_index(drop=True)
