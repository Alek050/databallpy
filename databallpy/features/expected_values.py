import json
import os

import numpy as np
import pandas as pd

from databallpy.features.angle import get_smallest_angle
from databallpy.models.utils import get_xt_prediction, scale_and_predict_logreg

path = os.path.join(os.path.dirname(__file__), "..", "models")
OPEN_PLAY_XT = np.load(f"{path}/open_play_xT.npy")
FREE_KICK_XT = np.load(f"{path}/free_kick_xT.npy")
THROW_IN_XT = np.load(f"{path}/throw_in_xT.npy")
with open(f"{path}/xg_params.json", "r") as f:
    XG_PARAMS = json.load(f)

XT_ACTION_TYPES = [
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "take_on",
    "shot",
    "shot_penalty",
    "shot_freekick",
    "clearance",
    "bad_touch",
    "dribble",
    "goalkick",
]
"""The SPADL action types of on-ball actions for which the xT is computed."""

GOAL_WIDTH = 7.32


def get_xt(event_data: pd.DataFrame, home_team_id: int | str) -> pd.Series:
    """Get the expected threat (xT) of the on-ball actions in the event data. The xT
    is based on the SPADL ``action_type`` and the start location of the action. Set
    pieces with a fixed value (penalties, kick offs, corners and goal kicks) get that
    value, throw ins and free kicks use their own xT grid and all other on-ball actions
    use the open play xT grid. Actions that are not on-ball actions get NaN.

    Args:
        event_data (pd.DataFrame): The event data, with the columns ``action_type``,
            ``team_id``, ``start_x`` and ``start_y``. The coordinates should be in
            meters with the origin in the centre of the pitch, and the home team
            playing from left to right.
        home_team_id (int | str): The id of the home team.

    Returns:
        pd.Series: The xT of every action, with the index of the event data.
    """
    action_type = event_data["action_type"]
    is_home = (event_data["team_id"] == home_team_id).to_numpy()
    x = np.where(is_home, event_data["start_x"], -event_data["start_x"])
    y = np.where(is_home, event_data["start_y"], -event_data["start_y"])
    valid_xy = ~(np.isnan(x) | np.isnan(y))
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)

    def _grid(grid: np.ndarray) -> np.ndarray:
        return np.where(valid_xy, get_xt_prediction(x, y, grid), np.nan)

    # SPADL has no kick off action type, a pass from the centre spot is a kick off
    is_kick_off = (action_type == "pass").to_numpy() & (np.hypot(x, y) < 1.0)

    conditions = [
        action_type == "shot_penalty",
        is_kick_off,
        action_type.isin(["corner_crossed", "corner_short"]),
        action_type == "goalkick",
        action_type == "throw_in",
        action_type.isin(["freekick_crossed", "freekick_short", "shot_freekick"]),
        action_type.isin(XT_ACTION_TYPES),
    ]
    choices = [
        0.797,
        0.001,
        0.049,
        0.0,
        _grid(THROW_IN_XT),
        _grid(FREE_KICK_XT),
        _grid(OPEN_PLAY_XT),
    ]
    return pd.Series(
        np.select(conditions, choices, default=np.nan),
        index=event_data.index,
        dtype=float,
    )


def get_shot_features(
    shots: pd.DataFrame,
    pitch_dimensions: list[float] | tuple[float, float],
    home_team_id: int | str,
) -> pd.DataFrame:
    """Get the distance and angle to goal and the expected goals (xG) of shots. The
    xG model is chosen with the SPADL columns: own goals get 0, penalties 0.79, free
    kicks use the free kick model, shots that are not taken with the foot use the
    header model and all other shots use the foot model.

    Args:
        shots (pd.DataFrame): The shots, with the columns ``team_id``, ``start_x``,
            ``start_y``, ``action_type``, ``action_result`` and ``action_bodypart``.
            The coordinates should be in meters with the origin in the centre of the
            pitch, and the home team playing from left to right.
        pitch_dimensions (list[float] | tuple[float, float]): The length and width of
            the pitch in meters.
        home_team_id (int | str): The id of the home team.

    Returns:
        pd.DataFrame: The columns ``ball_goal_distance``, ``shot_angle`` (in degrees)
            and ``xg``, with the index of the shots.
    """
    is_home = (shots["team_id"] == home_team_id).to_numpy()
    direction = np.where(is_home, 1.0, -1.0)
    ball_xy = shots[["start_x", "start_y"]].to_numpy(dtype=float)
    goal_x = direction * pitch_dimensions[0] / 2.0

    ball_goal_distance = np.hypot(goal_x - ball_xy[:, 0], ball_xy[:, 1])
    left_post_vector = np.stack(
        [goal_x - ball_xy[:, 0], direction * GOAL_WIDTH / 2 - ball_xy[:, 1]], axis=1
    )
    right_post_vector = np.stack(
        [goal_x - ball_xy[:, 0], -direction * GOAL_WIDTH / 2 - ball_xy[:, 1]], axis=1
    )
    shot_angle = (
        np.atleast_1d(
            get_smallest_angle(
                left_post_vector, right_post_vector, angle_format="degree"
            )
        ).astype(float)
        if len(shots) > 0
        else np.array([], dtype=float)
    )

    features = np.stack([ball_goal_distance, shot_angle], axis=1)
    is_own_goal = (shots["action_result"] == "owngoal").to_numpy()
    is_penalty = (shots["action_type"] == "shot_penalty").to_numpy()
    is_free_kick = (shots["action_type"] == "shot_freekick").to_numpy()
    is_foot = (
        shots["action_bodypart"].fillna("").astype(str).str.contains("foot").to_numpy()
    )
    # bodypart is missing when the provider does not specify it, use the foot model
    is_foot |= shots["action_bodypart"].isna().to_numpy()

    if len(shots) > 0:
        xg_free_kick = scale_and_predict_logreg(features, XG_PARAMS["xg_by_free_kick"])
        xg_head = scale_and_predict_logreg(features, XG_PARAMS["xg_by_head"])
        xg_foot = scale_and_predict_logreg(features, XG_PARAMS["xg_by_foot"])
    else:
        xg_free_kick = xg_head = xg_foot = np.array([], dtype=float)

    xg = np.select(
        [is_own_goal, is_penalty, is_free_kick, ~is_foot],
        [0.0, 0.79, xg_free_kick, xg_head],
        default=xg_foot,
    )
    missing = np.isnan(ball_goal_distance) | np.isnan(shot_angle)
    xg = np.where(missing & ~is_own_goal, np.nan, xg)

    return pd.DataFrame(
        {
            "ball_goal_distance": ball_goal_distance,
            "shot_angle": shot_angle,
            "xg": xg,
        },
        index=shots.index,
    )
