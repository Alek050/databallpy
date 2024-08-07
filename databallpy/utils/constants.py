MISSING_INT = -999
"""The value to use for missing integers in the data."""

DATABALLPY_EVENTS = ["pass", "shot", "dribble", "tackle"]
"""The databallpy events that are supported."""

DATABALLPY_SET_PIECES = [
    "goal_kick",
    "free_kick",
    "throw_in",
    "corner_kick",
    "kick_off",
    "penalty",
    "no_set_piece",
    "unspecified",
]
"""The set piece types that are supported in databallpy."""

DATBALLPY_BODY_PARTS = [
    "right_foot",
    "left_foot",
    "foot",
    "head",
    "other",
    "unspecified",
]
"""The body parts that are supported in databallpy."""

DATABALLPY_POSSESSION_TYPES = [
    "open_play",
    "counter_attack",
    "corner_kick",
    "free_kick",
    "throw_in",
    "penalty",
    "unspecified",
]
"""The possession types that are supported in databallpy."""

DATABALLPY_SHOT_OUTCOMES = [
    "goal",
    "miss_off_target",
    "miss_hit_post",
    "miss_on_target",
    "blocked",
    "own_goal",
    "miss",
    "unspecified",
]
"""The shot outcome strings that are supported in databallpy."""

DATABALLPY_PASS_OUTCOMES = [
    "successful",
    "unsuccessful",
    "offside",
    "results_in_shot",
    "assist",
    "fair_play",
    "unspecified",
]
"""The pass outcome strings that are supported in databallpy."""

DATABALLPY_PASS_TYPES = [
    "long_ball",
    "cross",
    "through_ball",
    "chipped",
    "lay-off",
    "lounge",
    "flick_on",
    "pull_back",
    "switch_off_play",
    "unspecified",
]
"""The pass type strings that are supported in databallpy."""
