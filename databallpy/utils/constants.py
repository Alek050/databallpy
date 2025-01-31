MISSING_INT = -999
"""The value to use for missing integers in the data."""

DATABALLPY_EVENTS = ["pass", "shot", "dribble", "tackle", "own_goal"]
"""The databallpy events that are supported."""

DATABALLPY_SET_PIECES = [
    "goal_kick",
    "free_kick",
    "throw_in",
    "corner_kick",
    "kick_off",
    "penalty",
    "goal_kick",
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
    "kick_off",
    "goal_kick",
    "rebound",
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

DATABALLPY_POSITIONS = ["goalkeeper", "defender", "midfielder", "forward", "unspecified"]

OPEN_GAME_IDS_DFL = {
    "J03WMX": "1. FC Köln vs. FC Bayern München",
    "J03WN1": "VfL Bochum 1848 vs. Bayer 04 Leverkusen",
    "J03WPY": "Fortuna Düsseldorf vs. 1. FC Nürnberg",
    "J03WOH": "Fortuna Düsseldorf vs. SSV Jahn Regensburg",
    "J03WQQ": "Fortuna Düsseldorf vs. FC St. Pauli",
    "J03WOY": "Fortuna Düsseldorf vs. F.C. Hansa Rostock",
    "J03WR9": "Fortuna Düsseldorf vs. 1. FC Kaiserslautern",
}
