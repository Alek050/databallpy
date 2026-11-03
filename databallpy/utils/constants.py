MISSING_INT = -999
"""The value to use for missing integers in the data."""

DATABALLPY_EVENTS = ["pass", "shot", "dribble", "tackle", "own_goal"]
"""The databallpy events that are supported."""

PERIOD_START_MINUTES = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}
"""The minutes at which the periods start."""

SPADL_FIELD_LENGTH = 105.0
"""The length of the pitch in SPADL (in meters)."""

SPADL_FIELD_WIDTH = 68.0
"""The width of the pitch in SPADL (in meters)."""

SPADL_ACTION_TYPES = [
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "take_on",
    "foul",
    "tackle",
    "interception",
    "shot",
    "shot_penalty",
    "shot_freekick",
    "keeper_save",
    "keeper_claim",
    "keeper_punch",
    "keeper_pick_up",
    "clearance",
    "bad_touch",
    "non_action",
    "dribble",
    "goalkick",
]
"""The SPADL action types, the index in the list is the SPADL type_id. Copied from
socceraction (Decroos et al., 2019)."""

SPADL_RESULTS = [
    "fail",
    "success",
    "offside",
    "owngoal",
    "yellow_card",
    "red_card",
]
"""The SPADL action results, the index in the list is the SPADL result_id. Copied from
socceraction (Decroos et al., 2019)."""

SPADL_BODYPARTS = ["foot", "head", "other", "head/other", "foot_left", "foot_right"]
"""The SPADL body parts, the index in the list is the SPADL bodypart_id. Copied from
socceraction (Decroos et al., 2019)."""

ATOMIC_SPADL_ACTION_TYPES = [
    action_type
    for action_type in SPADL_ACTION_TYPES
    if action_type
    not in ["freekick_crossed", "freekick_short", "corner_crossed", "corner_short"]
] + [
    "receival",
    "out",
    "offside",
    "goal",
    "owngoal",
    "yellow_card",
    "red_card",
    "corner",
    "freekick",
]
"""The Atomic-SPADL action types, the index in the list is the Atomic-SPADL type_id.
Based on socceraction (Decroos et al., 2020), but without the corner and freekick
types that are simplified to corner and freekick, and with a single interception
type. Therefore, the type_ids differ from socceraction."""

SPADL_COLUMNS = ["action_type", "action_result", "action_bodypart"]
"""The SPADL vocabulary columns in the event data."""

SPADL_TO_DATABALLPY_EVENTS = {
    "pass": "pass",
    "cross": "pass",
    "throw_in": "pass",
    "freekick_crossed": "pass",
    "freekick_short": "pass",
    "corner_crossed": "pass",
    "corner_short": "pass",
    "goalkick": "pass",
    "shot": "shot",
    "shot_penalty": "shot",
    "shot_freekick": "shot",
    "take_on": "dribble",
    "dribble": "dribble",
    "tackle": "tackle",
}
"""The databallpy event of the SPADL action types. Actions with an owngoal result are
always an own_goal."""

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
