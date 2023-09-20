from typing import Tuple

import bs4
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from databallpy.load_data.event_data.dribble_event import DribbleEvent
from databallpy.load_data.event_data.pass_event import PassEvent
from databallpy.load_data.event_data.shot_event import ShotEvent
from databallpy.load_data.metadata import Metadata
from databallpy.utils.tz_modification import utc_to_local_datetime
from databallpy.utils.utils import MISSING_INT

EVENT_TYPE_IDS = {
    1: "pass",
    2: "offside pass",
    3: "take on",
    4: "foul",
    5: "out",
    6: "corner awarded",
    7: "tackle",
    8: "interception",
    9: "turnover",
    10: "save",
    11: "claim",
    12: "clearance",
    13: "miss",
    14: "post",
    15: "attempt saved",
    16: "goal",
    17: "card",
    18: "player off",
    19: "player on",
    20: "player retired",
    21: "player returns",
    22: "player becomes goalkeeper",
    23: "goalkeeper becomes player",
    24: "condition change",
    25: "official change",
    27: "start delay",
    28: "end delay",
    30: "end",
    32: "start",
    34: "team set up",
    35: "player changed position",
    36: "player changed jersey number",
    37: "collection end",
    38: "temp_goal",
    39: "temp_attempt",
    40: "formation change",
    41: "punch",
    42: "good skill",
    43: "deleted event",
    44: "aerial",
    45: "challenge",
    47: "rescinded card",
    49: "ball recovery",
    50: "dispossessed",
    51: "error",
    52: "keeper pick-up",
    53: "cross not claimed",
    54: "smother",
    55: "offside provoked",
    56: "shield ball opp",
    57: "foul throw in",
    58: "penalty faced",
    59: "keeper sweeper",
    60: "chance missed",
    61: "ball touch",
    63: "temp_save",
    64: "resume",
    65: "contentious referee decision",
    66: "possession data",
    67: "50/50",
    68: "referee drop ball",
    69: "failed to block",
    70: "injury time announcement",
    71: "coach setup",
    72: "caught offside",
    73: "other ball contact",
    74: "blocked pass",
    75: "delayed start",
    76: "early end",
    77: "player off pitch",
}

OPTA_TO_DATABALLPY_MAP = {
    "pass": "pass",
    "take on": "dribble",
    "offside pass": "pass",
    "miss": "shot",
    "post": "shot",
    "attempt saved": "shot",
    "goal": "shot",
    "own goal": "own_goal",
}

SHOT_OUTCOMES = {
    13: "miss_off_target",
    14: "miss_hit_post",
    15: "miss_on_target",
    16: "goal",
}

SHOT_ORIGINS_QUALIFIERS = {
    9: "penalty",
    22: "regular_play",
    23: "counter_attack",
    24: "crossed_free_kick",
    25: "corner_kick",
    26: "free_kick",
}

SET_PIECE_QUALIFIERS = {
    124: "goal_kick",
    5: "free_kick",
    107: "throw_in",
    6: "corner_kick",
    9: "penalty",
    279: "kick_off",
}

BODY_PART_QUALIFIERS = {
    3: "head",
    15: "head",
    20: "right_foot",
    21: "other",
    72: "left_foot",
}

PASS_TYPE_QUALIFIERS = {
    1: "long_ball",
    2: "cross",
    4: "through_ball",
    155: "chipped",
    156: "lay-off",
    157: "lounge",
    168: "flick_on",
    195: "pull_back",
    196: "switch_off_play",
}

CREATED_OPPERTUNITY_QUALIFIERS = {
    29: "assisted",
    215: "individual_play",
}

DRIBBLE_DUEL_TYPE_QUALIFIERS = {
    286: "offensive",
    285: "defensive",
}


Y_TARGET_QUALIFIER = 102
Z_TARGET_QUALIFIER = 103
X_END_QUALIFIER = 140
Y_END_QUALIFIER = 141
PASS_LENGTH_QUALIFIER = 212
PASS_ANGLE_QUALIFIER = 213
ASSIST_TO_SHOT_QUALIFIER = 210
FAIR_PLAY_QUALIFIER = 238
RELATED_EVENT_ID_QUALIFIER = 55
Y_TARGET_QUALIFIER = 102
Z_TARGET_QUALIFIER = 103
FIRST_TOUCH_QUALIFIER = 328
SHOT_BLOCKED_QUALIFIER = 82
OWN_GOAL_QUALIFIER = 280
RELATED_EVENT_QUALIFIER = 55
OPPOSITE_RELATED_EVENT_ID = 233  # used for dribbles


def load_opta_event_data(
    f7_loc: str, f24_loc: str, pitch_dimensions: list = [106.0, 68.0]
) -> Tuple[pd.DataFrame, Metadata, dict]:
    """This function retrieves the metadata and event data of a specific match. The x
    and y coordinates provided have been scaled to the dimensions of the pitch, with
    (0, 0) being the center. Additionally, the coordinates have been standardized so
    that the home team is represented as playing from left to right for the entire
    match, and the away team is represented as playing from right to left.

    Args:
        f7_loc (str): location of the f7.xml file.
        f24_loc (str): location of the f24.xml file.
        pitch_dimensions (list, optional): the length and width of the pitch in meters

    Returns:
        Tuple[pd.DataFrame, Metadata, dict]: the event data of the match, the metadata,
        and the databallpy_events.
    """

    assert isinstance(f7_loc, str), f"f7_loc should be a string, not a {type(f7_loc)}"
    assert isinstance(
        f24_loc, str
    ), f"f24_loc should be a string, not a {type(f24_loc)}"

    assert f7_loc[-4:] == ".xml", "f7 opta file should by of .xml format"
    if not f24_loc == "pass":
        assert f24_loc[-4:] == ".xml", "f24 opta file should be of .xml format"

    metadata = _load_metadata(f7_loc, pitch_dimensions=pitch_dimensions)
    if f24_loc != "pass":
        event_data, databallpy_events = _load_event_data(
            f24_loc,
            metadata.country,
            metadata.away_team_id,
            pitch_dimensions=pitch_dimensions,
        )

        # update timezones for databallpy_events
        for event_types in databallpy_events.values():
            for event in event_types.values():
                event.datetime = utc_to_local_datetime(event.datetime, metadata.country)

        # Add player names to the event data dataframe
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

        event_data["player_name"] = None
        event_data.loc[home_mask, "player_name"] = event_data.loc[
            home_mask, "player_id"
        ].map(home_players)
        event_data.loc[away_mask, "player_name"] = event_data.loc[
            away_mask, "player_id"
        ].map(away_players)
        event_data["player_name"] = event_data["player_name"].replace({np.nan: None})

        # Rescale the x and y coordinates relative to the pitch dimensions
        # The original dimension of the x and y coordinates range from 0 to 100
        event_data.loc[:, ["start_x"]] = (
            event_data.loc[:, ["start_x"]] / 100 * pitch_dimensions[0]
        ) - (pitch_dimensions[0] / 2.0)
        event_data.loc[:, ["start_y"]] = (
            event_data.loc[:, ["start_y"]] / 100 * pitch_dimensions[1]
        ) - (pitch_dimensions[1] / 2.0)

        # Change direction of play of the away team so it is represented from right to
        # left
        event_data.loc[
            event_data["team_id"] == metadata.away_team_id, ["start_x", "start_y"]
        ] *= -1
    else:
        event_data = pd.DataFrame()
        databallpy_events = {}

    return event_data, metadata, databallpy_events


def _load_metadata(f7_loc: str, pitch_dimensions: list) -> Metadata:
    """Function to load metadata from the f7.xml opta file

    Args:
        f7_loc (str): location of the f7.xml opta file
        pitch_dimensions (list): the length and width of the pitch in meters

    Returns:
        MetaData: all metadata information of the current match
    """
    file = open(f7_loc, "r")
    lines = file.read()
    soup = BeautifulSoup(lines, "xml")

    if len(soup.find_all("SoccerDocument")) > 1:
        # Multiple matches found in f7.xml file
        # Eliminate the rest of the `SoccerDocument` elements
        for match in soup.find_all("SoccerDocument")[1:]:
            match.decompose()

    # Obtain match id
    match_id = int(soup.find("SoccerDocument").attrs["uID"][1:])
    country = soup.find("Country").text
    # Obtain match start and end of period datetime
    periods = {
        "period": [1, 2, 3, 4, 5],
        "start_datetime_ed": [],
        "end_datetime_ed": [],
    }
    start_period_1 = soup.find("Stat", attrs={"Type": "first_half_start"})
    end_period_1 = soup.find("Stat", attrs={"Type": "first_half_stop"})
    start_period_2 = soup.find("Stat", attrs={"Type": "second_half_start"})
    end_period_2 = soup.find("Stat", attrs={"Type": "second_half_stop"})
    if not all([start_period_1, end_period_1, start_period_2, end_period_2]):
        file.close()
        raise ValueError(
            "The f7.xml opta file does not contain the start and end of period datetime"
        )
    for start, end in zip(
        [start_period_1, start_period_2], [end_period_1, end_period_2]
    ):
        periods["start_datetime_ed"].append(pd.to_datetime(start.contents[0], utc=True))
        periods["end_datetime_ed"].append(pd.to_datetime(end.contents[0], utc=True))
    for _ in range(3):
        periods["start_datetime_ed"].append(pd.to_datetime("NaT", utc=True))
        periods["end_datetime_ed"].append(pd.to_datetime("NaT", utc=True))

    periods = pd.DataFrame(periods)

    # Set datetime to right timezone
    periods["start_datetime_ed"] = utc_to_local_datetime(
        periods["start_datetime_ed"], country
    )
    periods["end_datetime_ed"] = utc_to_local_datetime(
        periods["end_datetime_ed"], country
    )

    # Opta has a TeamData and Team attribute in the f7 file
    team_datas = soup.find_all("TeamData")
    teams = soup.find_all("Team")
    teams_info = {}
    teams_player_info = {}
    for team_data, team in zip(team_datas, teams):
        # Team information
        team_name = team.findChildren("Name")[0].contents[0]
        team_info = {}
        team_info["team_name"] = team_name
        team_info["side"] = team_data["Side"].lower()
        team_info["formation"] = team_data["Formation"]
        team_info["score"] = int(team_data["Score"])
        team_info["team_id"] = int(team_data["TeamRef"][1:])
        teams_info[team_info["side"]] = team_info

        # Player information
        players_data = [
            player.attrs for player in team_data.findChildren("MatchPlayer")
        ]
        players_names = {}
        for player in team.findChildren("Player"):
            player_id = int(player.attrs["uID"][1:])
            first_name = player.contents[1].contents[1].text

            if "Last" in str(player.contents[1].contents[3]):
                last_name_idx = 3
            else:
                last_name_idx = 5
            last_name = player.contents[1].contents[last_name_idx].contents[0]
            if first_name:
                players_names[str(player_id)] = f"{first_name} {last_name}"
            else:
                players_names[str(player_id)] = last_name

        player_info = _get_player_info(players_data, players_names)
        teams_player_info[team_info["side"]] = player_info

    file.close()

    metadata = Metadata(
        match_id=match_id,
        pitch_dimensions=pitch_dimensions,
        periods_frames=periods,
        frame_rate=MISSING_INT,
        home_team_id=teams_info["home"]["team_id"],
        home_team_name=str(teams_info["home"]["team_name"]),
        home_players=teams_player_info["home"],
        home_score=teams_info["home"]["score"],
        home_formation=teams_info["home"]["formation"],
        away_team_id=teams_info["away"]["team_id"],
        away_team_name=str(teams_info["away"]["team_name"]),
        away_players=teams_player_info["away"],
        away_score=teams_info["away"]["score"],
        away_formation=teams_info["away"]["formation"],
        country=country,
    )
    return metadata


def _get_player_info(players_data: list, players_names: dict) -> pd.DataFrame:
    """Function to loop over all players and save data in a pd.DataFrame

    Args:
        players_data (list): for every player a dictionary with info about the player
        except the player name
        players_names (dict): dictionary with player id as key and the player name as
        value

    Returns:
        pd.DataFrame: all information of the players
    """
    result_dict = {
        "id": [],
        "full_name": [],
        "formation_place": [],
        "position": [],
        "starter": [],
        "shirt_num": [],
    }

    for player in players_data:
        player_id = int(player["PlayerRef"][1:])
        result_dict["id"].append(player_id)
        result_dict["full_name"].append(players_names[str(player_id)])
        result_dict["formation_place"].append(int(player["Formation_Place"]))
        position = (
            player["Position"]
            if player["Position"] != "Substitute"
            else player["SubPosition"]
        )
        result_dict["position"].append(position.lower())
        result_dict["starter"].append(player["Status"] == "Start")
        result_dict["shirt_num"].append(int(player["ShirtNumber"]))

    return pd.DataFrame(result_dict)


def _load_event_data(
    f24_loc: str,
    country: str,
    away_team_id: int,
    pitch_dimensions: list = [106.0, 68.0],
) -> Tuple[pd.DataFrame, dict]:
    """Function to load the f27 .xml, the events of the match.
    Note: this function does ignore most qualifiers for now.

    Args:
        f24_loc (str): location of the f24.xml file
        country (str): country of the match
        away_team_id (int): id of the away team
        pitch_dimensions (list, optional): dimensions of the pitch.
                                           Defaults to [106.0, 68.0].

    Returns:
        pd.DataFrame: all events of the match in a pd dataframe
        dict: dict with "shot_events" as key and a dict with the ShotEvent instances
    """

    dribble_events = {}
    shot_events = {}
    pass_events = {}

    with open(f24_loc, "r") as file:
        lines = file.read()
    soup = BeautifulSoup(lines, "xml")

    result_dict = {
        "event_id": [],
        "type_id": [],
        "databallpy_event": [],
        "period_id": [],
        "minutes": [],
        "seconds": [],
        "player_id": [],
        "team_id": [],
        "outcome": [],
        "start_x": [],
        "start_y": [],
        "datetime": [],
        "opta_event": [],
        "opta_id": [],
    }

    events = soup.find_all("Event")
    for event in events:
        result_dict["event_id"].append(int(event.attrs["id"]))
        result_dict["opta_id"].append(int(event.attrs["event_id"]))
        event_type_id = int(event.attrs["type_id"])
        result_dict["type_id"].append(event_type_id)

        if event_type_id in EVENT_TYPE_IDS.keys():
            event_name = EVENT_TYPE_IDS[event_type_id].lower()

            # check if goal is a own goal
            if event_type_id == 16:
                for qualifier in event.find_all("Q"):
                    if (
                        qualifier.attrs["qualifier_id"] == "280"
                        and qualifier.attrs["value"] == "OWN_GOAL"
                    ):
                        event_name = "own goal"
        else:
            event_name = "unknown event"

        result_dict["opta_event"].append(event_name)
        result_dict["period_id"].append(int(event.attrs["period_id"]))
        result_dict["minutes"].append(int(event.attrs["min"]))
        result_dict["seconds"].append(int(event.attrs["sec"]))

        if "player_id" in event.attrs.keys():
            result_dict["player_id"].append(int(event.attrs["player_id"]))
        else:
            result_dict["player_id"].append(MISSING_INT)

        result_dict["team_id"].append(int(event.attrs["team_id"]))
        if event_name in ["pass", "take on"]:
            result_dict["outcome"].append(int(event.attrs["outcome"]))
        else:
            result_dict["outcome"].append(MISSING_INT)
        result_dict["start_x"].append(float(event.attrs["x"]))
        result_dict["start_y"].append(float(event.attrs["y"]))
        datetime = (
            pd.to_datetime(event.attrs["timestamp_utc"], utc=True)
            if event.attrs["timestamp_utc"]
            else pd.to_datetime(event.attrs["timestamp"], utc=True)
        )
        result_dict["datetime"].append(datetime)

        # get extra information for databallpy events

        if event_name in ["pass", "offside pass"]:
            pass_events[int(event.attrs["id"])] = _make_pass_instance(
                event, away_team_id, pitch_dimensions=pitch_dimensions
            )

        if event_name in [
            "miss",
            "post",
            "attempt saved",
            "not past goal line",
            "goal",
            "own goal",
        ]:
            shot_events[int(event.attrs["id"])] = _make_shot_event_instance(
                event, away_team_id, pitch_dimensions=pitch_dimensions
            )

        if event_name == "take on":
            dribble_events[int(event.attrs["id"])] = _make_dribble_event_instance(
                event, away_team_id, pitch_dimensions=pitch_dimensions
            )

    result_dict["databallpy_event"] = [None] * len(result_dict["event_id"])
    event_data = pd.DataFrame(result_dict)
    event_data["databallpy_event"] = (
        event_data["opta_event"].map(OPTA_TO_DATABALLPY_MAP).replace([np.nan], [None])
    )
    event_data.loc[
        event_data["opta_event"].isin(["miss", "post", "attempt saved"]), "outcome"
    ] = 0
    event_data.loc[event_data["opta_event"].isin(["goal", "own goal"]), "outcome"] = 1
    event_data["datetime"] = utc_to_local_datetime(event_data["datetime"], country)

    # reassign the outcome of passes that result in a shot that is scored to 'assist'
    pass_events = _update_pass_outcome(event_data, shot_events, pass_events)

    return event_data, {
        "shot_events": shot_events,
        "pass_events": pass_events,
        "dribble_events": dribble_events,
    }


def _make_pass_instance(
    event: bs4.element.Tag, away_team_id: int, pitch_dimensions: list = [106.0, 68.0]
) -> PassEvent:
    """Function to create a pass class based on the qualifiers of the event

    Args:
        event (bs4.element.Tag): pass event from the f24.xml
        away_team_id (int): id of the away team
        pitch_dimensions (list, optional): size of the pitch in x and y direction.
            Defaults to [106.0, 68.0].

    Returns:
        PassEvent: Returns a PassEvent instance
    """
    pass_id = int(event.attrs["id"])

    outcome = "successful" if int(event.attrs["outcome"]) else "unsuccessful"
    outcome = "offside" if int(event.attrs["type_id"]) == 2 else outcome

    qualifiers = event.find_all("Q")
    qualifier_ids = [int(q["qualifier_id"]) for q in qualifiers]

    outcome = (
        "results_in_shot"
        if any([q == ASSIST_TO_SHOT_QUALIFIER for q in qualifier_ids])
        else outcome
    )
    outcome = (
        "fair_play"
        if any([q == FAIR_PLAY_QUALIFIER for q in qualifier_ids])
        else outcome
    )

    # sometimes there are multiple pass type qualifiers added to the event
    # we only pick one, based on the following hiearchy:
    # cross > through_ball > long_ball > first_event_available
    pass_type_list = [
        PASS_TYPE_QUALIFIERS[q] for q in qualifier_ids if q in PASS_TYPE_QUALIFIERS
    ]
    pass_type = pass_type_list[0] if len(pass_type_list) > 0 else "not_specified"
    pass_type_hiearchy = ["cross", "through_ball", "long_ball"]
    for pass_type_option in pass_type_hiearchy:
        if pass_type_option in pass_type_list:
            pass_type = pass_type_option
            break

    set_piece_list = [
        SET_PIECE_QUALIFIERS[q] for q in qualifier_ids if q in SET_PIECE_QUALIFIERS
    ]
    set_piece = set_piece_list[0] if len(set_piece_list) > 0 else "no_set_piece"

    x_start, y_start = _rescale_opta_dimensions(
        float(event.attrs["x"]),
        float(event.attrs["y"]),
        pitch_dimensions=pitch_dimensions,
    )

    if event.find("Q", attrs={"qualifier_id": str(X_END_QUALIFIER)}) and event.find(
        "Q", attrs={"qualifier_id": str(Y_END_QUALIFIER)}
    ):
        x_end, y_end = _rescale_opta_dimensions(
            float(
                event.find("Q", attrs={"qualifier_id": str(X_END_QUALIFIER)})["value"]
            ),
            float(
                event.find("Q", attrs={"qualifier_id": str(Y_END_QUALIFIER)})["value"]
            ),
            pitch_dimensions=pitch_dimensions,
        )
    else:
        x_end, y_end = np.nan, np.nan

    if int(event.attrs["team_id"]) == away_team_id:
        x_start *= -1
        y_start *= -1
        x_end *= -1
        y_end *= -1

    return PassEvent(
        event_id=pass_id,
        period_id=int(event.attrs["period_id"]),
        minutes=int(event.attrs["min"]),
        seconds=int(event.attrs["sec"]),
        datetime=pd.to_datetime(event.attrs["timestamp"], utc=True),
        start_x=x_start,
        start_y=y_start,
        outcome=outcome,
        team_id=int(event.attrs["team_id"]),
        player_id=int(event.attrs["player_id"]),
        end_x=x_end,
        end_y=y_end,
        pass_type=pass_type,
        set_piece=set_piece,
    )


def _make_shot_event_instance(
    event: bs4.element.Tag, away_team_id: int, pitch_dimensions: list = [106.0, 68.0]
):
    """Function to create a shot class based on the qualifiers of the event

    Args:
        event (bs4.element.Tag): shot event from the f24.xml
        away_team_id (int): id of the away team
        pitch_dimensions (list, optional): size of the pitch in x and y direction.
        Defaults to [106.0, 68.0].

    Returns:
        dict: Returns a dict with the shot data. The key is the id of the event and
        the value is a ShotEvent instance.
    """
    shot_outcome = SHOT_OUTCOMES[int(event["type_id"])]

    if (
        event.find("Q", {"qualifier_id": str(SHOT_BLOCKED_QUALIFIER)}) is not None
        and shot_outcome != "goal"
    ):
        shot_outcome = "blocked"
    elif event.find("Q", {"qualifier_id": str(OWN_GOAL_QUALIFIER)}) is not None:
        shot_outcome = "own_goal"

    if shot_outcome in ["goal", "miss_on_target", "own_goal"]:
        y_target = (
            7.32
            / 100
            * float(event.find("Q", {"qualifier_id": str(Y_TARGET_QUALIFIER)})["value"])
            - 3.66
        )
        z_target = (
            2.44
            / 100
            * float(event.find("Q", {"qualifier_id": str(Z_TARGET_QUALIFIER)})["value"])
        )
    else:
        y_target, z_target = np.nan, np.nan

    qualifiers = event.find_all("Q")

    type_of_play_list = [
        SHOT_ORIGINS_QUALIFIERS[int(q["qualifier_id"])]
        for q in qualifiers
        if int(q["qualifier_id"]) in SHOT_ORIGINS_QUALIFIERS
    ]
    type_of_play = (
        type_of_play_list[0] if len(type_of_play_list) > 0 else "regular_play"
    )

    body_part_list = [
        BODY_PART_QUALIFIERS[int(q["qualifier_id"])]
        for q in qualifiers
        if int(q["qualifier_id"]) in BODY_PART_QUALIFIERS
    ]
    body_part = body_part_list[0] if len(body_part_list) > 0 else None

    x_start, y_start = _rescale_opta_dimensions(
        float(event.attrs["x"]),
        float(event.attrs["y"]),
        pitch_dimensions=pitch_dimensions,
    )

    first_touch = (
        event.find("Q", {"qualifier_id": str(FIRST_TOUCH_QUALIFIER)}) is not None
    )

    created_oppertunity_list = [
        int(q["qualifier_id"])
        for q in qualifiers
        if int(q["qualifier_id"]) in CREATED_OPPERTUNITY_QUALIFIERS
    ]
    created_oppertunity = (
        CREATED_OPPERTUNITY_QUALIFIERS[created_oppertunity_list[0]]
        if len(created_oppertunity_list) > 0
        else "regular_play"
    )

    if event.find("Q", {"qualifier_id": str(RELATED_EVENT_QUALIFIER)}) is not None:
        related_event_id = int(
            event.find("Q", {"qualifier_id": str(RELATED_EVENT_QUALIFIER)})["value"]
        )
    else:
        related_event_id = MISSING_INT

    if int(event.attrs["team_id"]) == away_team_id:
        x_start *= -1
        y_start *= -1

    return ShotEvent(
        player_id=int(event.attrs["player_id"]),
        event_id=int(event.attrs["id"]),
        period_id=int(event.attrs["period_id"]),
        minutes=int(event.attrs["min"]),
        seconds=int(event.attrs["sec"]),
        datetime=pd.to_datetime(event.attrs["timestamp"], utc=True),
        start_x=x_start,
        start_y=y_start,
        team_id=int(event.attrs["team_id"]),
        shot_outcome=shot_outcome,
        y_target=y_target,
        z_target=z_target,
        body_part=body_part,
        type_of_play=type_of_play,
        first_touch=first_touch,
        created_oppertunity=created_oppertunity,
        related_event_id=related_event_id,
    )


def _make_dribble_event_instance(
    event: bs4.element.Tag, away_team_id: int, pitch_dimensions: list = [106.0, 68.0]
) -> DribbleEvent:
    """Function to create a dribble class based on the qualifiers of the event

    Args:
        event (bs4.element.Tag): dribble event from the f24.xml
        away_team_id (int): id of the away team
        pitch_dimensions (list, optional): pitch dimensions in x and y direction.
            Defaults to [106.0, 68.0].

    Returns:
        DribbleEvent: instance of the DribbleEvent class
    """
    qualifiers = event.find_all("Q")

    if event.find("Q", {"qualifier_id": str(OPPOSITE_RELATED_EVENT_ID)}) is not None:
        related_event_id = int(
            event.find("Q", {"qualifier_id": str(OPPOSITE_RELATED_EVENT_ID)})["value"]
        )
    else:
        related_event_id = MISSING_INT

    duel_type_list = [
        DRIBBLE_DUEL_TYPE_QUALIFIERS[int(q["qualifier_id"])]
        for q in qualifiers
        if int(q["qualifier_id"]) in DRIBBLE_DUEL_TYPE_QUALIFIERS
    ]
    duel_type = duel_type_list[0] if len(duel_type_list) > 0 else None

    x_start, y_start = _rescale_opta_dimensions(
        float(event.attrs["x"]),
        float(event.attrs["y"]),
        pitch_dimensions=pitch_dimensions,
    )

    if int(event.attrs["team_id"]) == away_team_id:
        x_start *= -1
        y_start *= -1

    dribble_event = DribbleEvent(
        player_id=int(event.attrs["player_id"]),
        event_id=int(event.attrs["id"]),
        period_id=int(event.attrs["period_id"]),
        minutes=int(event.attrs["min"]),
        seconds=int(event.attrs["sec"]),
        datetime=pd.to_datetime(event.attrs["timestamp"], utc=True),
        start_x=x_start,
        start_y=y_start,
        team_id=int(event.attrs["team_id"]),
        related_event_id=related_event_id,
        duel_type=duel_type,
        outcome=bool(event.attrs["outcome"]),
        has_opponent=True,  # opta take ons are always against an opponent
    )

    return dribble_event


def _rescale_opta_dimensions(
    x: float, y: float, pitch_dimensions: list = [106.0, 68.0]
) -> Tuple[float, float]:
    """Function to rescale the x and y coordinates from the opta data to the pitch
    dimensions. This funciton assumes taht the x and y coordinates range from 0 to 100,
    with (0, 0) being the bottom left corner of the pitch.

    Args:
        x (float): x coordinate of the event
        y (float): y coordinate of the event
        pitch_dimensions (list, optional): dimensions of the pitch in x and y direction.
            Defaults to [106.0, 68.0].

    Returns:
        Tuple[float, float]: rescaled x and y coordinates
    """
    x = x / 100 * pitch_dimensions[0] - (pitch_dimensions[0] / 2)
    y = y / 100 * pitch_dimensions[1] - (pitch_dimensions[1] / 2)
    return x, y


def _update_pass_outcome(
    event_data: pd.DataFrame, shot_events: dict, pass_events: dict
) -> dict:
    """Function to update the outcome of passes that result in a shot that is scored to
    'assist'. This function is needed because the opta data does not provide the
    outcome of passes that result in a goal.

    Args:
        event_data (pd.DataFrame): all event data of the match
        shot_events (dict): list of ShotEvent instances
        pass_events (dict): list of PassEvent instances

    Returns:
        dict: updated list of PassEvent instances
    """
    event_ids = event_data[event_data["opta_event"] == "goal"].event_id.to_list()
    for goal_event_ids in event_ids:
        shot_event = shot_events[goal_event_ids]
        related_event_id = shot_event.related_event_id

        if related_event_id is None or related_event_id == MISSING_INT:
            continue

        related_pass = event_data[
            (event_data["opta_id"] == related_event_id)
            & (event_data["databallpy_event"] == "pass")
        ]

        if related_pass.shape[0] == 0:
            continue

        if related_pass.shape[0] > 1:
            # compute time diff to find the pass that is closest and before the shot
            shot_secs = shot_event.minutes * 60 + shot_event.seconds
            related_pass_secs = related_pass["minutes"] * 60 + related_pass["seconds"]
            time_diff = shot_secs - related_pass_secs
            min_diff_idx = np.argmin(np.where(time_diff >= 0, time_diff, np.inf))
            related_pass = (
                related_pass.iloc[min_diff_idx] if min_diff_idx is not None else None
            )
        else:
            related_pass = related_pass.iloc[0]
        # assign assist to pass
        pass_events[related_pass["event_id"]].outcome = "assist"

    return pass_events
