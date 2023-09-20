import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from databallpy.load_data.event_data.dribble_event import DribbleEvent
from databallpy.load_data.event_data.pass_event import PassEvent
from databallpy.load_data.event_data.shot_event import ShotEvent
from databallpy.load_data.metadata import Metadata
from databallpy.match import Match
from databallpy.utils.align_player_ids import align_player_ids
from databallpy.utils.utils import MISSING_INT
from databallpy.warnings import DataBallPyWarning

SCISPORTS_SET_PIECES = [
    "Kick Off",
    "Set Piece",
    "Corner",
    "Goal Kick",
    "Throw In",
    "Free Kick",
    "Penalty",
    "Open Play",
    "Crossed Free Kick",
    "Crossed Throw In",
]

SCISPORTS_LOCATIONS = [
    "Final 3rd",
    "Between Lines",
    "Own Half",
    "Att. Half",
    "Behind last Line",
    "Score Box",
    "Switch",
]

SCISPORTS_PASS_EVENTS = [
    "Pass",
    "Cross",
    "GK Pass",
    "Key Pass",
    "Pre-Key Pass",
    "GK Throw",
    "Assist",
]

SCISPORTS_SHOT_EVENTS = [
    "Shot",
    "Wide",
    "on Target",
    "Goal",
    "Own Goal",
]

SCISPORTS_DUEL_EVENTS = [
    "Duel",
    "Air Duel",
    "Duel Defensive",
    "Take On",
    "Take On Faced",
]

SCISPORTS_DEFENSIVE_EVENTS = [
    "Tackle",
    "Interception",
    "Clearance",
    "Foul",
    "Ball Recovery",
    "GK Save",
    "GK Punch",
    "GK Claim",
    "GK Pick Up",
    "Shot Blocked",
]

SCISPORTS_OUTCOMES = [
    "Possession Loss",
    "Pass (Successful)",
]

SCISPORTS_OTHER = [
    "Defensive",
    "Yellow Card",
    "Red Card",
    "2nd Yellow Card",
    "Big Chance",
    "Substitute",
    "2nd Ball",
    "Penetration",
    "Physical",
]

SCISPORTS_PHYSICAL_EVENTS = [
    "Deep Run",
    "Box To Box Run",
    "Flank To Centre Run",
    "Centre To Flank Run",
]
SCISPORTS_PHYSICAL_RUN_TYPES = [
    "Run",  # 15-20 km/h
    "High Run",  # 20-25 km/h
    "Sprint",  # 25-30 km/h
    "High Speed Sprint",  # 30+ km/h
]

SCISPORTS_EVENTS = (
    SCISPORTS_PASS_EVENTS
    + SCISPORTS_SHOT_EVENTS
    + SCISPORTS_DUEL_EVENTS
    + SCISPORTS_DEFENSIVE_EVENTS
    + SCISPORTS_OTHER
    + SCISPORTS_OUTCOMES
    + SCISPORTS_LOCATIONS
    + SCISPORTS_SET_PIECES
    + SCISPORTS_PHYSICAL_EVENTS
    + SCISPORTS_PHYSICAL_RUN_TYPES
)

SCISPORTS_TO_DATABALLPY_MAP = {
    "Pass": "pass",
    "Cross": "pass",
    "GK Pass": "pass",
    "Key Pass": "pass",
    "Pre-Key Pass": "pass",
    "GK Throw": "pass",
    "Assist": "pass",
    "Shot": "shot",
    "Shot Wide": "shot",
    "Shot on Target": "shot",
    "Shot Goal": "shot",
    "Own Goal": "shot",
    "Take On": "dribble",
}


def load_scisports_event_data(events_xml: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Function to load the SciSports XML file. The SciSports XML file contains
    all events of a match, and is used to create the event data DataFrame. On top of
    that, the SciSports XML file is used to create the event data for the databallpy
    events, possessions and match periods.

    Args:
        events_xml (str): Path to the SciSports XML file

    Raises:
        TypeError: If events_xml is not a string

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the event data DataFrame,
           and a DataFrame with the possessions per team.
    """

    if not isinstance(events_xml, str):
        raise TypeError("events_xml must be a string, not {}".format(type(events_xml)))

    event_data, possessions = _load_event_data(events_xml)
    event_data["databallpy_event"] = event_data["scisports_event"].map(
        SCISPORTS_TO_DATABALLPY_MAP
    )
    event_data["databallpy_event"] = event_data["databallpy_event"].replace(
        {np.nan: None}
    )

    return event_data, possessions


def _load_event_data(events_xml: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Function to load the SciSports XML file. The SciSports XML file contains
    all events of a match, and is used to create the event data DataFrame. The
    event data DataFrame contains all events of a match, with the following columns:
        - seconds: The time of the event in seconds
        - team: The team of the player that performed the event
        - jersey_number: The jersey number of the player that performed the event
        - player_name: The name of the player that performed the event
        - event_name: The name of event, e.g. "Pass", "Shot", "Duel"
        - event_location: The location of the event, e.g. "Final 3rd", "Own Half"
        - set_piece: The set piece of the event, e.g. "Corner", "Open Play"
        - outcome: The outcome of the event, 1 if successful, 0 if not
        - run_type: The run type of the event, e.g. "Run", "Sprint", "High Speed Sprint"
        - period: The period of the event, 1 if first half, 2 if second half
        - event_id: The event id of the event, this is a unique identifier of the event


    Args:
        events_xml (str): Path to the SciSports XML file

    Returns:
        Tuple [pd.DataFrame, pd.DataFrame]: A tuple containing the event data DataFrame,
            and a DataFrame with the possessions per team.
    """

    possessions, individual_events, match_periods = _get_all_events(
        events_xml=events_xml
    )

    # group the individual events by team, player, and time,
    # to get a single row per event
    event_identifiers = {}
    grouped_individual_events = (
        individual_events.groupby(["seconds", "team", "player_name", "jersey_number"])
        .agg(
            {
                "event_name": lambda x: list(set(x.dropna())),
                "event_type": lambda x: list(set(x.dropna())),
                "identifier": lambda x: list(set(x.dropna())),
                "run_type": lambda x: list(set(x.dropna())),
            }
        )
        .reset_index()
    )
    grouped_individual_events["event_id"] = grouped_individual_events.index
    for event in grouped_individual_events.itertuples():
        identifiers = np.concatenate(
            [event.event_name, event.event_type, event.identifier, event.run_type]
        )
        identifiers = np.unique(identifiers)
        event_identifiers[event.event_id] = identifiers

    individual_events = grouped_individual_events.drop(
        columns=["event_name", "event_type", "identifier", "run_type"]
    )

    # check if all events are present
    event_info_dict = {
        "scisports_event": [],
        "event_location": [],
        "set_piece": [],
        "event_id": [],
        "outcome": [],
        "run_type": [],
    }
    all_events = SCISPORTS_EVENTS
    for i, event_list in event_identifiers.items():
        for event_name in event_list:
            # check if all events are known
            if event_name not in all_events:
                warnings.warn(
                    f"{event_name} in the scisports events is an unknown event.",
                    DataBallPyWarning,
                )
                continue

        run_type = None
        if "Physical" in event_list:
            set_piece, location, outcome = None, None, None
            event = [x for x in event_list if x in SCISPORTS_PHYSICAL_EVENTS][0]
            run_type = [x for x in event_list if x in SCISPORTS_PHYSICAL_RUN_TYPES][0]

        else:
            set_piece = _find_set_piece(event_list)
            location = _find_location(event_list)
            event, outcome = _find_event(event_list)

            # set event type for set pieces
            if set_piece != "Open Play" and event == "Unknown":
                if "Penalty" in set_piece:
                    event = "Shot"
                elif set_piece in ["Crossed Free Kick", "Crossed Throw In", "Corner"]:
                    event = "Cross"
                    mapping = {
                        "Crossed Free Kick": "Free Kick",
                        "Crossed Throw In": "Throw In",
                        "Corner": "Corner",
                    }
                    set_piece = mapping[set_piece]
                elif "Free Kick" in set_piece:
                    if "Own Half" in location:
                        event = "Pass"
                else:
                    event = "Pass"

        event_info_dict["event_location"].append(location)
        event_info_dict["set_piece"].append(set_piece)
        event_info_dict["event_id"].append(i)
        event_info_dict["outcome"].append(outcome)
        event_info_dict["run_type"].append(run_type)
        event_info_dict["scisports_event"].append(event)

    events_info = pd.DataFrame(event_info_dict)
    event_data = pd.merge(individual_events, events_info, on="event_id", how="left")
    first_half_end = match_periods[1]
    event_data["period_id"] = np.where(event_data.seconds <= first_half_end, 1, 2)

    return event_data, possessions


def _get_all_events(events_xml: str) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """Function to get all events from the SciSports XML file. Note that some of the
    events are double. Doubled events are characterized by a similar time, team, and
    jersey number. The event name, event type, identifier, and run type are esentially
    all identifiers of this one event.

    Args:
        events_xml (str): Path to the SciSports XML file

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, list]: A tuple containing the possessions,
            individual events, and match periods.
    """

    with open(events_xml, "r") as file:
        contents = file.read()

    soup = BeautifulSoup(contents, "xml")

    events = soup.find_all("instance")
    team_events = {
        "seconds": [],
        "team": [],
        "jersey_number": [],
        "player_name": [],
        "event_type": [],
        "event_name": [],
        "identifier": [],
        "run_type": [],
    }
    possessions = {
        "team": [],
        "start_seconds": [],
        "end_seconds": [],
        "possession_type": [],
    }
    match_periods = []

    # loop over events
    for event in events:
        start = float(event.find("start").text)
        end = float(event.find("end").text)
        time = (start + end) / 2.0

        code = event.find("code").text
        if code == "SciSports timestamps":
            match_periods.append(time)
            continue

        # player events is in team events, otherwise all events are double
        elif not code.split(" ")[0].isdigit():
            team, event_info = code.split(" - ")

            event_type, event_name = (
                event_info.split(":") if ":" in event_info else (None, event_info)
            )
            event_type = event_type.strip() if event_type else None
            event_name = event_name.strip()
            event_name, identifier = (
                event_name.split(" | ") if " | " in event_name else (event_name, None)
            )

            if event_type == "Possession":
                possessions["team"].append(team)
                possessions["start_seconds"].append(start)
                possessions["end_seconds"].append(end)
                possessions["possession_type"].append(event_name)

            # player events
            else:
                jersey_number, player_name, run_type = None, None, None
                for label in event.find_all("label"):
                    if (
                        label.find("group").text == "Player"
                        or label.find("group").text == "player"
                    ):
                        jersey_number, player_name = label.find("text").text.split(
                            " - "
                        )
                        jersey_number = int(jersey_number)
                    elif label.find("group").text == "run_type":
                        run_type = label.find("text").text.split(":")[1].strip()
                team_events["seconds"].append(time)
                team_events["team"].append(team)
                team_events["jersey_number"].append(jersey_number)
                team_events["player_name"].append(player_name)
                team_events["event_type"].append(event_type)
                team_events["event_name"].append(event_name)
                team_events["identifier"].append(identifier)
                team_events["run_type"].append(run_type)

    individual_events = (
        pd.DataFrame(team_events).sort_values(by="seconds").reset_index(drop=True)
    )
    possessions = (
        pd.DataFrame(possessions).sort_values(by="start_seconds").reset_index(drop=True)
    )
    return possessions, individual_events, match_periods


def _find_set_piece(identifiers: list) -> str:
    """Function to find the set piece of an event, if any.

    Args:
        identifiers (list): List of all identifiers of an event

    Returns:
        str: The set piece of the event, if none is found, "Open Play" is returned.
    """
    set_pieces = [x for x in identifiers if x in SCISPORTS_SET_PIECES]
    if len(set_pieces) == 0:
        set_piece = "Open Play"
    elif len(set_pieces) == 1:
        set_piece = set_pieces[0]
    else:
        set_pieces = [x for x in set_pieces if x != "Set Piece"]
        set_piece = set_pieces[0]
    return set_piece


def _find_location(identifiers: list) -> str:
    """Function to find the location(s) of an event, if any.

    Args:
        identifiers (list): List of all identifiers of an event

    Returns:
        str: The location(s) of the event, if none is found, "Unknown" is returned.
            If multiple are found, they are joined with " - " into one string.
    """
    locations = [x for x in identifiers if x in SCISPORTS_LOCATIONS]
    if len(locations) == 0:
        location = "Unknown"
    elif len(locations) == 1:
        location = locations[0]
    else:
        location = " - ".join(locations)
    return location


def _find_event(identifiers: list) -> Tuple[str, int]:
    """Function to find the event name of an event, and the outcome of the event
    if applicable.

    Args:
        identifiers (list): List of all identifiers of an event

    Returns:
        Tuple [str, int]: The event name and outcome of the event, if none is found,
            "Unknown" and None are returned.
    """
    pass_events = [x for x in identifiers if x in SCISPORTS_PASS_EVENTS]
    shot_events = [x for x in identifiers if x in SCISPORTS_SHOT_EVENTS]
    duel_events = [x for x in identifiers if x in SCISPORTS_DUEL_EVENTS]
    defensive_events = [x for x in identifiers if x in SCISPORTS_DEFENSIVE_EVENTS]
    outcome_events = [x for x in identifiers if x in SCISPORTS_OUTCOMES]

    event = "Unknown"
    outcome = None

    # outcome
    if len(outcome_events) > 0:
        if "Pass (Successful)" in outcome_events:
            outcome = 1
        elif "Possession Loss" in outcome_events:
            outcome = 0

    # shot events
    if len(shot_events) > 0:
        if "Goal" in shot_events:
            event = "Goal"
            outcome = 1
        elif "Own Goal" in shot_events:
            event = "Own Goal"
            outcome = 1
        else:
            shot_events = [x for x in shot_events if x != "Shot"]
            event = f"Shot {shot_events[0]}"
            outcome = 0

    # pass events
    elif len(pass_events) > 0:
        # type of pass
        if "Assist" in pass_events:
            event = "Assist"
            outcome = 1
        elif len(pass_events) == 1:
            event = pass_events[0]
        else:
            pass_events = [x for x in pass_events if x != "Pass"]
            event = pass_events[0] if len(pass_events) > 0 else "Pass"

    # duel events
    elif len(duel_events) > 0:
        if len(duel_events) == 1:
            event = duel_events[0]
        else:
            duel_events = [
                x for x in duel_events if x != "Duel" and x != "Duel Defensive"
            ]
            event = duel_events[0] if len(duel_events) > 0 else "Duel"

    # defensive events
    elif len(defensive_events) > 0:
        if len(defensive_events) == 1:
            event = defensive_events[0]
        else:
            defensive_events = [x for x in defensive_events if x != "GK Save"]
            event = defensive_events[0]

    # handle other events, note that set piece events are handled in _load_event_data
    else:
        if "Penetration" in identifiers:
            event = "Penetration"
        elif "Possession Loss" in identifiers:
            event = "Possession Loss"
            outcome = 1
        elif "2nd Ball" in identifiers:
            event = "2nd Ball"
        elif "Substitute" in identifiers:
            event = "Substitute"

    return event, outcome


def _get_databallpy_events_scisports(
    tracking_data: pd.DataFrame,
    tracking_metadata: Metadata,
    event_data: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """Function to get the databallpy events from the scisports event data using
    the tracking data

    Args:
        tracking_data (pd.DataFrame): The tracking data of the match
        tracking_metadata (Metadata): The metadata of the tracking data
        event_data (pd.DataFrame): The scisports event data of the match
        verbose (bool, optional): Whether to print info about the progress.
            Defaults to True.

    Returns:
        dict: The databallpy events
    """
    temp_match = Match(
        tracking_data=tracking_data,
        tracking_data_provider="unimportant",
        event_data=event_data,
        event_data_provider="scisports",
        pitch_dimensions=tracking_metadata.pitch_dimensions,
        periods=tracking_metadata.periods_frames,
        frame_rate=tracking_metadata.frame_rate,
        home_team_id=tracking_metadata.home_team_id,
        home_formation=tracking_metadata.home_formation,
        home_score=tracking_metadata.home_score,
        home_team_name=tracking_metadata.home_team_name,
        home_players=tracking_metadata.home_players,
        away_team_id=tracking_metadata.away_team_id,
        away_formation=tracking_metadata.away_formation,
        away_score=tracking_metadata.away_score,
        away_team_name=tracking_metadata.away_team_name,
        away_players=tracking_metadata.away_players,
        country=tracking_metadata.country,
        allow_synchronise_tracking_and_event_data=True,
    )

    if verbose:
        print("Syncing tracking and event data to obtain databallpy events")
    temp_match.synchronise_tracking_and_event_data(verbose=False)
    tracking_data = temp_match.tracking_data
    event_data = temp_match.event_data

    # update the event data with the synced event data
    mask = ~pd.isnull(event_data["databallpy_event"])
    for db_event in event_data.loc[mask].itertuples():
        td_frame = tracking_data.loc[
            tracking_data["event_id"] == db_event.event_id
        ].iloc[0]
        event_data.loc[db_event.Index, "start_x"] = td_frame["ball_x"]
        event_data.loc[db_event.Index, "start_y"] = td_frame["ball_y"]

    databallpy_events = {}

    databallpy_events["shot_events"] = _get_shot_instances(event_data=event_data)
    databallpy_events["dribble_events"] = _get_dribble_instances(event_data=event_data)
    databallpy_events["pass_events"] = _get_pass_instances(event_data=event_data)

    return databallpy_events


def _get_shot_instances(event_data: pd.DataFrame) -> dict:
    """Function to get the ShotEvent instances from the scisports event data

    Args:
        event_data (pd.DataFrame): The scisports event data of the match

    Returns:
        dict: The ShotEvent instances
    """
    shots_map = {
        "Shot on Target": "miss_on_target",
        "Shot Goal": "goal",
        "Shot Wide": "miss_off_target",
        "Own Goal": "own_goal",
        "Shot": "not_specified",
    }
    shots_mask = event_data["databallpy_event"] == "shot"
    shot_events = {}
    for shot in event_data.loc[shots_mask].itertuples():
        shot_events[shot.event_id] = ShotEvent(
            event_id=shot.event_id,
            period_id=shot.period_id,
            minutes=int(shot.minutes),
            seconds=int(shot.seconds),
            datetime=shot.datetime,
            start_x=shot.start_x,
            start_y=shot.start_y,
            y_target=np.nan,
            z_target=np.nan,
            team_id=shot.team_id,
            player_id=shot.player_id,
            shot_outcome=shots_map[shot.scisports_event],
            type_of_play=None,
            body_part=None,
            created_oppertunity=None,
        )
    return shot_events


def _get_dribble_instances(event_data: pd.DataFrame) -> dict:
    """Function to get the DribbleEvent instances from the scisports event data

    Args:
        event_data (pd.DataFrame): The scisports event data of the match

    Returns:
        dict: The DribbleEvent instances
    """
    dribbles_mask = event_data["databallpy_event"] == "dribble"
    dribble_events = {}
    for dribble in event_data.loc[dribbles_mask].itertuples():
        dribble_events[dribble.Index] = DribbleEvent(
            event_id=dribble.event_id,
            period_id=dribble.period_id,
            minutes=int(dribble.minutes),
            seconds=int(dribble.seconds),
            datetime=dribble.datetime,
            start_x=dribble.start_x,
            start_y=dribble.start_y,
            team_id=dribble.team_id,
            player_id=dribble.player_id,
            related_event_id=MISSING_INT,
            duel_type=None,
            outcome=None,
            has_opponent=True,
        )
    return dribble_events


def _get_pass_instances(event_data: pd.DataFrame) -> dict:
    """Function to get the PassEvent instances from the scisports event data

    Args:
        event_data (pd.DataFrame): The scisports event data of the match

    Returns:
        dict: The PassEvent instances
    """
    set_piece_map = {
        "Kick Off": "kick_off",
        "Set Piece": "unspecified_set_piece",
        "Corner": "corner_kick",
        "Goal Kick": "goal_kick",
        "Throw In": "throw_in",
        "Free Kick": "free_kick",
        "Penalty": "penalty",
        "Open Play": "no_set_piece",
        "Crossed Free Kick": "free_kick",
        "Crossed Throw In": "throw_in",
    }
    passes_map = {
        "Pass": "not_specified",
        "Cross": "cross",
        "GK Pass": "not_specified",
        "Key Pass": "not_specified",
        "Pre-Key Pass": "not_specified",
        "GK Throw": "not_specified",
        "Assist": "assist",
    }
    passes_mask = event_data["databallpy_event"] == "pass"
    pass_events = {}
    for pass_ in event_data.loc[passes_mask].itertuples():
        outcome = (
            ["successful", "unsuccessful"][pass_.outcome == 0]
            if not pd.isnull(pass_.outcome)
            else None
        )

        pass_events[pass_.Index] = PassEvent(
            event_id=pass_.event_id,
            period_id=pass_.period_id,
            minutes=int(pass_.minutes),
            seconds=int(pass_.seconds),
            datetime=pass_.datetime,
            start_x=pass_.start_x,
            start_y=pass_.start_y,
            team_id=pass_.team_id,
            outcome=outcome,
            player_id=pass_.player_id,
            end_x=np.nan,
            end_y=np.nan,
            length=np.nan,
            angle=np.nan,
            pass_type=passes_map[pass_.scisports_event],
            set_piece=set_piece_map[pass_.set_piece],
        )
    return pass_events


def _update_scisports_event_data_with_metadata(
    scisports_event_data: pd.DataFrame, metadata: Metadata
) -> pd.DataFrame:
    """Function to update the scisports event data with metadata. Among other things,
    this function adds the player id's to the scisports event data.

    Args:
        scisports_event_data (pd.DataFrame): scisports event data
        metadata (Metadata): metadata of the match

    Returns:
        pd.DataFrame: updataed scisports event data
    """

    event_data = pd.DataFrame(
        index=scisports_event_data.index,
        columns=[
            "event_id",
            "databallpy_event",
            "period_id",
            "minutes",
            "seconds",
            "player_id",
            "team_id",
            "outcome",
            "start_x",
            "start_y",
            "datetime",
            "scisports_event",
            "event_location",
            "run_type",
            "player_name",
        ],
    )

    event_data["event_id"] = scisports_event_data["event_id"]
    event_data["databallpy_event"] = scisports_event_data["databallpy_event"]
    event_data["period_id"] = scisports_event_data["period_id"]
    event_data["minutes"] = (scisports_event_data["seconds"] // 60).astype("int64")
    event_data["seconds"] = scisports_event_data["seconds"] % 60
    event_data["outcome"] = scisports_event_data["outcome"]
    event_data["start_x"] = np.nan
    event_data["start_y"] = np.nan

    # datetime
    time_in_seconds = (
        scisports_event_data["seconds"] - scisports_event_data["seconds"].iloc[0]
    )
    if "start_datetime_ed" in metadata.periods_frames.columns:
        start_datetime = metadata.periods_frames["start_datetime_ed"].iloc[0]
    elif "start_datetime_td" in metadata.periods_frames.columns:
        start_datetime = metadata.periods_frames["start_datetime_td"].iloc[0]

    if "start_datetime" in vars():
        event_data["datetime"] = start_datetime + pd.to_timedelta(
            time_in_seconds, unit="s"
        )
    else:
        event_data["datetime"] = pd.to_datetime("NaT")

    event_data["scisports_event"] = scisports_event_data["scisports_event"]
    event_data["event_location"] = scisports_event_data["event_location"]
    event_data["run_type"] = scisports_event_data["run_type"]
    event_data["team_id"] = np.nan
    event_data["player_id"] = np.nan
    event_data["player_name"] = scisports_event_data["player_name"]
    event_data["set_piece"] = scisports_event_data["set_piece"]

    def get_player_id(row):
        team_name = row["team"]
        jersey_number = row["jersey_number"]
        if team_name == metadata.home_team_name:
            current_player_id = metadata.home_players[
                metadata.home_players["shirt_num"] == jersey_number
            ]["id"].iloc[0]
        else:
            current_player_id = metadata.away_players[
                metadata.away_players["shirt_num"] == jersey_number
            ]["id"].iloc[0]
        return current_player_id

    def get_player_name(row):
        team_name = row["team"]
        jersey_number = row["jersey_number"]
        if team_name == metadata.home_team_name:
            current_player_name = metadata.home_players[
                metadata.home_players["shirt_num"] == jersey_number
            ]["full_name"].iloc[0]
        else:
            current_player_name = metadata.away_players[
                metadata.away_players["shirt_num"] == jersey_number
            ]["full_name"].iloc[0]
        return current_player_name

    team_id_mask = scisports_event_data["team"] == metadata.home_team_name
    event_data.loc[team_id_mask, "team_id"] = metadata.home_team_id
    team_id_mask = scisports_event_data["team"] == metadata.away_team_name
    event_data.loc[team_id_mask, "team_id"] = metadata.away_team_id

    player_id_mask = scisports_event_data["jersey_number"].isin(
        metadata.home_players["shirt_num"]
    )
    event_data.loc[player_id_mask, "player_id"] = scisports_event_data[
        player_id_mask
    ].apply(get_player_id, axis=1)
    event_data.loc[player_id_mask, "player_name"] = scisports_event_data[
        player_id_mask
    ].apply(get_player_name, axis=1)

    player_id_mask = scisports_event_data["jersey_number"].isin(
        metadata.away_players["shirt_num"]
    )
    event_data.loc[player_id_mask, "player_id"] = scisports_event_data[
        player_id_mask
    ].apply(get_player_id, axis=1)
    event_data.loc[player_id_mask, "player_name"] = scisports_event_data[
        player_id_mask
    ].apply(get_player_name, axis=1)

    return event_data


def _add_team_possessions_to_tracking_data(
    tracking_data: pd.DataFrame,
    possessions: pd.DataFrame,
    frame_rate: int,
    home_team_name: str,
) -> pd.DataFrame:
    """Function to add the team possessions to the tracking data based on the
    possession information from the scisports event data

    Args:
        tracking_data (pd.DataFrame): The tracking data
        possessions (pd.DataFrame): The possessions of the match
        frame_rate (int): The frame rate of the tracking data
        home_team_name (str): The name of the home team

    Returns:
        pd.DataFrame: The tracking data with the team possessions
    """

    if (tracking_data["ball_status"].unique() == "dead").all():
        match_start_idx = tracking_data.index[0]
    else:
        match_start_idx = tracking_data[tracking_data["ball_status"] == "alive"].index[
            0
        ]

    tracking_data["td_seconds"] = np.nan
    td_seconds = np.array(range(len(tracking_data.loc[match_start_idx:]))) / frame_rate
    tracking_data.loc[match_start_idx:, "td_seconds"] = td_seconds
    team_shift_idxs = (
        np.where(possessions.team[1:].values != possessions.team[:-1].values)[0] + 1
    )
    next_possession_start_s = 0
    for idx in team_shift_idxs:
        possession_end_s = (
            possessions.iloc[idx]["start_seconds"]
            + possessions.iloc[idx - 1]["end_seconds"]
        ) / 2
        mask = (tracking_data["td_seconds"] >= next_possession_start_s) & (
            tracking_data["td_seconds"] < possession_end_s
        )
        side = "home" if home_team_name == possessions.iloc[idx - 1]["team"] else "away"
        tracking_data.loc[mask, "ball_possession"] = side
        next_possession_start_s = possession_end_s

    # last possession
    mask = tracking_data["td_seconds"] >= next_possession_start_s
    side = "home" if home_team_name == possessions.iloc[-1]["team"] else "away"
    tracking_data.loc[mask, "ball_possession"] = side

    tracking_data.drop(columns=["td_seconds"], inplace=True)
    return tracking_data


def _handle_scisports_data(
    scisports_ed_loc: str,
    tracking_data: pd.DataFrame,
    event_metadata: Metadata,
    tracking_metadata: Metadata,
    databallpy_events: dict,
    verbose: bool,
) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Funciton to handle scisports event data. It updates the event data based on
    provided metadata. Adds possessions to the tracking data if needed, and syncs
    player id's between tracking and event data if they are different.

    Args:
        scisports_ed_loc (str): location of the scisports event data (*.xml)
        tracking_data (pd.DataFrame): tracking data of the match
        event_metadata (Metadata): metadata based on the (non-scisports) event data
        tracking_metadata (Metadata): metadata based on the tracking data
        databallpy_events (dict): dict with extra info about the events
        verbose (bool): whether to print info about the progress.

    Returns:
        Tuple[pd.DataFrame, dict, pd.DataFrame]: updated scisports event data,
            updated databallpy events, updated tracking data
    """
    scisports_event_data, possessions = load_scisports_event_data(
        events_xml=scisports_ed_loc
    )

    if event_metadata is not None:
        scisports_event_data = _update_scisports_event_data_with_metadata(
            scisports_event_data, event_metadata
        )
    elif tracking_metadata is not None:
        scisports_event_data = _update_scisports_event_data_with_metadata(
            scisports_event_data, tracking_metadata
        )

    if tracking_data is not None:
        if event_metadata is not None:
            if (
                not set(event_metadata.home_players["id"])
                == set(tracking_metadata.home_players["id"])
            ) or (
                not set(event_metadata.away_players["id"])
                == set(tracking_metadata.away_players["id"])
            ):
                event_metadata = align_player_ids(event_metadata, tracking_metadata)
                full_name_id_map = dict(
                    zip(
                        event_metadata.home_players["full_name"],
                        event_metadata.home_players["id"],
                    )
                )
                full_name_id_map.update(
                    dict(
                        zip(
                            event_metadata.away_players["full_name"],
                            event_metadata.away_players["id"],
                        )
                    )
                )

                scisports_event_data["player_id"] = (
                    scisports_event_data["player_name"]
                    .map(full_name_id_map)
                    .fillna(MISSING_INT)
                    .astype("int64")
                )

            # also check team_id's
            if not event_metadata.home_team_id == tracking_metadata.home_team_id:
                event_home_team_id = event_metadata.home_team_id
                tracking_home_team_id = tracking_metadata.home_team_id
                scisports_event_data["team_id"] = scisports_event_data[
                    "team_id"
                ].replace({event_home_team_id: tracking_home_team_id})
            if not event_metadata.away_team_id == tracking_metadata.away_team_id:
                event_away_team_id = event_metadata.away_team_id
                tracking_away_team_id = tracking_metadata.away_team_id
                scisports_event_data["team_id"] = scisports_event_data[
                    "team_id"
                ].replace({event_away_team_id: tracking_away_team_id})

            if str(scisports_event_data["team_id"].dtype) == "float64":
                scisports_event_data["team_id"] = scisports_event_data[
                    "team_id"
                ].astype("int64")
            if str(scisports_event_data["player_id"].dtype) == "float64":
                scisports_event_data["player_id"] = scisports_event_data[
                    "player_id"
                ].astype("int64")

        if tracking_data["ball_possession"].isnull().all():
            tracking_data = _add_team_possessions_to_tracking_data(
                tracking_data,
                possessions,
                frame_rate=tracking_metadata.frame_rate,
                home_team_name=tracking_metadata.home_team_name,
            )

    if databallpy_events is None or len(databallpy_events) == 0:
        databallpy_events = _get_databallpy_events_scisports(
            tracking_data, tracking_metadata, scisports_event_data, verbose
        )

    return scisports_event_data, databallpy_events, tracking_data
