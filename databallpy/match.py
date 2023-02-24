from dataclasses import dataclass
from typing import List

import pandas as pd

from databallpy.load_data.event_data.metrica_event_data import (
    load_metrica_event_data,
    load_metrica_open_event_data,
)
from databallpy.load_data.event_data.opta import load_opta_event_data
from databallpy.load_data.tracking_data.metrica_tracking_data import (
    load_metrica_open_tracking_data,
    load_metrica_tracking_data,
)
from databallpy.load_data.tracking_data.tracab import load_tracab_tracking_data


@dataclass
class Match:
    """This is the match class. It contains all information of the match and has
    some simple functions to easily obtain information about the match.

    Args:
        tracking_data (pd.DataFrame): Tracking data of the match.
        tracking_data_provider (str): Provider of the tracking data.
        event_data (pd.DataFrame): Event data of the match.
        event_data_provider (str): Provider of the event data.
        pitch_dimensions (Tuple): The size of the pitch in meters in x and y direction.
        periods (pd.DataFrame): The start and end idicators of all periods.
        frame_rate (int): The frequency of the tracking data.
        home_team_id (int): The id of the home team.
        home_team_name (str): The name of the home team.
        home_players (pd.DataFrame): Information about the home players.
        home_score (int): Number of goals scored over the match by the home team.
        home_formation (str): Indication of the formation of the home team.
        away_team_id (int): The id of the away team.
        away_team_name (str): The name of the away team.
        away_players (pd.DataFrame): Information about the away players.
        away_score (int): Number of goals scored over the match by the away team.
        away_formation (str): Indication of the formation of the away team.
        name (str): The home and away team name and score (e.g "nameH 3 - 1 nameA").
        home_players_column_ids (list): All column ids of the tracking data that refer
                                        to information about the home team players.
        away_players_column_ids (list): All column ids of the tracking data that refer
                                        to information about the away team players.

    Funcs
        player_column_id_to_full_name: Simple function to get the full name of a player
                                       from the column id
    """

    tracking_data: pd.DataFrame
    tracking_data_provider: str
    event_data: pd.DataFrame
    event_data_provider: str
    pitch_dimensions: List[float]
    periods: pd.DataFrame
    frame_rate: int
    home_team_id: int
    home_team_name: str
    home_players: pd.DataFrame
    home_score: int
    home_formation: str
    away_team_id: int
    away_team_name: str
    away_players: pd.DataFrame
    away_score: int
    away_formation: str

    @property
    def name(self) -> str:
        home_text = f"{self.home_team_name} {self.home_score}"
        away_text = f"{self.away_score} {self.away_team_name}"
        return f"{home_text} - {away_text}"

    @property
    def home_players_column_ids(self) -> List[str]:
        return [id for id in self.tracking_data.columns if "home" in id]

    @property
    def away_players_column_ids(self) -> List[str]:
        return [id for id in self.tracking_data.columns if "away" in id]

    def player_column_id_to_full_name(self, column_id: str) -> str:
        """Simple function to get the full name of a player from the column id

        Args:
            column_id (str): the column id of a player, for instance "home_1"

        Returns:
            str: full name of the player
        """
        shirt_num = int(column_id.split("_")[1])
        if column_id[:4] == "home":
            return self.home_players.loc[
                self.home_players["shirt_num"] == shirt_num, "full_name"
            ].iloc[0]
        else:
            return self.away_players.loc[
                self.away_players["shirt_num"] == shirt_num, "full_name"
            ].iloc[0]

    def __eq__(self, other):
        if isinstance(other, Match):
            res = [
                self.tracking_data.equals(other.tracking_data),
                self.tracking_data_provider == other.tracking_data_provider,
                self.event_data.equals(other.event_data),
                self.pitch_dimensions == other.pitch_dimensions,
                self.periods.equals(other.periods),
                self.frame_rate == other.frame_rate,
                self.home_team_id == other.home_team_id,
                self.home_team_name == other.home_team_name,
                self.home_formation == other.home_formation,
                self.home_players.equals(other.home_players),
                self.home_score == other.home_score,
                self.away_team_id == other.away_team_id,
                self.away_team_name == other.away_team_name,
                self.away_formation == other.away_formation,
                self.away_players.equals(other.away_players),
                self.away_score == other.away_score,
            ]
            return all(res)
        else:
            return False


def get_match(
    *,
    tracking_data_loc: str,
    tracking_metadata_loc: str,
    event_data_loc: str,
    event_metadata_loc: str,
    tracking_data_provider: str,
    event_data_provider: str,
) -> Match:
    """Function to get all information of a match given its datasources

    Args:
        tracking_data_loc (str): location of the tracking data
        tracking_metadata_loc (str): location of the metadata of the tracking data
        event_data_loc (str): location of the event data
        event_metadata_loc (str): location of the metadata of the event data
        tracking_data_provider (str): provider of the tracking data
        event_data_provider (str): provider of the event data

    Returns:
        Match: All information about the match
    """

    assert tracking_data_provider in [
        "tracab",
        "metrica",
    ], f"We do not support '{tracking_data_provider}' as tracking data provider yet, "
    "please open an issue in our Github repository."
    assert event_data_provider in [
        "opta",
        "metrica",
    ], f"We do not supper '{event_data_provider}' as event data provider yet, "
    "please open an issue in our Github repository."

    # Get event data and event metadata
    if event_data_provider == "opta":
        event_data, event_metadata = load_opta_event_data(
            f7_loc=event_metadata_loc, f24_loc=event_data_loc
        )
    elif event_data_provider == "metrica":
        event_data, event_metadata = load_metrica_event_data(
            event_data_loc=event_data_loc, metadata_loc=event_metadata_loc
        )

    # Get tracking data and tracking metadata
    if tracking_data_provider == "tracab":
        tracking_data, tracking_metadata = load_tracab_tracking_data(
            tracking_data_loc, tracking_metadata_loc
        )
    elif tracking_data_provider == "metrica":
        tracking_data, tracking_metadata = load_metrica_tracking_data(
            tracking_data_loc=tracking_data_loc, metadata_loc=tracking_metadata_loc
        )

    # Check if the event data is scaled the right way
    if not tracking_metadata.pitch_dimensions == event_metadata.pitch_dimensions:
        x_correction = (
            tracking_metadata.pitch_dimensions[0] / event_metadata.pitch_dimensions[0]
        )
        y_correction = (
            tracking_metadata.pitch_dimensions[1] / event_metadata.pitch_dimensions[1]
        )
        event_data["start_x"] *= x_correction
        event_data["start_y"] *= y_correction

    if tracking_data_provider == event_data_provider == "metrica":
        merged_periods = tracking_metadata.periods_frames
        home_players = tracking_metadata.home_players
        away_players = tracking_metadata.away_players
    else:
        # Merge periods
        merged_periods = pd.concat(
            (
                tracking_metadata.periods_frames,
                event_metadata.periods_frames.drop("period", axis=1),
            ),
            axis=1,
        )

        # Merged player info
        home_players = tracking_metadata.home_players.merge(
            event_metadata.home_players[
                ["id", "formation_place", "position", "starter"]
            ],
            on="id",
        )
        away_players = tracking_metadata.away_players.merge(
            event_metadata.away_players[
                ["id", "formation_place", "position", "starter"]
            ],
            on="id",
        )

    match = Match(
        tracking_data=tracking_data,
        tracking_data_provider=tracking_data_provider,
        event_data=event_data,
        event_data_provider=event_data_provider,
        pitch_dimensions=tracking_metadata.pitch_dimensions,
        periods=merged_periods,
        frame_rate=tracking_metadata.frame_rate,
        home_team_id=event_metadata.home_team_id,
        home_formation=event_metadata.home_formation,
        home_score=event_metadata.home_score,
        home_team_name=event_metadata.home_team_name,
        home_players=home_players,
        away_team_id=event_metadata.away_team_id,
        away_formation=event_metadata.away_formation,
        away_score=event_metadata.away_score,
        away_team_name=event_metadata.away_team_name,
        away_players=away_players,
    )

    return match


def get_open_match(provider: str = "metrica") -> Match:
    """Function to load a match object from an open datasource

    Args:
        provider (str, optional): What provider to get the open data from.
        Defaults to "metrica".

    Returns:
        Match: All information about the match
    """
    assert provider in ["metrica"]

    if provider == "metrica":
        tracking_data, metadata = load_metrica_open_tracking_data()
        event_data, _ = load_metrica_open_event_data()

    match = Match(
        tracking_data=tracking_data,
        tracking_data_provider=provider,
        event_data=event_data,
        event_data_provider=provider,
        pitch_dimensions=metadata.pitch_dimensions,
        periods=metadata.periods_frames,
        frame_rate=metadata.frame_rate,
        home_team_id=metadata.home_team_id,
        home_formation=metadata.home_formation,
        home_score=metadata.home_score,
        home_team_name=metadata.home_team_name,
        home_players=metadata.home_players,
        away_team_id=metadata.away_team_id,
        away_formation=metadata.away_formation,
        away_score=metadata.away_score,
        away_team_name=metadata.away_team_name,
        away_players=metadata.away_players,
    )
    return match
