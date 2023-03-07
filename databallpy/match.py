from dataclasses import dataclass
from typing import List
from databallpy import DataBallPyError

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

    def __post_init__(self):
        # tracking_data
        if not isinstance(self.tracking_data, pd.DataFrame):
            raise TypeError(
                f"tracking data should be a pandas df, not a {type(self.tracking_data)}"
            )
        for col in ["timestamp", "ball_x", "ball_y"]:
            if col not in self.tracking_data.columns.to_list():
                raise ValueError(
                    f"No {col} in tracking_data columns, this is manditory!"
                )

        # tracking_data_provider
        if not isinstance(self.tracking_data_provider, str):
            raise TypeError(
                f"tracking data provider should be a string, not a \
                    {type(self.tracking_data_provider)}"
            )

        # event_data
        if not isinstance(self.event_data, pd.DataFrame):
            raise TypeError(
                f"event data should be a pandas df, not a {type(self.event_data)}"
            )
        for col in [
            "event_id",
            "event",
            "period_id",
            "team_id",
            "player_id",
            "start_x",
            "start_y",
            "datetime",
        ]:
            if col not in self.event_data.columns.to_list():
                raise ValueError(f"{col} not in event data columns, this is manditory!")

        # event_data_provider
        if not isinstance(self.event_data_provider, str):
            raise TypeError(
                f"event data provider should be a string, not a \
                    {type(self.event_data_provider)}"
            )

        # pitch_dimensions
        if not isinstance(self.pitch_dimensions, list):
            raise TypeError(
                f"pitch_dimensions ({self.pitch_dimensions}) should be a \
                    list, not a {type(self.pitch_dimensions)}"
            )
        if not len(self.pitch_dimensions) == 2:
            raise ValueError(
                f"pitch_dimensions should contain, two values: a length and a width \
                    of the pitch, current input is {self.pitch_dimensions}"
            )
        if not all([isinstance(x, float) for x in self.pitch_dimensions]):
            raise TypeError(
                f"Both values in pitch dimensions should by floats, current inputs \
                    {[type(x) for x in self.pitch_dimensions]}"
            )

        # periods
        if not isinstance(self.periods, pd.DataFrame):
            raise TypeError(
                f"periods_frames should be a pandas dataframe, not a \
                    {type(self.periods)}"
            )
        if "period" not in self.periods.columns:
            raise ValueError("'period' should be one of the columns in period_frames")
        if any(
            [
                x not in self.periods["period"].value_counts().index
                for x in [1, 2, 3, 4, 5]
            ]
        ) or not all(self.periods["period"].value_counts() == 1):

            res = self.periods["period"]
            raise ValueError(
                f"'period' column in period_frames should contain only the values \
                    [1, 2, 3, 4, 5]. Now it's {res}"
            )

        # frame_rate
        if not pd.isnull(self.frame_rate):
            if not isinstance(self.frame_rate, int):
                raise TypeError(
                    f"frame_rate should be an integer, not a {type(self.frame_rate)}"
                )
            if self.frame_rate < 1:
                raise ValueError(
                    f"frame_rate should be a positive integer, not {self.frame_rate}"
                )

        # team id's
        for team, team_id in zip(
            ["home", "away"], [self.home_team_id, self.away_team_id]
        ):
            if not isinstance(team_id, int) and not isinstance(team_id, str):
                raise TypeError(
                    f"{team} team id should be an integer or string, not a \
                        {type(team_id)}"
                )

        # team names
        for team, name in zip(
            ["home", "away"], [self.home_team_name, self.away_team_name]
        ):
            if not isinstance(name, str):
                raise TypeError(
                    f"{team} team name should be a string, not a {type(name)}"
                )

        # team scores
        for team, score in zip(["home", "away"], [self.home_score, self.away_score]):
            if not pd.isnull(score):
                if not isinstance(score, int):
                    raise TypeError(
                        f"{team} team score should be an integer, not a {type(score)}"
                    )
                if score < 0:
                    raise ValueError(f"{team} team score should positive, not {score}")

        # team formations
        for team, form in zip(
            ["home", "away"], [self.home_formation, self.away_formation]
        ):
            if not isinstance(form, str):
                raise TypeError(
                    f"{team} team formation should be a string, not a {type(form)}"
                )
            if len(form) > 4:
                raise ValueError(
                    f"{team} team formation should be of length 4 or smaller \
                        ('1433'), not {len(form)}"
                )

        # team players
        for team, players in zip(
            ["home", "away"], [self.home_players, self.away_players]
        ):
            if not isinstance(players, pd.DataFrame):
                raise TypeError(
                    f"{team} team players should be a pandas dataframe, not a \
                        {type(players)}"
                )
            for col in ["id", "full_name", "shirt_num"]:
                if col not in players.columns:
                    raise ValueError(
                        f"{team} team players should contain at least the column \
                            ['id', 'full_name', 'shirt_num'], {col} is missing."
                    )
        
        # check for pitch axis
        if not abs(
            self.tracking_data["ball_x"].min() + self.tracking_data["ball_x"].max()
        ) < 5.:
            max_x = self.tracking_data["ball_x"].max()
            min_x = self.tracking_data["ball_x"].min()
            raise DataBallPyError(f"The middle point of the pitch should be (0, 0),\
                                now th min x = {min_x} and the max x = {max_x}")

        if not abs(
            self.tracking_data["ball_y"].min() + self.tracking_data["ball_y"].max()
        ) < 5.:
            max_y = self.tracking_data["ball_y"].max()
            min_y = self.tracking_data["ball_y"].min()
            raise DataBallPyError(f"The middle point of the pitch should be (0, 0),\
                                now th min y = {min_y} and the max y = {max_y}")
        
        # check for direction of play
        for _, period_row in self.periods.iterrows():
            frame = period_row["start_frame"]
            if len(self.tracking_data[self.tracking_data["timestamp"] == frame].index) == 0:
                continue
            idx = self.tracking_data[self.tracking_data["timestamp"] == frame].index[0]
            period = period_row["period"]
            home_x = [x for x in self.home_players_column_ids if "_x" in x]
            away_x = [x for x in self.away_players_column_ids if "_x" in x]
            if self.tracking_data.loc[idx, home_x].mean() > 0:
                centroid_x = self.tracking_data.loc[idx, home_x].mean()
                raise DataBallPyError(f"The home team should be represented as playing\
from left to right the whole match. At the start of period {period} the x centroid of \
the home team is {centroid_x}.")
        
            if self.tracking_data.loc[idx, away_x].mean() < 0:
                centroid_x = self.tracking_data.loc[idx, away_x].mean()
                raise DataBallPyError(f"The away team should be represented as playing\
from right to left the whole match. At the start  of period {period} the x centroid of \
the away team is {centroid_x}.")


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
