import os
import pickle
import warnings
from dataclasses import dataclass, field
from functools import wraps
from typing import List

import pandas as pd

from databallpy.errors import DataBallPyError
from databallpy.load_data.event_data.dribble_event import DribbleEvent
from databallpy.load_data.event_data.pass_event import PassEvent
from databallpy.load_data.event_data.shot_event import ShotEvent
from databallpy.utils.synchronise_tracking_and_event_data import (
    synchronise_tracking_and_event_data,
)
from databallpy.utils.utils import MISSING_INT
from databallpy.warnings import DataBallPyWarning


def requires_tracking_data(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args[0].tracking_data) > 0:
            return func(*args, **kwargs)
        else:
            raise DataBallPyError(
                "No tracking data available, please load \
Match object with tracking data first."
            )

    return wrapper


def requires_event_data(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args[0].event_data) > 0:
            return func(*args, **kwargs)
        else:
            raise DataBallPyError(
                "No event data available, please load\
 Match object with event data first."
            )

    return wrapper


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
        country (str): The country where the match was played.
        shot_events (dict): A dictionary with all instances of shot events.
        dribble_events (dict): A dictionary with all instances of dribble events.
        pass_events (dict): A dictionary with all instances of pass events.
        allow_synchronise_tracking_and_event_data (bool): If True, the tracking and
        event data can be synchronised. If False, it can not. Default = False.

    Iargs:
        name (str): The home and away team name and score: "nameH 3 - 1 nameA {date}".
        home_players_column_ids (list): All column ids of the tracking data that refer
                                        to information about the home team players.
        away_players_column_ids (list): All column ids of the tracking data that refer
                                        to information about the away team players.
        preprocessing_status (dict): A string with the status of the preprocessing.
        shots_df (pd.DataFrame): A dataframe with all info of shots in the match.
        dribbles_df (pd.DataFrame): A dataframe with all info of dribbles in the match.
        passes_df (pd.DataFrame): A dataframe with all info of passes in the match.


    Funcs
        player_column_id_to_full_name: Simple function to get the full name of a player
                                       from the column id
        synchronise_tracking_and_event_data: Synchronises the tracking and event data
        save_match: Saves the match to a pickle file.
        copy: Creates a copy of the match object.
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
    country: str
    shot_events: dict = field(default_factory=dict)
    dribble_events: dict = field(default_factory=dict)
    pass_events: dict = field(default_factory=dict)
    allow_synchronise_tracking_and_event_data: bool = False
    extra_data: pd.DataFrame = None
    _shots_df: pd.DataFrame = None
    _dribbles_df: pd.DataFrame = None
    _passes_df: pd.DataFrame = None
    # to save the preprocessing status
    _is_synchronised: bool = False

    def __repr__(self):
        return "databallpy.match.Match object: " + self.name

    def __post_init__(self):
        check_inputs_match_object(self)

    @property
    def is_synchronised(self) -> bool:
        return self._is_synchronised

    @property
    def name(self) -> str:
        home_text = f"{self.home_team_name} {self.home_score}"
        away_text = f"{self.away_score} {self.away_team_name}"
        if "start_datetime_td" in self.periods.columns:
            date = str(
                self.periods.loc[self.periods["period"] == 1, "start_datetime_td"].iloc[
                    0
                ]
            )[:19]
        else:
            date = str(
                self.periods.loc[self.periods["period"] == 1, "start_datetime_ed"].iloc[
                    0
                ]
            )[:19]
        return f"{home_text} - {away_text} {date}"

    @requires_tracking_data
    def home_players_column_ids(self) -> List[str]:
        return [id for id in self.tracking_data.columns if "home" in id]

    @requires_tracking_data
    def away_players_column_ids(self) -> List[str]:
        return [id for id in self.tracking_data.columns if "away" in id]

    @requires_tracking_data
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

    @property
    def preprocessing_status(self):
        return f"Preprocessing status:\n\tis_synchronised = {self.is_synchronised}"

    def player_id_to_column_id(self, player_id: int) -> str:
        """Simple function to get the column id based on player id

        Args:
            player_id (int): id of the player

        Returns:
            str: column id of the player, for instance "home_1"
        """
        if (self.home_players["id"].eq(player_id)).any():
            num = self.home_players[self.home_players["id"] == player_id][
                "shirt_num"
            ].iloc[0]
            return f"home_{num}"
        elif (self.away_players["id"].eq(player_id)).any():
            num = self.away_players[self.away_players["id"] == player_id][
                "shirt_num"
            ].iloc[0]
            return f"away_{num}"
        else:
            raise ValueError(f"{player_id} is not in either one of the teams")

    @property
    @requires_event_data
    def shots_df(self) -> pd.DataFrame:
        """Function to get all shots in the match

        Returns:
            pd.DataFrame: DataFrame with all information of the shots in the match"""

        if self._shots_df is None:
            if (
                not self.is_synchronised
                and self.allow_synchronise_tracking_and_event_data
            ):
                self.synchronise_tracking_and_event_data()
            if self.is_synchronised:
                self.add_tracking_data_features_to_shots()
            res_dict = {
                "event_id": [shot.event_id for shot in self.shot_events.values()],
                "player_id": [shot.player_id for shot in self.shot_events.values()],
                "period_id": [shot.period_id for shot in self.shot_events.values()],
                "minutes": [shot.minutes for shot in self.shot_events.values()],
                "seconds": [shot.seconds for shot in self.shot_events.values()],
                "datetime": [shot.datetime for shot in self.shot_events.values()],
                "start_x": [shot.start_x for shot in self.shot_events.values()],
                "start_y": [shot.start_y for shot in self.shot_events.values()],
                "team_id": [shot.team_id for shot in self.shot_events.values()],
                "shot_outcome": [
                    shot.shot_outcome for shot in self.shot_events.values()
                ],
                "y_target": [shot.y_target for shot in self.shot_events.values()],
                "z_target": [shot.z_target for shot in self.shot_events.values()],
                "body_part": [shot.body_part for shot in self.shot_events.values()],
                "type_of_play": [
                    shot.type_of_play for shot in self.shot_events.values()
                ],
                "first_touch": [shot.first_touch for shot in self.shot_events.values()],
                "created_oppertunity": [
                    shot.created_oppertunity for shot in self.shot_events.values()
                ],
                "related_event_id": [
                    shot.related_event_id for shot in self.shot_events.values()
                ],
                "ball_goal_distance": [
                    shot.ball_goal_distance for shot in self.shot_events.values()
                ],
                "ball_gk_distance": [
                    shot.ball_gk_distance for shot in self.shot_events.values()
                ],
                "shot_angle": [shot.shot_angle for shot in self.shot_events.values()],
                "gk_angle": [shot.gk_angle for shot in self.shot_events.values()],
                "pressure_on_ball": [
                    shot.pressure_on_ball for shot in self.shot_events.values()
                ],
                "n_obstructive_players": [
                    shot.n_obstructive_players for shot in self.shot_events.values()
                ],
            }
            self._shots_df = pd.DataFrame(res_dict)
        return self._shots_df

    @property
    @requires_event_data
    def dribbles_df(self) -> pd.DataFrame:
        """Function to get all info of the dribbles in the match

        Returns:
            pd.DataFrame: DataFrame with all information of the dribbles in the match"""

        if self._dribbles_df is None:
            res_dict = {
                "event_id": [
                    dribble.event_id for dribble in self.dribble_events.values()
                ],
                "player_id": [
                    dribble.player_id for dribble in self.dribble_events.values()
                ],
                "period_id": [
                    dribble.period_id for dribble in self.dribble_events.values()
                ],
                "minutes": [
                    dribble.minutes for dribble in self.dribble_events.values()
                ],
                "seconds": [
                    dribble.seconds for dribble in self.dribble_events.values()
                ],
                "datetime": [
                    dribble.datetime for dribble in self.dribble_events.values()
                ],
                "start_x": [
                    dribble.start_x for dribble in self.dribble_events.values()
                ],
                "start_y": [
                    dribble.start_y for dribble in self.dribble_events.values()
                ],
                "team_id": [
                    dribble.team_id for dribble in self.dribble_events.values()
                ],
                "related_event_id": [
                    dribble.related_event_id for dribble in self.dribble_events.values()
                ],
                "duel_type": [
                    dribble.duel_type for dribble in self.dribble_events.values()
                ],
                "outcome": [
                    dribble.outcome for dribble in self.dribble_events.values()
                ],
                "has_opponent": [
                    dribble.has_opponent for dribble in self.dribble_events.values()
                ],
            }
            self._dribbles_df = pd.DataFrame(res_dict)
        return self._dribbles_df

    @property
    @requires_event_data
    def passes_df(self) -> pd.DataFrame:
        """Function to get all info of the passes in the match

        Returns:
            pd.DataFrame: DataFrame with all information of the passes in the match"""

        if self._passes_df is None:
            res_dict = {
                "event_id": [pass_.event_id for pass_ in self.pass_events.values()],
                "player_id": [pass_.player_id for pass_ in self.pass_events.values()],
                "period_id": [pass_.period_id for pass_ in self.pass_events.values()],
                "minutes": [pass_.minutes for pass_ in self.pass_events.values()],
                "seconds": [pass_.seconds for pass_ in self.pass_events.values()],
                "datetime": [pass_.datetime for pass_ in self.pass_events.values()],
                "start_x": [pass_.start_x for pass_ in self.pass_events.values()],
                "start_y": [pass_.start_y for pass_ in self.pass_events.values()],
                "team_id": [pass_.team_id for pass_ in self.pass_events.values()],
                "outcome": [pass_.outcome for pass_ in self.pass_events.values()],
                "end_x": [pass_.end_x for pass_ in self.pass_events.values()],
                "end_y": [pass_.end_y for pass_ in self.pass_events.values()],
                "length": [pass_.length for pass_ in self.pass_events.values()],
                "angle": [pass_.angle for pass_ in self.pass_events.values()],
                "pass_type": [pass_.pass_type for pass_ in self.pass_events.values()],
                "set_piece": [pass_.set_piece for pass_ in self.pass_events.values()],
            }
            self._passes_df = pd.DataFrame(res_dict)
        return self._passes_df

    @requires_event_data
    @requires_tracking_data
    def add_tracking_data_features_to_shots(self):
        """Function to add tracking data features to the shots. This function
              should be run after the tracking and event data are synchronised.

        Raises:
            ValueError: if the tracking and event data are not synchronised yet
        """
        if not self.is_synchronised:
            raise DataBallPyError(
                "Tracking and event data are not synchronised yet. Please run the"
                " synchronise_tracking_and_event_data() method first"
            )

        for shot in self.shot_events.values():
            team_side = (
                "home" if shot.player_id in self.home_players["id"].values else "away"
            )
            column_id = self.player_id_to_column_id(shot.player_id)
            tracking_data_frame = self.tracking_data.loc[
                self.tracking_data["event_id"] == shot.event_id
            ]

            # if, for some reason, the shot is not found in the tracking data, continue
            if len(tracking_data_frame) == 0:
                continue
            tracking_data_frame = tracking_data_frame.iloc[0]

            if team_side == "home":
                mask = (
                    (
                        (
                            self.away_players["start_frame"]
                            <= tracking_data_frame["frame"]
                        )
                        | (self.away_players["start_frame"] == MISSING_INT)
                    )
                    & (
                        (self.away_players["end_frame"] >= tracking_data_frame["frame"])
                        | (self.away_players["end_frame"] == MISSING_INT)
                    )
                    & (self.away_players["position"] == "goalkeeper")
                )
                gk_column_id = (
                    f"away_{self.away_players.loc[mask, 'shirt_num'].iloc[0]}"
                )
            else:
                mask = (
                    (
                        (
                            self.home_players["start_frame"]
                            <= tracking_data_frame["frame"]
                        )
                        | (self.home_players["start_frame"] == MISSING_INT)
                    )
                    & (
                        (self.home_players["end_frame"] >= tracking_data_frame["frame"])
                        | (self.home_players["end_frame"] == MISSING_INT)
                    )
                    & (self.home_players["position"] == "goalkeeper")
                )

                gk_column_id = (
                    f"home_{self.home_players.loc[mask, 'shirt_num'].iloc[0]}"
                )

            shot.add_tracking_data_features(
                tracking_data_frame,
                team_side,
                self.pitch_dimensions,
                column_id,
                gk_column_id,
            )

    @requires_tracking_data
    @requires_event_data
    def synchronise_tracking_and_event_data(
        self, n_batches_per_half: int = 100, verbose: bool = True
    ):
        """Function that synchronises tracking and event data using Needleman-Wunsch
           algorithmn. Based on: https://kwiatkowski.io/sync.soccer

        Args:
            match (Match): Match object
            n_batches_per_half (int): the number of batches that are created per half.
            A higher number of batches reduces the time the code takes to load, but
            reduces the accuracy for events close to the splits. Default = 100
            verbose (bool, optional): Wheter or not to print info about the progress
            in the terminal. Defaults to True.

        Currently works for the following databallpy events:
            'pass', 'shot', and 'dribble'

        """
        if not self.allow_synchronise_tracking_and_event_data:
            raise DataBallPyError(
                "Synchronising tracking and event data is not allowed."
                "The quality of the data is not good enough to ensure valid results."
            )
        synchronise_tracking_and_event_data(self, n_batches_per_half, verbose)

    def __eq__(self, other):
        if isinstance(other, Match):
            result = [
                self.tracking_data.equals(other.tracking_data),
                self.tracking_data_provider == other.tracking_data_provider,
                self.event_data.round(6).equals(other.event_data.round(6)),
                self.pitch_dimensions == other.pitch_dimensions,
                self.periods.equals(other.periods),
                self.frame_rate == other.frame_rate
                if not pd.isnull(self.frame_rate)
                else pd.isnull(other.frame_rate),
                self.home_team_id == other.home_team_id,
                self.home_team_name == other.home_team_name,
                self.home_formation == other.home_formation,
                self.home_players.equals(other.home_players),
                self.home_score == other.home_score
                if not pd.isnull(self.home_score)
                else pd.isnull(other.home_score),
                self.away_team_id == other.away_team_id,
                self.away_team_name == other.away_team_name,
                self.away_formation == other.away_formation,
                self.away_players.equals(other.away_players),
                self.away_score == other.away_score
                if not pd.isnull(self.away_score)
                else pd.isnull(other.away_score),
                self.shot_events == other.shot_events,
                self._shots_df.equals(other._shots_df)
                if self._shots_df is not None
                else other._shots_df is None,
                self.dribble_events == other.dribble_events,
                self._dribbles_df.equals(other._dribbles_df)
                if self._dribbles_df is not None
                else other._dribbles_df is None,
                self.pass_events == other.pass_events,
                self._passes_df.equals(other._passes_df)
                if self._passes_df is not None
                else other._passes_df is None,
                self.country == other.country,
                self.extra_data.equals(other.extra_data)
                if self.extra_data is not None
                else other.extra_data is None,
            ]
            return all(result)
        else:
            return False

    def copy(self):
        """Function to return a copy of the current match object"""
        return Match(
            tracking_data=self.tracking_data.copy(),
            tracking_data_provider=self.tracking_data_provider,
            event_data=self.event_data.copy(),
            event_data_provider=self.event_data_provider,
            pitch_dimensions=list(self.pitch_dimensions),
            periods=self.periods.copy(),
            frame_rate=self.frame_rate,
            home_team_id=self.home_team_id,
            home_formation=self.home_formation,
            home_score=self.home_score,
            home_team_name=self.home_team_name,
            home_players=self.home_players.copy(),
            away_team_id=self.away_team_id,
            away_formation=self.away_formation,
            away_score=self.away_score,
            away_team_name=self.away_team_name,
            away_players=self.away_players.copy(),
            shot_events=self.shot_events.copy(),
            _shots_df=self._shots_df.copy() if self._shots_df is not None else None,
            dribble_events=self.dribble_events.copy(),
            _dribbles_df=self._dribbles_df.copy()
            if self._dribbles_df is not None
            else None,
            pass_events=self.pass_events.copy(),
            _passes_df=self._passes_df.copy() if self._passes_df is not None else None,
            country=self.country,
            extra_data=self.extra_data,
        )

    def save_match(
        self, name: str = None, path: str = None, verbose: bool = True
    ) -> None:
        """Function to save the current match object to a pickle file

        Args:
            name (str): name of the pickle file, if not provided or not a string,
            the name will be the name of the match
            path (str): path to the directory where the pickle file will be saved, if
            not provided, the current working directory will be used
            verbose (bool): if True, saved name will be printed
        """
        if name is None or not isinstance(name, str):
            name = self.name
        if path is None:
            path = os.getcwd()
        with open(os.path.join(path, name + ".pickle"), "wb") as f:
            pickle.dump(self, f)
        if verbose:
            print(f"Match saved to {os.path.join(path, name)}.pickle")


def check_inputs_match_object(match: Match):
    """Function to check if the inputs of the match object are correct

    Args:
        match (Match): match object
    """
    # tracking_data
    if not isinstance(match.tracking_data, pd.DataFrame):
        raise TypeError(
            f"tracking data should be a pandas df, not a {type(match.tracking_data)}"
        )
    if len(match.tracking_data) > 0:
        for col in ["frame", "ball_x", "ball_y"]:
            if col not in match.tracking_data.columns.to_list():
                raise ValueError(
                    f"No {col} in tracking_data columns, this is manditory!"
                )

        # tracking_data_provider
        if not isinstance(match.tracking_data_provider, str):
            raise TypeError(
                f"tracking data provider should be a string, not a \
                    {type(match.tracking_data_provider)}"
            )

        # check if the first frame is at (0, 0)
        ball_alive_mask = match.tracking_data["ball_status"] == "alive"
        first_frame = match.tracking_data.loc[
            ball_alive_mask, "ball_x"
        ].first_valid_index()
        if (
            not abs(match.tracking_data.loc[first_frame, "ball_x"]) < 5.0
            or not abs(match.tracking_data.loc[first_frame, "ball_y"]) < 5.0
        ):
            x_start = match.tracking_data.loc[first_frame, "ball_x"]
            y_start = match.tracking_data.loc[first_frame, "ball_y"]
            warnings.warn(
                DataBallPyWarning(
                    "The middle point of the pitch should be (0, 0), "
                    f"now the kick-off is at ({x_start}, {y_start}). "
                    "Either the recording has started too late or the ball_status "
                    "is not set to 'alive' in the beginning. Please check and change "
                    "the tracking data if desired."
                    "\n NOTE: The quality of the synchronisation of the tracking "
                    "and event data might be affected."
                )
            )

    # event_data
    if not isinstance(match.event_data, pd.DataFrame):
        raise TypeError(
            f"event data should be a pandas df, not a {type(match.event_data)}"
        )
    if len(match.event_data) > 0:
        for col in [
            "event_id",
            "databallpy_event",
            "period_id",
            "team_id",
            "player_id",
            "start_x",
            "start_y",
            "datetime",
        ]:
            if col not in match.event_data.columns.to_list():
                raise ValueError(f"{col} not in event data columns, this is manditory!")

        if not pd.api.types.is_datetime64_any_dtype(match.event_data["datetime"]):
            raise TypeError(
                f"datetime column in event_data should be a datetime dtype, not a \
                    {type(match.event_data['datetime'])}"
            )

        if match.event_data["datetime"].dt.tz is None:
            raise ValueError("datetime column in event_data should have a timezone")

        # event_data_provider
        if not isinstance(match.event_data_provider, str):
            raise TypeError(
                f"event data provider should be a string, not a \
                    {type(match.event_data_provider)}"
            )

    # pitch_dimensions
    if not isinstance(match.pitch_dimensions, list):
        raise TypeError(
            f"pitch_dimensions ({match.pitch_dimensions}) should be a \
                list, not a {type(match.pitch_dimensions)}"
        )
    if not len(match.pitch_dimensions) == 2:
        raise ValueError(
            f"pitch_dimensions should contain, two values: a length and a width \
                of the pitch, current input is {match.pitch_dimensions}"
        )
    if not all([isinstance(x, float) for x in match.pitch_dimensions]):
        raise TypeError(
            f"Both values in pitch dimensions should by floats, current inputs \
                {[type(x) for x in match.pitch_dimensions]}"
        )

    # periods
    if not isinstance(match.periods, pd.DataFrame):
        raise TypeError(
            f"periods_frames should be a pandas dataframe, not a \
                {type(match.periods)}"
        )
    if "period" not in match.periods.columns:
        raise ValueError("'period' should be one of the columns in period_frames")
    if any(
        [x not in match.periods["period"].value_counts().index for x in [1, 2, 3, 4, 5]]
    ) or not all(match.periods["period"].value_counts() == 1):
        res = match.periods["period"]
        raise ValueError(
            f"'period' column in period_frames should contain only the values \
                [1, 2, 3, 4, 5]. Now it's {res}"
        )

    for col in [col for col in match.periods if "datetime" in col]:
        if pd.isnull(match.periods[col]).all():
            continue
        if match.periods[col].dt.tz is None:
            raise ValueError(f"{col} column in periods should have a timezone")

    # frame_rate
    if not pd.isnull(match.frame_rate) and not match.frame_rate == MISSING_INT:
        if not isinstance(match.frame_rate, int):
            raise TypeError(
                f"frame_rate should be an integer, not a {type(match.frame_rate)}"
            )
        if match.frame_rate < 1:
            raise ValueError(f"frame_rate should be positive, not {match.frame_rate}")

    # team id's
    for team, team_id in zip(
        ["home", "away"], [match.home_team_id, match.away_team_id]
    ):
        if not isinstance(team_id, int) and not isinstance(team_id, str):
            raise TypeError(
                f"{team} team id should be an integer or string, not a \
                    {type(team_id)}"
            )

    # team names
    for team, name in zip(
        ["home", "away"], [match.home_team_name, match.away_team_name]
    ):
        if not isinstance(name, str):
            raise TypeError(f"{team} team name should be a string, not a {type(name)}")

    # team scores
    for team, score in zip(["home", "away"], [match.home_score, match.away_score]):
        if not pd.isnull(score) and not score == MISSING_INT:
            if not isinstance(score, int):
                raise TypeError(
                    f"{team} team score should be an integer, not a {type(score)}"
                )
            if score < 0:
                raise ValueError(f"{team} team score should positive, not {score}")

    # team formations
    for team, form in zip(
        ["home", "away"], [match.home_formation, match.away_formation]
    ):
        if not isinstance(form, str):
            raise TypeError(
                f"{team} team formation should be a string, not a {type(form)}"
            )
        if len(form) > 5:
            raise ValueError(
                f"{team} team formation should be of length 5 or smaller \
                    ('1433'), not {len(form)}"
            )

    # team players
    for team, players in zip(
        ["home", "away"], [match.home_players, match.away_players]
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

        # check for direction of play
        for _, period_row in match.periods.iterrows():
            if "start_frame" not in period_row.index:
                continue
            frame = period_row["start_frame"]
            if (
                len(match.tracking_data[match.tracking_data["frame"] == frame].index)
                == 0
            ):
                continue
            idx = match.tracking_data[match.tracking_data["frame"] == frame].index[0]
            period = period_row["period"]
            home_x = [x for x in match.home_players_column_ids() if "_x" in x]
            away_x = [x for x in match.away_players_column_ids() if "_x" in x]
            if match.tracking_data.loc[idx, home_x].mean() > 0:
                centroid_x = match.tracking_data.loc[idx, home_x].mean()
                raise DataBallPyError(
                    "The home team should be represented as playing from left to "
                    f"right the whole match. At the start of period {period} the x "
                    f"centroid of the home team is {centroid_x}."
                )

            if match.tracking_data.loc[idx, away_x].mean() < 0:
                centroid_x = match.tracking_data.loc[idx, away_x].mean()
                raise DataBallPyError(
                    "The away team should be represented as playingfrom right to "
                    f"left the whole match. At the start  of period {period} the x "
                    f"centroid ofthe away team is {centroid_x}."
                )

        # check databallpy_events
        databallpy_events = [match.dribble_events, match.shot_events, match.pass_events]
        for event_dict, event_name, event_type in zip(
            databallpy_events,
            ["dribble", "shot", "pass"],
            [DribbleEvent, ShotEvent, PassEvent],
        ):
            if not isinstance(event_dict, dict):
                raise TypeError(
                    f"{event_name}_events should be a dictionary, not a "
                    f"{type(event_dict)}"
                )

            for event in event_dict.values():
                if not isinstance(event, event_type):
                    raise TypeError(
                        f"{event_name}_events should contain only {event_type} objects,"
                        f" not {type(event)}"
                    )

        # country
        if not isinstance(match.country, str):
            raise TypeError(f"country should be a string, not a {type(match.country)}")
