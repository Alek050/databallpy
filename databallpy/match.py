import os
import pickle
import warnings
from dataclasses import dataclass, field, fields
from functools import wraps

import numpy as np
import pandas as pd
from tqdm import tqdm

from databallpy.events import (
    DribbleEvent,
    IndividualCloseToBallEvent,
    PassEvent,
    ShotEvent,
)
from databallpy.utils.constants import DATABALLPY_POSITIONS, MISSING_INT
from databallpy.utils.errors import DataBallPyError
from databallpy.utils.logging import create_logger, logging_wrapper
from databallpy.utils.match_utils import (
    create_event_attributes_dataframe,
    player_column_id_to_full_name,
    player_id_to_column_id,
)
from databallpy.utils.synchronise_tracking_and_event_data import (
    align_event_data_datetime,
    pre_compute_synchronisation_variables,
    synchronise_tracking_and_event_data,
)
from databallpy.utils.utils import (
    _copy_value_,
    _values_are_equal_,
    get_next_possession_frame,
)
from databallpy.utils.warnings import DataBallPyWarning

LOGGER = create_logger(__file__)


def requires_tracking_data(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args[0].tracking_data) > 0:
            return func(*args, **kwargs)
        else:
            raise DataBallPyError(
                "No tracking data available, please load "
                "Match object with tracking data first."
            )

    return wrapper


def requires_event_data(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args[0].event_data) > 0:
            return func(*args, **kwargs)
        else:
            raise DataBallPyError(
                "No event data available, please load "
                "Match object with event data first."
            )

    return wrapper


@dataclass
class Match:
    """
    This is the match class. It contains all information of the match and has
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
        other_events (dict): A dictionary with all instances of other supported events.
        allow_synchronise_tracking_and_event_data (bool): If True, the tracking and
        event data can be synchronised. If False, it can not. Default = False.
    """

    tracking_data: pd.DataFrame
    tracking_data_provider: str
    event_data: pd.DataFrame
    event_data_provider: str
    pitch_dimensions: list[float, float]
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
    shot_events: dict[int | str, ShotEvent] = field(default_factory=dict)
    dribble_events: dict[int | str, DribbleEvent] = field(default_factory=dict)
    pass_events: dict[int | str, PassEvent] = field(default_factory=dict)
    other_events: dict[int | str, IndividualCloseToBallEvent] = field(
        default_factory=dict
    )
    allow_synchronise_tracking_and_event_data: bool = False
    _shots_df: pd.DataFrame = None
    _dribbles_df: pd.DataFrame = None
    _passes_df: pd.DataFrame = None
    _other_events_df: pd.DataFrame = None
    # to save the preprocessing status
    _is_synchronised: bool = False
    # to indicate if the timestamps are precise or just the proposed timestamps of the
    # match (precisely 20:00 for start of match). This is important for the
    # synchronisation of the tracking and event data
    _tracking_timestamp_is_precise: bool = False
    _event_timestamp_is_precise: bool = False
    _periods_changed_playing_direction: list[int] = None

    def __repr__(self):
        return "databallpy.match.Match object: " + self.name

    def __post_init__(self):
        check_inputs_match_object(self)

    @property
    def tracking_timestamp_is_precise(self) -> bool:
        """Function to check if the tracking timestamps are precise or not
        Timesamp is considered precise if the start of the match is not exactly
        at the start of the initial match time (e.g. 20:00:00), but at the
        actual start of the match (e.g. 20:00:03.2378).

        Returns:
            bool: True if the tracking timestamps are precise, False otherwise
        """
        return self._tracking_timestamp_is_precise

    @property
    def event_timestamp_is_precise(self) -> bool:
        """Function to check if the event timestamps are precise or not
        Timesamp is considered precise if the start of the match is not exactly
        at the start of the initial match time (e.g. 20:00:00), but at the
        actual start of the match (e.g. 20:00:03.2378).

        Returns:
            bool: True if the event timestamps are precise, False otherwise
        """
        return self._event_timestamp_is_precise

    @property
    def is_synchronised(self) -> bool:
        """Function to check if the tracking and event data are synchronised

        Returns:
            bool: True if the tracking and event data are synchronised, False otherwise
        """
        return self._is_synchronised

    @property
    def date(self) -> pd.Timestamp | None:
        """Function to get the date of the match

        Returns:
            pd.Timestamp | None: The date of the match
        """
        if "start_datetime_td" in self.periods.columns:
            return self.periods.loc[
                self.periods["period_id"] == 1, "start_datetime_td"
            ].iloc[0]
        elif "start_datetime_ed" in self.periods.columns:
            return self.periods.loc[
                self.periods["period_id"] == 1, "start_datetime_ed"
            ].iloc[0]

        return None

    @property
    def all_events(
        self,
    ) -> dict[
        int | str, DribbleEvent | PassEvent | ShotEvent | IndividualCloseToBallEvent
    ]:
        """Function to get all events in the match

        Returns:
            dict[
                int | str,
                DribbleEvent | PassEvent | ShotEvent | IndividualCloseToBallEvent
                ]: All events in the match
        """
        return {
            **self.shot_events,
            **self.dribble_events,
            **self.pass_events,
            **self.other_events,
        }

    @property
    def name(self) -> str:
        """Function to get the name of the match

        Returns:
            str: The name of the match
        """
        home_text = f"{self.home_team_name} {self.home_score}"
        away_text = f"{self.away_score} {self.away_team_name}"

        date = self.date
        if date is None:
            return f"{home_text} - {away_text}"

        return f"{home_text} - {away_text} {date.strftime('%Y-%m-%d %H:%M:%S')}"

    @requires_tracking_data
    def get_column_ids(
        self,
        team: str | None = None,
        positions: list[str] = ["goalkeeper", "defender", "midfielder", "forward"],
        min_minutes_played: float | int = 0.1,
    ) -> list[str]:
        """Function to get the column ids that are used in the tracking data. With this
        function you can filter on team side, position, or minimum minutes played.
        If no arguments are specified, all column ids are returned of players that j
        played at least 0.1 minute.

        Args:
            team (str | None, optional): Which team to add, can be {home, away, None}.
                If None, both teams are added. Defaults to None.
            positions (list[str], optional): The positions to include
                {goalkeeper, defender, midfielder, forward}. Defaults to
                ["goalkeeper", "defender", "midfielder", "forward"].
            min_minutes_played (float | int, optional): The minimum number of minutes a
                player needs to have played during the match to be returned.
                Defaults to 1.0.

        Raises:
            ValueError: If team is not in {None, home, away}
            ValueError: If there is an unknown position
            TypeError: if min_minutes_played is not numeric

        Returns:
            list[str]: The column ids of the players.
        """
        if team and team not in ["home", "away"]:
            raise ValueError(f"team should be either 'home' or 'away', not {team}")

        for pos in positions:
            if pos not in DATABALLPY_POSITIONS:
                raise ValueError(
                    f"Position {pos} is not supported in databallpy, should be in "
                    f"{DATABALLPY_POSITIONS}"
                )

        if not isinstance(min_minutes_played, (float, int, np.floating, np.integer)):
            raise TypeError("min_minutes_played should be a float or integer")

        if team:
            players = self.home_players if team == "home" else self.away_players
        else:
            players = pd.concat(
                [self.home_players, self.away_players], ignore_index=True
            )

        if len(positions) > 0:
            if "position" not in players.columns:
                warnings.warn(
                    "No position information found in players, can not filter on "
                    "positions. Returning all positions.",
                    DataBallPyWarning,
                )
            else:
                players = players[players["position"].isin(positions)]

        if not (players["start_frame"] == MISSING_INT).all():
            players = players[
                (players["end_frame"] - players["start_frame"]) / self.frame_rate / 60
                >= min_minutes_played
            ]

        return [
            f"home_{int(row.shirt_num)}"
            if row.id in self.home_players["id"].to_list()
            else f"away_{int(row.shirt_num)}"
            for row in players.itertuples(index=False)
        ]

    @requires_tracking_data
    def home_players_column_ids(self) -> list[str]:
        """Function to get all column ids of the tracking data that refer to information
        about the home team players

        Depreciation: This function is depreciated and will be removed in version
        0.7.0. Please use match.get_column_ids(team="home").

        Returns:
            list[str]: All column ids of the home team players
        """

        warnings.warn(
            "match.home_players_column_ids is depreciated and will be removed in "
            "version 0.7. Please use match.get_column_ids(team='home')",
            DeprecationWarning,
        )
        return [
            id[:-2]
            for id in self.tracking_data.columns
            if id[:4] == "home" and id[-2:] == "_x"
        ]

    @requires_tracking_data
    def away_players_column_ids(self) -> list[str]:
        """Function to get all column ids of the tracking data that refer to information
        about the away team players

        Depreciation: This function is depreciated and will be removed in version
        0.7.0. Please use match.get_column_ids(team="away").

        Returns:
            list[str]: All column ids of the away team players
        """

        warnings.warn(
            "match.away_players_column_ids is depreciated and will be removed in "
            "version 0.7. Please use match.get_column_ids(team='away')",
            DeprecationWarning,
        )
        return [
            id[:-2]
            for id in self.tracking_data.columns
            if id[:4] == "away" and id[-2:] == "_x"
        ]

    @requires_tracking_data
    def player_column_id_to_full_name(self, column_id: str) -> str:
        """Simple function to get the full name of a player from the column id

        Args:
            column_id (str): the column id of a player, for instance "home_1"

        Returns:
            str: full name of the player
        """
        return player_column_id_to_full_name(
            self.home_players, self.away_players, column_id
        )

    @property
    def preprocessing_status(self):
        return f"Preprocessing status:\n\tis_synchronised = {self.is_synchronised}"

    @logging_wrapper(__file__)
    def player_id_to_column_id(self, player_id: int) -> str:
        """Simple function to get the column id based on player id

        Args:
            player_id (int): id of the player

        Returns:
            str: column id of the player, for instance "home_1"
        """
        return player_id_to_column_id(self.home_players, self.away_players, player_id)

    @property
    @requires_event_data
    def shots_df(self) -> pd.DataFrame:
        """Function to get all shots in the match

        Returns:
            pd.DataFrame: DataFrame with all information of the shots in the match
        """

        if self._shots_df is None:
            LOGGER.info("Creating the match._shots_df dataframe in match.shots_df")

            self._shots_df = create_event_attributes_dataframe(self.shot_events)

            LOGGER.info(
                "Successfully created match._shots_df dataframe in match.shots_df"
            )
        LOGGER.info("Returning the pre-loaded match._shots_df in match.shots_df")
        return self._shots_df

    @property
    @requires_event_data
    def dribbles_df(self) -> pd.DataFrame:
        """Function to get all info of the dribbles in the match

        Returns:
            pd.DataFrame: DataFrame with all information of the dribbles in the match"""

        if self._dribbles_df is None:
            LOGGER.info("Creating the match._dribbles_df dataframe in match.dribbles_df")
            self._dribbles_df = create_event_attributes_dataframe(self.dribble_events)

            LOGGER.info(
                "Successfully created match._dribbles_df dataframe in match.dribbles_df"
            )
        LOGGER.info("Returning the pre-loaded match._dribbles_df in match.dribbles_df")
        return self._dribbles_df

    @property
    @requires_event_data
    def passes_df(self) -> pd.DataFrame:
        """Function to get all info of the passes in the match

        Returns:
            pd.DataFrame: DataFrame with all information of the passes in the match"""

        if self._passes_df is None:
            LOGGER.info("Creating the match._passes_df dataframe in match.passes_df")
            self._passes_df = create_event_attributes_dataframe(self.pass_events)

            LOGGER.info(
                "Successfully created match._passes_df dataframe in match.passes_df"
            )
        LOGGER.info("Returning the pre-loaded match._passes_df in match.passes_df")
        return self._passes_df

    @property
    @requires_event_data
    def other_events_df(self) -> pd.DataFrame:
        """Function to get all info of the other events in the match

        Returns:
            pd.DataFrame: DataFrame with all information of the other events in the
                match
        """

        if self._other_events_df is None:
            LOGGER.info(
                "Creating the match.other_events_df dataframe in match.other_events_df"
            )
            other_events_df = create_event_attributes_dataframe(
                self.other_events, add_name=True
            )

            LOGGER.info(
                "Successfully created match.other_events_df dataframe in "
                "match.other_events_df"
            )
        return other_events_df

    @requires_event_data
    def get_event(self, event_id: int):
        """Function to get the event with the given event_id

        Args:
            event_id (int): The id of the event

        Raises:
            ValueError: if the event with the given event_id is not found in the match

        Returns:
            Databallpy Event: The event with the given event_id
        """
        if event_id in self.all_events.keys():
            return self.all_events[event_id]
        else:
            raise ValueError(f"Event with id {event_id} not found in the match.")

    def get_frames(
        self, frames: int | list[int], playing_direction: str = "team_oriented"
    ) -> pd.DataFrame:
        """Function to get the frame of the match with the given frame

        Args:
            frames (int|list[int]): The frames of the match
            playing_direction (str, optional): The coordinate system of the frame.
                Defaults to "team_oriented", options are {team_oriented,
                possession_oriented}. For more info on the coordinate systems, see
                the documentation

        Returns:
            pd.DataFrame: The frame of the match with the given frames
        """
        if isinstance(frames, (int, np.integer)):
            frames = [frames]

        unrecognized_frames = [
            frame for frame in frames if frame not in self.tracking_data["frame"].values
        ]
        if len(unrecognized_frames) > 0:
            raise ValueError(f"Frame(s) {unrecognized_frames} not found in the match.")

        if playing_direction == "team_oriented":
            return self.tracking_data.loc[self.tracking_data["frame"].isin(frames)]
        elif playing_direction == "possession_oriented":
            if (
                "ball_possession" not in self.tracking_data.columns
                or pd.isna(self.tracking_data["ball_possession"]).all()
            ):
                raise ValueError(
                    "No `ball_possession` column found in tracking data, can not get"
                    " frames in possession coordinate system without this column. "
                    "Please run the add_team_possession() function first."
                )

            # current coordinate system: home from left to right, away right to left
            suffixes = ("_x", "_y", "_vx", "_vy", "_ax", "_ay")
            cols_to_swap = [
                col for col in self.tracking_data.columns if col.endswith(suffixes)
            ]
            temp_td = self.tracking_data.loc[
                self.tracking_data["frame"].isin(frames)
            ].copy()
            temp_td.loc[
                self.tracking_data["ball_possession"] == "away", cols_to_swap
            ] *= -1
            return temp_td

        else:
            raise ValueError(f"Coordinate system {playing_direction} is not supported.")

    def get_event_frame(
        self, event_id: int | str, playing_direction: str = "team_oriented"
    ) -> pd.DataFrame:
        """Function to get the frame of the event with the given event_id

        Args:
            event_id (int | str): The id of the event
            playing_direction (str, optional): The coordinate system of the frame.
                Defaults to "team_oriented", options are {team_oriented,
                possession_oriented}. For more info on the coordinate systems, see
                the databallpy documentation

        Raises:
            ValueError: if the event with the given event_id is not found in the match
            ValueError: if the event with the given event_id is not found in the
                tracking data

        Returns:
            pd.DataFrame: The frame of the match with the given event_id
        """
        if not self._is_synchronised:
            raise DataBallPyError(
                "Tracking and event data are not synchronised yet. Please run the"
                " synchronise_tracking_and_event_data() method first."
            )
        event = self.get_event(event_id)
        frame_id = self.tracking_data.loc[
            self.tracking_data["event_id"] == event.event_id, "frame"
        ].iloc[0]
        frame = self.get_frames(frame_id, playing_direction="team_oriented")
        if playing_direction == "team_oriented":
            return frame
        elif playing_direction == "possession_oriented":
            if event.team_side == "away":
                suffixes = ("_x", "_y", "_vx", "_vy", "_ax", "_ay")
                cols_to_swap = [
                    col for col in self.tracking_data.columns if col.endswith(suffixes)
                ]
                frame = self.tracking_data.loc[
                    self.tracking_data["frame"] == frame_id
                ].copy()
                frame.loc[:, cols_to_swap] *= -1
        else:
            raise ValueError(f"Coordinate system {playing_direction} is not supported.")
        return frame

    @requires_event_data
    @requires_tracking_data
    def add_tracking_data_features_to_shots(self):
        """Function to add tracking data features to the shots. This function
              should be run after the tracking and event data are synchronised.

        Raises:
            ValueError: if the tracking and event data are not synchronised yet
        """

        LOGGER.info("Trying to add tracking data features to shots")
        if not self.is_synchronised:
            LOGGER.error(
                "Tracking and event data should be synced before adding tracking data"
                " features to the shot events."
            )
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

                gk_column_id = f"away_{self.away_players.loc[mask, 'shirt_num'].iloc[0]}"
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

                gk_column_id = f"home_{self.home_players.loc[mask, 'shirt_num'].iloc[0]}"

            shot.add_tracking_data_features(
                tracking_data_frame,
                column_id,
                gk_column_id,
            )
        LOGGER.info("Successfully added tracking data features to shots.")

    @requires_event_data
    @requires_tracking_data
    def add_tracking_data_features_to_passes(self, verbose: bool = True):
        """Function to add tracking data features to the passes. This function
              should be run after the tracking and event data are synchronised.

        Args:
            verbose (bool, optional): Wheter or not to print info about the progress
            in the terminal. Defaults to True.

        Raises:
            ValueError: if the tracking and event data are not synchronised yet
            ValueError: if the tracking data does not contain the player_possession
            column.
        """
        LOGGER.info("Trying to add tracking data features to passes")
        if not self.is_synchronised:
            message = (
                "Tracking and event data should be synced before adding tracking data"
                " features to the pass events."
            )
            LOGGER.error(message)
            raise DataBallPyError(message)
        if "player_possession" not in self.tracking_data.columns:
            message = (
                "No `player_possession` column found in tracking data, can not add"
                "tracking data features to passes without this column."
            )
            LOGGER.error(message)
            raise DataBallPyError(message)

        home_column_ids = [
            x[:-2] for x in self.tracking_data.columns if x[-2:] == "_x" and "home" in x
        ]
        away_column_ids = [
            x[:-2] for x in self.tracking_data.columns if x[-2:] == "_x" and "away" in x
        ]
        all_passes = self.pass_events.values()
        if verbose:
            all_passes = tqdm(
                all_passes,
                desc="Adding tracking data features to passes",
                leave=False,
            )

        for pass_ in all_passes:
            team_side = (
                "home" if pass_.player_id in self.home_players["id"].values else "away"
            )
            passer_column_id = self.player_id_to_column_id(pass_.player_id)
            tracking_data_frame = self.tracking_data.loc[
                self.tracking_data["event_id"] == pass_.event_id
            ]

            tracking_data_frame = tracking_data_frame.iloc[0]

            # find the end location of the pass
            # every pass should have been arrived within 10 seconds
            end_loc_diff = np.inf
            n_tries = 0
            end_loc_ed = np.array([pass_.end_x, pass_.end_y])
            start_frame_idx = tracking_data_frame.name
            possession_id = passer_column_id

            # find the end location of the pass
            while end_loc_diff > 10.0 and n_tries < 3:
                n_tries += 1
                end_frame_idx = min(
                    self.tracking_data.index[-1],
                    tracking_data_frame.name + 10 * self.frame_rate,
                )
                end_pos_frame = get_next_possession_frame(
                    self.tracking_data.loc[start_frame_idx:end_frame_idx],
                    self.tracking_data.loc[start_frame_idx],
                    possession_id,
                )
                end_loc_td = end_pos_frame[["ball_x", "ball_y"]].values
                if not pd.isnull(end_loc_ed).any():
                    end_loc_diff = np.linalg.norm(end_loc_td - end_loc_ed)
                    start_frame_idx = end_pos_frame.name
                    possession_id = end_pos_frame["player_possession"]
                else:
                    # can not compare end locations, assume tracking data is correct
                    end_loc_diff = 0.0
            if n_tries == 3 and end_loc_diff > 10:
                continue

            end_loc_td = end_loc_td.astype(float)
            start_loc = tracking_data_frame[
                [f"{passer_column_id}_x", f"{passer_column_id}_y"]
            ].values
            distance = np.linalg.norm(end_loc_td - start_loc)
            if distance < 1.0 or pd.isnull(distance):
                # end location pass is too close
                continue

            # find the proposed receiver of the pass
            if (
                not pd.isnull(end_pos_frame["player_possession"])
                and end_pos_frame["player_possession"][:4] == team_side
                and end_pos_frame["player_possession"] != passer_column_id
            ):
                receiver_column_id = end_pos_frame["player_possession"]
            else:
                # find the closest teammate to the end location
                if team_side == "home":
                    team_mate_column_ids = [
                        x
                        for x in home_column_ids
                        if x != passer_column_id
                        and not pd.isnull(end_pos_frame[x + "_x"])
                    ]
                else:
                    team_mate_column_ids = [
                        x
                        for x in away_column_ids
                        if x != passer_column_id
                        and not pd.isnull(end_pos_frame[x + "_x"])
                    ]

                team_mate_xlocs = end_pos_frame[
                    [f"{col_id}_x" for col_id in team_mate_column_ids]
                ]
                team_mate_ylocs = end_pos_frame[
                    [f"{col_id}_y" for col_id in team_mate_column_ids]
                ]
                team_mate_locs = np.array([team_mate_xlocs, team_mate_ylocs]).T.astype(
                    float
                )
                dists = np.linalg.norm(end_loc_td - team_mate_locs, axis=1)

                closest_player_idx = np.argmin(dists)
                closest_player = team_mate_column_ids[closest_player_idx]
                receiver_column_id = closest_player

            if not pd.isnull(end_loc_td).any():
                pass_.end_x = end_loc_td[0]
                pass_.end_y = end_loc_td[1]
            else:
                continue

            opponent_column_ids = (
                home_column_ids if team_side == "away" else away_column_ids
            )

            pass_.add_tracking_data_features(
                tracking_data_frame,
                passer_column_id,
                receiver_column_id,
                end_loc_td,
                self.pitch_dimensions,
                opponent_column_ids,
            )
        LOGGER.info("Successfully added tracking data features to passes.")

    @requires_tracking_data
    @requires_event_data
    def synchronise_tracking_and_event_data(
        self,
        n_batches: int | str = "smart",
        verbose: bool = True,
        offset: int = 1.0,
        cost_functions: dict = {},
    ):
        """Function that synchronises tracking and event data using Needleman-Wunsch
           algorithmn. Based on: https://kwiatkowski.io/sync.soccer

        Args:
            match (Match): Match object
            n_batches (int or str): the number of batches that are created. A
                higher number of batches reduces the time the code takes to load,
                but reduces the accuracy for events close to the splits.
                Default = 'smart'. If 'smart', the number of batches is determined
                based on the number of dead moments in the game.
            verbose (bool, optional): Wheter or not to print info about the progress
                in the terminal. Defaults to True.
            offset (float, optional): Offset in seconds that is added to the difference
                between the first event and the first tracking frame. This is done
                because this way the event is synced to the last frame the ball is close
                to a player. Which often corresponds with the event (pass and shots).
                Defaults to 1.0.
            cost_functions (dict, optional): Dictionary containing the cost functions
                that are used to calculate the similarity between the tracking and event
                data. The keys of the dictionary are the event types, the values are the
                cost functions. The cost functions will be called with the tracking data
                and the event as arguments. The cost functions should return an array
                containing the cost of the similarity between the tracking data and the
                event, scaled between 0 and 1. If no cost functions are passed, the
                default cost functions are used.

        Currently works for the following databallpy events:
            'pass', 'shot', 'dribble', and 'tackle'

        """
        LOGGER.info(f"Trying to synchronise tracking and event data of {self.name}.")
        if not self.allow_synchronise_tracking_and_event_data:
            message = (
                "Synchronising tracking and event data is not allowed. The quality "
                "checks of the tracking data showed that the quality was poor."
            )
            LOGGER.error(message)
            raise DataBallPyError(message)

        self.tracking_data = pre_compute_synchronisation_variables(
            self.tracking_data, self.frame_rate, self.pitch_dimensions
        )
        # reduce standard error by aligning trakcing and event data on first event
        changed_event_data = align_event_data_datetime(
            self.event_data.copy(), self.tracking_data, offset=offset
        )

        tracking_info, event_info = synchronise_tracking_and_event_data(
            tracking_data=self.tracking_data,
            event_data=changed_event_data,
            all_events=self.all_events,
            cost_functions=cost_functions,
            n_batches=n_batches,
            verbose=verbose,
        )
        # update tracking and event data
        self.tracking_data = pd.concat([self.tracking_data, tracking_info], axis=1)
        self.event_data = pd.concat([self.event_data, event_info], axis=1)
        self.tracking_data["databallpy_event"] = self.tracking_data[
            "databallpy_event"
        ].replace({np.nan: None})
        self.tracking_data["event_id"] = (
            self.tracking_data["event_id"]
            .infer_objects(copy=False)
            .fillna(MISSING_INT)
            .astype(np.int64)
        )
        self.tracking_data["sync_certainty"] = self.tracking_data[
            "sync_certainty"
        ].infer_objects()
        self.event_data["tracking_frame"] = (
            self.event_data["tracking_frame"]
            .infer_objects(copy=False)
            .fillna(MISSING_INT)
            .astype(np.int64)
        )
        self.event_data["sync_certainty"] = self.event_data[
            "sync_certainty"
        ].infer_objects()

        # remove columns that are not needed anymore (added for synchronisation)
        self.tracking_data.drop(
            ["goal_angle_home_team", "goal_angle_away_team"],
            axis=1,
            inplace=True,
        )

        self._is_synchronised = True

    def __eq__(self, other):
        if not isinstance(other, Match):
            return False
        for current_field in fields(self):
            if not _values_are_equal_(
                getattr(self, current_field.name), getattr(other, current_field.name)
            ):
                return False

        return True

    def copy(self) -> "Match":
        """Function to create a copy of the match object

        Returns:
            Match: copy of the match object
        """
        copied_kwargs = {
            f.name: _copy_value_(getattr(self, f.name)) for f in fields(self)
        }
        return Match(**copied_kwargs)

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
        name = name if isinstance(name, str) else self.name
        path = path if path is not None else os.getcwd()
        name = name.replace(":", "_")
        pickle_path = os.path.join(path, f"{name}.pickle")
        assert os.path.exists(path), f"Path {path} does not exist"
        with open(pickle_path, "wb") as f:
            pickle.dump(self, f)
        if verbose:
            print(f"Match saved to {os.path.join(path, name)}.pickle")
        LOGGER.info(f"Match saved to {os.path.join(path, name)}.pickle")


@logging_wrapper(__file__)
def check_inputs_match_object(match: Match):
    """Function to check if the inputs of the match object are correct

    Args:
        match (Match): match object
    """
    LOGGER.info("Checking the inputs of the match object")
    # tracking_data
    if not isinstance(match.tracking_data, pd.DataFrame):
        raise TypeError(
            f"tracking data should be a pandas df, not a {type(match.tracking_data)}"
        )

    if len(match.tracking_data) > 0:
        for col in ["frame", "ball_x", "ball_y", "datetime"]:
            if col not in match.tracking_data.columns.to_list():
                raise ValueError(
                    f"No {col} in tracking_data columns, this is manditory!"
                )

        # tracking_data_provider
        if not isinstance(match.tracking_data_provider, str):
            raise TypeError(
                "tracking data provider should be a string, not a "
                f"{type(match.tracking_data_provider)}"
            )

        # tracking data ball status
        ball_status_value_counts = match.tracking_data["ball_status"].value_counts()
        if len(ball_status_value_counts) != 2:
            message = (
                "ball status should be divided over dead and alive, "
                f"found value counts: {ball_status_value_counts}. Any further "
                "function that uses the ball status, such as the synchronisation, "
                "might not work anymore."
            )
            LOGGER.warning(message)
            warnings.warn(
                message=message,
                category=DataBallPyWarning,
            )
        else:
            frames_alive = ball_status_value_counts["alive"]
            minutes_alive = frames_alive / (match.frame_rate * 60)
            if minutes_alive < 45:
                message = (
                    f"The ball status is alive for {round(minutes_alive, 2)}"
                    " in the full match. ball status is uses for synchronisation "
                    "check the quality of the data before synchronising event and "
                    "tracking data."
                )
                LOGGER.warning(message)
                warnings.warn(
                    message=message,
                    category=DataBallPyWarning,
                )

        # check if the first frame is at (0, 0)
        ball_alive_mask = match.tracking_data["ball_status"] == "alive"
        if len(match.tracking_data.loc[ball_alive_mask]) > 0:
            first_frame = match.tracking_data.loc[
                ball_alive_mask, "ball_x"
            ].first_valid_index()
            if (
                not abs(match.tracking_data.loc[first_frame, "ball_x"]) < 7.0
                or not abs(match.tracking_data.loc[first_frame, "ball_y"]) < 5.0
            ):
                x_start = match.tracking_data.loc[first_frame, "ball_x"]
                y_start = match.tracking_data.loc[first_frame, "ball_y"]
                message = (
                    "The middle point of the pitch should be (0, 0), "
                    f"now the kick-off is at ({x_start}, {y_start}). "
                    "Either the recording has started too late or the ball_status "
                    "is not set to 'alive' in the beginning. Please check and "
                    " change the tracking data if desired."
                    "\n NOTE: The quality of the synchronisation of the tracking "
                    "and event data might be affected."
                )
                LOGGER.warning(message)
                warnings.warn(message=message, category=DataBallPyWarning)

        # check if there is a valid datetime object
        if not pd.api.types.is_datetime64_any_dtype(match.tracking_data["datetime"]):
            raise TypeError(
                "datetime column in tracking data should be a datetime dtype, not a "
                f"{type(match.tracking_data['datetime'])}"
            )
        # also make sure it is tz sensitive
        if match.tracking_data["datetime"].dt.tz is None:
            raise ValueError("datetime column in tracking data should have a timezone")

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
            "player_name",
        ]:
            if col not in match.event_data.columns.to_list():
                raise ValueError(f"{col} not in event data columns, this is manditory!")

        if not pd.api.types.is_datetime64_any_dtype(match.event_data["datetime"]):
            raise TypeError(
                "datetime column in event_data should be a datetime dtype, not a "
                f"{type(match.event_data['datetime'])}"
            )

        if match.event_data["datetime"].dt.tz is None:
            raise ValueError("datetime column in event_data should have a timezone")

        # event_data_provider
        if not isinstance(match.event_data_provider, str):
            raise TypeError(
                "event data provider should be a string, not a "
                f"{type(match.event_data_provider)}"
            )

    # pitch_dimensions
    if not isinstance(match.pitch_dimensions, (list, tuple)):
        raise TypeError(
            f"pitch_dimensions ({match.pitch_dimensions}) should be a "
            f"list, not a {type(match.pitch_dimensions)}"
        )
    if not len(match.pitch_dimensions) == 2:
        raise ValueError(
            "pitch_dimensions should contain, two values: a length and a width "
            f"of the pitch, current input is {match.pitch_dimensions}"
        )
    if not all([isinstance(x, (float, np.floating)) for x in match.pitch_dimensions]):
        raise TypeError(
            "Both values in pitch dimensions should by floats, current inputs "
            f"{[type(x) for x in match.pitch_dimensions]}"
        )
    if not 70 < match.pitch_dimensions[0] < 130:
        raise ValueError(
            "The length of the pitch should be between 70 and 130 meters, "
            f"current input is {match.pitch_dimensions[0]}"
        )

    if not 45 < match.pitch_dimensions[1] < 90:
        raise ValueError(
            "The width of the pitch should be between 45 and 90 meters, "
            f"current input is {match.pitch_dimensions[1]}"
        )

    # periods
    if not isinstance(match.periods, pd.DataFrame):
        raise TypeError(
            "periods_frames should be a pandas dataframe, not a "
            f"{type(match.periods)}"
        )
    if "period_id" not in match.periods.columns:
        raise ValueError("'period' should be one of the columns in period_frames")
    if any(
        [
            x not in match.periods["period_id"].value_counts().index
            for x in [1, 2, 3, 4, 5]
        ]
    ) or not all(match.periods["period_id"].value_counts() == 1):
        res = match.periods["period_id"]
        raise ValueError(
            "'period' column in period_frames should contain only the values "
            f"[1, 2, 3, 4, 5]. Now it's {res}"
        )

    for col in [col for col in match.periods if "datetime" in col]:
        if not pd.isnull(match.periods[col]).all() and match.periods[col].dt.tz is None:
            raise ValueError(f"{col} column in periods should have a timezone")

    # frame_rate
    if not pd.isnull(match.frame_rate) and not match.frame_rate == MISSING_INT:
        if not isinstance(match.frame_rate, (int, np.integer)):
            raise TypeError(
                f"frame_rate should be an integer, not a {type(match.frame_rate)}"
            )
        if match.frame_rate < 1:
            raise ValueError(f"frame_rate should be positive, not {match.frame_rate}")

    # team id's
    for team, team_id in zip(["home", "away"], [match.home_team_id, match.away_team_id]):
        if not isinstance(team_id, (int, np.integer)) and not isinstance(team_id, str):
            raise TypeError(
                f"{team} team id should be an integer or string, not a "
                f"{type(team_id)}"
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
            if not isinstance(score, (int, np.integer)):
                raise TypeError(
                    f"{team} team score should be an integer, not a {type(score)}"
                )
            if score < 0:
                raise ValueError(f"{team} team score should positive, not {score}")

    # team formations
    for team, form in zip(
        ["home", "away"], [match.home_formation, match.away_formation]
    ):
        if form is not None and not form == MISSING_INT:
            if not isinstance(form, str):
                raise TypeError(
                    f"{team} team formation should be a string, not a {type(form)}"
                )
            if len(form) > 5:
                raise ValueError(
                    f"{team} team formation should be of length 5 or smaller "
                    f"('1433'), not {len(form)}"
                )

    # team players
    for team, players in zip(["home", "away"], [match.home_players, match.away_players]):
        if not isinstance(players, pd.DataFrame):
            raise TypeError(
                f"{team} team players should be a pandas dataframe, not a "
                f"{type(players)}"
            )
        for col in ["id", "full_name", "shirt_num"]:
            if col not in players.columns:
                raise ValueError(
                    f"{team} team players should contain at least the column "
                    f"['id', 'full_name', 'shirt_num', 'position'], {col} is missing."
                )
        if (
            "position" in players.columns
            and not players["position"].isin(DATABALLPY_POSITIONS + [""]).all()
        ):
            raise ValueError(
                f"{team} team players should have a position that is in "
                f"{DATABALLPY_POSITIONS}, not {players['position'].unique()}"
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
            period = period_row["period_id"]
            home_x = [x + "_x" for x in match.home_players_column_ids()]
            away_x = [x + "_x" for x in match.away_players_column_ids()]
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
