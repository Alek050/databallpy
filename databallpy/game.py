import json
import os
import warnings
from dataclasses import dataclass, fields
from functools import wraps

import numpy as np
import pandas as pd

from databallpy.schemas import (
    EventData,
    EventDataSchema,
    PlayersSchema,
    TrackingData,
    TrackingDataSchema,
)
from databallpy.utils.constants import DATABALLPY_POSITIONS, MISSING_INT
from databallpy.utils.errors import DataBallPyError
from databallpy.utils.game_utils import (
    player_column_id_to_full_name,
    player_id_to_column_id,
)
from databallpy.utils.logging import create_logger, logging_wrapper
from databallpy.utils.synchronise_tracking_and_event_data import (
    align_event_data_datetime,
    pre_compute_synchronisation_variables,
    synchronise_tracking_and_event_data,
)
from databallpy.utils.utils import (
    _copy_value_,
    _values_are_equal_,
)

LOGGER = create_logger(__file__)


def requires_tracking_data(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args[0].tracking_data) > 0:
            return func(*args, **kwargs)
        else:
            raise DataBallPyError(
                "No tracking data available, please load "
                "Game object with tracking data first."
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
                "Game object with event data first."
            )

    return wrapper


@dataclass
class Game:
    """
    This is the game class. It contains all information of the game and has
    some simple functions to easily obtain information about the game.

    Args:
        tracking_data (TrackingData): Tracking data of the game.
        event_data (EventData[pd.DataFrame]): Event data of the game.
        pitch_dimensions (Tuple): The size of the pitch in meters in x and y direction.
        periods (pd.DataFrame): The start and end idicators of all periods.
        home_team_id (int): The id of the home team.
        home_team_name (str): The name of the home team.
        home_players (pd.DataFrame): Information about the home players.
        home_score (int): Number of goals scored over the game by the home team.
        home_formation (str): Indication of the formation of the home team.
        away_team_id (int): The id of the away team.
        away_team_name (str): The name of the away team.
        away_players (pd.DataFrame): Information about the away players.
        away_score (int): Number of goals scored over the game by the away team.
        away_formation (str): Indication of the formation of the away team.
        country (str): The country where the game was played.
        shot_events (pd.DataFrame): A df with all th shot events.
        dribble_events (pd.DataFrame): A df with all the dribble events.
        pass_events (pd.DataFrame): A df with all the pass events.
        allow_synchronise_tracking_and_event_data (bool): If True, the tracking and
        event data can be synchronised. If False, it can not. Default = False.
    """

    tracking_data: TrackingData
    event_data: EventData
    pitch_dimensions: list[float, float]
    periods: pd.DataFrame
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
    shot_events: pd.DataFrame
    dribble_events: pd.DataFrame
    pass_events: pd.DataFrame
    allow_synchronise_tracking_and_event_data: bool = False
    # to save the preprocessing status
    _is_synchronised: bool = False
    # to indicate if the timestamps are precise or just the proposed timestamps of the
    # game (precisely 20:00 for start of game). This is important for the
    # synchronisation of the tracking and event data
    _tracking_timestamp_is_precise: bool = False
    _event_timestamp_is_precise: bool = False
    _periods_changed_playing_direction: list[int] = None
    _check_inputs_: bool = True

    def __repr__(self):
        return "databallpy.game.Game object: " + self.name

    def __post_init__(self):
        if self._check_inputs_:
            check_inputs_game_object(self)
        self._tracking_data_provider = self.tracking_data.provider
        self._frame_rate = self.tracking_data.frame_rate
        self._event_data_provider = self.event_data.provider

    @property
    def tracking_data_provider(self) -> str:
        warnings.warn(
            "`game.tracking_data_provider` is deprecated and will be removed in version 0.8.0. Please use `game.tracking_data.provider` instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self._tracking_data_provider

    @property
    def frame_rate(self) -> str:
        warnings.warn(
            "`game.frame_rate` is deprecated and will be removed in version 0.8.0. Please use `game.tracking_data.frame_rate` instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self._frame_rate

    @property
    def event_data_provider(self) -> str:
        warnings.warn(
            "`game.event_data_provider` is deprecated and will be removed in version 0.8.0. Please use `game.event_data.provider` instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self._event_data_provider

    @property
    def tracking_timestamp_is_precise(self) -> bool:
        """Function to check if the tracking timestamps are precise or not
        Timesamp is considered precise if the start of the game is not exactly
        at the start of the initial game time (e.g. 20:00:00), but at the
        actual start of the game (e.g. 20:00:03.2378).

        Returns:
            bool: True if the tracking timestamps are precise, False otherwise
        """
        return self._tracking_timestamp_is_precise

    @property
    def event_timestamp_is_precise(self) -> bool:
        """Function to check if the event timestamps are precise or not
        Timesamp is considered precise if the start of the game is not exactly
        at the start of the initial game time (e.g. 20:00:00), but at the
        actual start of the game (e.g. 20:00:03.2378).

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
        """Function to get the date of the game

        Returns:
            pd.Timestamp | None: The date of the game
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
    def name(self) -> str:
        """Function to get the name of the game

        Returns:
            str: The name of the game
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
        positions: list[str] = DATABALLPY_POSITIONS,
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
                player needs to have played during the game to be returned.
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
            players = players[players["position"].isin(positions)]

        if not (players["start_frame"] == MISSING_INT).all():
            players = players[
                (players["end_frame"] - players["start_frame"])
                / self.tracking_data.frame_rate
                / 60
                >= min_minutes_played
            ]
        col_ids = [
            f"home_{int(row.shirt_num)}"
            if row.id in self.home_players["id"].to_list()
            else f"away_{int(row.shirt_num)}"
            for row in players.itertuples(index=False)
        ]
        return [
            col_id for col_id in col_ids if f"{col_id}_x" in self.tracking_data.columns
        ]

    @requires_tracking_data
    def home_players_column_ids(self) -> list[str]:
        """Function to get all column ids of the tracking data that refer to information
        about the home team players

        Depreciation: This function is depreciated and will be removed in version
        0.7.0. Please use game.get_column_ids(team="home").

        Returns:
            list[str]: All column ids of the home team players
        """

        warnings.warn(
            "game.home_players_column_ids is depreciated and will be removed in "
            "version 0.7. Please use game.get_column_ids(team='home')",
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
        0.7.0. Please use game.get_column_ids(team="away").

        Returns:
            list[str]: All column ids of the away team players
        """

        warnings.warn(
            "game.away_players_column_ids is depreciated and will be removed in "
            "version 0.7. Please use game.get_column_ids(team='away')",
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

    @requires_event_data
    def get_event(self, event_id: int) -> pd.Series:
        """Function to get the event with the given event_id

        Args:
            event_id (int): The id of the event

        Raises:
            ValueError: if the event with the given event_id is not found in the game

        Returns:
            pd.Series: The event with the given event_id
        """

        if event_id in self.pass_events["event_id"].values:
            return self.pass_events[self.pass_events["event_id"] == event_id].iloc[0]
        elif event_id in self.shot_events["event_id"].values:
            return self.shot_events[self.shot_events["event_id"] == event_id].iloc[0]
        elif event_id in self.dribble_events["event_id"].values:
            return self.dribble_events[self.dribble_events["event_id"] == event_id].iloc[
                0
            ]
        elif event_id in self.event_data["event_id"].values:
            return self.event_data[self.event_data["event_id"] == event_id].iloc[0]
        else:
            raise ValueError(f"Event with id {event_id} not found in the game.")

    def get_frames(
        self, frames: int | list[int], playing_direction: str = "team_oriented"
    ) -> pd.DataFrame:
        """Function to get the frame of the game with the given frame

        Args:
            frames (int|list[int]): The frames of the game
            playing_direction (str, optional): The coordinate system of the frame.
                Defaults to "team_oriented", options are {team_oriented,
                possession_oriented}. For more info on the coordinate systems, see
                the documentation

        Returns:
            pd.DataFrame: The frame of the game with the given frames
        """
        if isinstance(frames, (int, np.integer)):
            frames = [frames]

        unrecognized_frames = [
            frame for frame in frames if frame not in self.tracking_data["frame"].values
        ]
        if len(unrecognized_frames) > 0:
            raise ValueError(f"Frame(s) {unrecognized_frames} not found in the game.")

        if playing_direction == "team_oriented":
            return self.tracking_data.loc[self.tracking_data["frame"].isin(frames)]
        elif playing_direction == "possession_oriented":
            # current coordinate system: home from left to right, away right to left
            suffixes = ("_x", "_y", "_vx", "_vy", "_ax", "_ay")
            cols_to_swap = [
                col for col in self.tracking_data.columns if col.endswith(suffixes)
            ]
            temp_td = self.tracking_data.loc[
                self.tracking_data["frame"].isin(frames)
            ].copy()
            temp_td.loc[
                self.tracking_data["team_possession"] == "away", cols_to_swap
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
            ValueError: if the event with the given event_id is not found in the game
            ValueError: if the event with the given event_id is not found in the
                tracking data

        Returns:
            pd.DataFrame: The frame of the game with the given event_id
        """
        if not self._is_synchronised:
            raise DataBallPyError(
                "Tracking and event data are not synchronised yet. Please run the"
                " synchronise_tracking_and_event_data() method first."
            )
        event_series = self.get_event(event_id)
        frame_id = self.tracking_data.loc[
            self.tracking_data["event_id"] == event_series.event_id, "frame"
        ].iloc[0]
        frame = self.get_frames(frame_id, playing_direction="team_oriented")
        if playing_direction == "team_oriented":
            return frame
        elif playing_direction == "possession_oriented":
            if sum(self.away_players["id"].values == event_series["player_id"]) > 0:
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

    @requires_tracking_data
    @requires_event_data
    def synchronise_tracking_and_event_data(
        self,
        n_batches: int | str = "smart",
        verbose: bool = True,
        offset: int = 1.0,
        optimize: bool = True,
        cost_functions: dict = {},
    ):
        """Function that synchronises tracking and event data using Needleman-Wunsch
           algorithmn. Based on: https://kwiatkowski.io/sync.soccer

        Args:
            game (Game): Game object
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
            optimize (bool, optional): Whether or not to optimize the algorithm. If
                errors or warnings are raised, try if setting to False works. Defaults
                to True.
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
            self.tracking_data, self.tracking_data.frame_rate, self.pitch_dimensions
        )
        # reduce standard error by aligning trakcing and event data on first event
        changed_event_data = align_event_data_datetime(
            self.event_data.copy(), self.tracking_data, offset=offset
        )
        tracking_info, event_info = synchronise_tracking_and_event_data(
            tracking_data=self.tracking_data,
            event_data=changed_event_data,
            home_players=self.home_players,
            away_players=self.away_players,
            cost_functions=cost_functions,
            n_batches=n_batches,
            optimize=optimize,
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
        if not isinstance(other, Game):
            return False
        for current_field in fields(self):
            if current_field.name == "_check_inputs_":
                continue
            if not _values_are_equal_(
                getattr(self, current_field.name), getattr(other, current_field.name)
            ):
                return False

        return True

    def copy(self) -> "Game":
        """Function to create a copy of the game object

        Returns:
            Game: copy of the game object
        """
        copied_kwargs = {
            f.name: _copy_value_(getattr(self, f.name)) for f in fields(self)
        }
        return Game(**copied_kwargs)

    def save_game(
        self,
        name: str = None,
        path: str = None,
        verbose: bool = True,
        allow_overwrite: bool = False,
    ) -> None:
        """Function to save the current game object. The path name will create a
        folder with different parquet and json files that stores all the information
        of the match.

        Args:
            name (str): name of the folder where the match will be saved,
            if not provided or not a string the name will be the name of the game.
            path (str): path to the directory where the folder will be saved. If
            not provided, the current working directory will be used.
            verbose (bool): if True, saved name will be printed
            allow_overwrite (bool): if True, the function will overwrite the
            existing folder with the same name.
        """
        name = name if isinstance(name, str) else self.name
        path = path if path is not None else os.getcwd()
        name = name.replace(":", "_")

        folder_path = os.path.join(path, name)

        if os.path.exists(folder_path) and not allow_overwrite:
            raise ValueError(
                f"Folder {folder_path} already exists, set allow_overwrite to True to overwrite"
            )

        os.makedirs(folder_path, exist_ok=True)
        self.tracking_data.to_parquet(os.path.join(folder_path, "tracking_data.parquet"))
        self.event_data.to_parquet(os.path.join(folder_path, "event_data.parquet"))
        self.periods.to_parquet(os.path.join(folder_path, "periods.parquet"))
        self.home_players.to_parquet(os.path.join(folder_path, "home_players.parquet"))
        self.away_players.to_parquet(os.path.join(folder_path, "away_players.parquet"))
        self.dribble_events.to_parquet(
            os.path.join(folder_path, "dribble_events.parquet")
        )
        self.shot_events.to_parquet(os.path.join(folder_path, "shot_events.parquet"))
        self.pass_events.to_parquet(os.path.join(folder_path, "pass_events.parquet"))

        metadata_info = {
            "event_data_provider": self.event_data.provider,
            "tracking_data_provider": self.tracking_data.provider,
            "tracking_data_frame_rate": self.tracking_data.frame_rate,
            "pitch_dimensions": self.pitch_dimensions,
            "home_team_id": self.home_team_id,
            "home_team_name": self.home_team_name,
            "home_score": self.home_score,
            "home_formation": self.home_formation,
            "away_team_id": self.away_team_id,
            "away_team_name": self.away_team_name,
            "away_score": self.away_score,
            "away_formation": self.away_formation,
            "country": self.country,
            "allow_synchronise_tracking_and_event_data": self.allow_synchronise_tracking_and_event_data,
            "_is_synchronised": self._is_synchronised,
            "_tracking_timestamp_is_precise": self._tracking_timestamp_is_precise,
            "_event_timestamp_is_precise": self._event_timestamp_is_precise,
            "_periods_changed_playing_direction": self._periods_changed_playing_direction,
        }
        with open(os.path.join(folder_path, "metadata.json"), "w") as f:
            json.dump(metadata_info, f)

        if verbose:
            print(f"Game saved in {folder_path}")


@logging_wrapper(__file__)
def check_inputs_game_object(game: Game):
    """Function to check if the inputs of the game object are correct

    Args:
        game (Game): game object
    """
    LOGGER.info("Checking the inputs of the game object")
    # tracking_data
    if not isinstance(game.tracking_data, TrackingData):
        raise TypeError(
            f"tracking data should be an instance of the TrackingData class, not a {type(game.tracking_data)}"
        )

    if len(game.tracking_data) > 0:
        TrackingDataSchema.validate(game.tracking_data)
        # tracking data provider
        if not isinstance(game.tracking_data.provider, str):
            raise TypeError(
                "tracking data provider should be a string, not a "
                f"{type(game.tracking_data.provider)}"
            )

    # event_data
    if not isinstance(game.event_data, EventData):
        raise TypeError(
            f"event data should be a EventData class, not a {type(game.event_data)}"
        )
    if len(game.event_data) > 0:
        EventDataSchema.validate(game.event_data)

    # pitch_dimensions
    if not isinstance(game.pitch_dimensions, (list, tuple)):
        raise TypeError(
            f"pitch_dimensions ({game.pitch_dimensions}) should be a "
            f"list, not a {type(game.pitch_dimensions)}"
        )
    if not len(game.pitch_dimensions) == 2:
        raise ValueError(
            "pitch_dimensions should contain, two values: a length and a width "
            f"of the pitch, current input is {game.pitch_dimensions}"
        )
    if not all([isinstance(x, (float, np.floating)) for x in game.pitch_dimensions]):
        raise TypeError(
            "Both values in pitch dimensions should by floats, current inputs "
            f"{[type(x) for x in game.pitch_dimensions]}"
        )
    if not 70 < game.pitch_dimensions[0] < 130:
        raise ValueError(
            "The length of the pitch should be between 70 and 130 meters, "
            f"current input is {game.pitch_dimensions[0]}"
        )

    if not 45 < game.pitch_dimensions[1] < 90:
        raise ValueError(
            "The width of the pitch should be between 45 and 90 meters, "
            f"current input is {game.pitch_dimensions[1]}"
        )

    # periods
    if not isinstance(game.periods, pd.DataFrame):
        raise TypeError(
            "periods_frames should be a pandas dataframe, not a " f"{type(game.periods)}"
        )
    if "period_id" not in game.periods.columns:
        raise ValueError("'period' should be one of the columns in period_frames")
    if any(
        [
            x not in game.periods["period_id"].value_counts().index
            for x in [1, 2, 3, 4, 5]
        ]
    ) or not all(game.periods["period_id"].value_counts() == 1):
        res = game.periods["period_id"]
        raise ValueError(
            "'period' column in period_frames should contain only the values "
            f"[1, 2, 3, 4, 5]. Now it's {res}"
        )

    for col in [col for col in game.periods if "datetime" in col]:
        if not pd.isnull(game.periods[col]).all() and game.periods[col].dt.tz is None:
            raise ValueError(f"{col} column in periods should have a timezone")

    # frame_rate
    if (
        not pd.isnull(game.tracking_data.frame_rate)
        and not game.tracking_data.frame_rate == MISSING_INT
    ):
        if not isinstance(game.tracking_data.frame_rate, (int, np.integer)):
            raise TypeError(
                f"frame_rate should be an integer, not a {type(game.tracking_data.frame_rate)}"
            )
        if game.tracking_data.frame_rate < 1:
            raise ValueError(
                f"frame_rate should be positive, not {game.tracking_data.frame_rate}"
            )

    # team id's
    for team, team_id in zip(["home", "away"], [game.home_team_id, game.away_team_id]):
        if not isinstance(team_id, (int, np.integer)) and not isinstance(team_id, str):
            raise TypeError(
                f"{team} team id should be an integer or string, not a "
                f"{type(team_id)}"
            )

    # team names
    for team, name in zip(["home", "away"], [game.home_team_name, game.away_team_name]):
        if not isinstance(name, str):
            raise TypeError(f"{team} team name should be a string, not a {type(name)}")

    # team scores
    for team, score in zip(["home", "away"], [game.home_score, game.away_score]):
        if not pd.isnull(score) and not score == MISSING_INT:
            if not isinstance(score, (int, np.integer)):
                raise TypeError(
                    f"{team} team score should be an integer, not a {type(score)}"
                )
            if score < 0:
                raise ValueError(f"{team} team score should positive, not {score}")

    # team formations
    for team, form in zip(["home", "away"], [game.home_formation, game.away_formation]):
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
    for players_df in [game.home_players, game.away_players]:
        if not isinstance(players_df, pd.DataFrame):
            raise TypeError(
                f"home and away players should be a pd df, not {type(players_df)}"
            )
        PlayersSchema.validate(players_df)

    # check for direction of play
    for _, period_row in game.periods.iterrows():
        if "start_frame" not in period_row.index:
            continue
        frame = period_row["start_frame"]
        if len(game.tracking_data[game.tracking_data["frame"] == frame].index) == 0:
            continue
        idx = game.tracking_data[game.tracking_data["frame"] == frame].index[0]
        period = period_row["period_id"]
        home_x = [x + "_x" for x in game.home_players_column_ids()]
        away_x = [x + "_x" for x in game.away_players_column_ids()]
        if game.tracking_data.loc[idx, home_x].mean() > 0:
            centroid_x = game.tracking_data.loc[idx, home_x].mean()
            raise DataBallPyError(
                "The home team should be represented as playing from left to "
                f"right the whole game. At the start of period {period} the x "
                f"centroid of the home team is {centroid_x}."
            )

        if game.tracking_data.loc[idx, away_x].mean() < 0:
            centroid_x = game.tracking_data.loc[idx, away_x].mean()
            raise DataBallPyError(
                "The away team should be represented as playingfrom right to "
                f"left the whole game. At the start  of period {period} the x "
                f"centroid ofthe away team is {centroid_x}."
            )

    # check databallpy_events
    databallpy_events = [game.dribble_events, game.shot_events, game.pass_events]
    for event_df, event_name in zip(
        databallpy_events,
        ["dribble", "shot", "pass"],
    ):
        if not isinstance(event_df, pd.DataFrame):
            raise TypeError(
                f"{event_name}_events should be a dataframe, not a " f"{type(event_df)}"
            )

    # country
    if not isinstance(game.country, str):
        raise TypeError(f"country should be a string, not a {type(game.country)}")
