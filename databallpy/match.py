import datetime as dt
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from databallpy import DataBallPyError
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
                    f"frame_rate should be positive, not {self.frame_rate}"
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
        if (
            not abs(
                self.tracking_data["ball_x"].min() + self.tracking_data["ball_x"].max()
            )
            < 5.0
        ):
            max_x = self.tracking_data["ball_x"].max()
            min_x = self.tracking_data["ball_x"].min()
            raise DataBallPyError(
                f"The middle point of the pitch should be (0, 0),\
                                now the min x = {min_x} and the max x = {max_x}"
            )

        if (
            not abs(
                self.tracking_data["ball_y"].min() + self.tracking_data["ball_y"].max()
            )
            < 5.0
        ):
            max_y = self.tracking_data["ball_y"].max()
            min_y = self.tracking_data["ball_y"].min()
            raise DataBallPyError(
                f"The middle point of the pitch should be (0, 0),\
                                now th min y = {min_y} and the max y = {max_y}"
            )

        # check for direction of play
        for _, period_row in self.periods.iterrows():
            frame = period_row["start_frame"]
            if (
                len(self.tracking_data[self.tracking_data["timestamp"] == frame].index)
                == 0
            ):
                continue
            idx = self.tracking_data[self.tracking_data["timestamp"] == frame].index[0]
            period = period_row["period"]
            home_x = [x for x in self.home_players_column_ids if "_x" in x]
            away_x = [x for x in self.away_players_column_ids if "_x" in x]
            if self.tracking_data.loc[idx, home_x].mean() > 0:
                centroid_x = self.tracking_data.loc[idx, home_x].mean()
                raise DataBallPyError(
                    f"The home team should be represented as playing\
from left to right the whole match. At the start of period {period} the x centroid of \
the home team is {centroid_x}."
                )

            if self.tracking_data.loc[idx, away_x].mean() < 0:
                centroid_x = self.tracking_data.loc[idx, away_x].mean()
                raise DataBallPyError(
                    f"The away team should be represented as playing\
from right to left the whole match. At the start  of period {period} the x centroid of \
the away team is {centroid_x}."
                )

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

    def synchronise_tracking_and_event_data(
        self, n_batches_per_half: int = 100, verbose: bool = True
    ):
        """Function that synchronises tracking and event data using Needleman-Wunsch
           algorithmn. Based on: https://kwiatkowski.io/sync.soccer

        Args:
            n_batches_per_half (int): the number of batches that are created per half.
            A higher number of batches reduces the time the code takes to load, but
            reduces the accuracy for events close to the splits.
            Default = 100
            verbose (bool, optional): Wheter or not to print info about the progress
            in the terminal. Defaults to True.

        Currently works for the following events:
            'pass', 'aerial', 'interception', 'ball recovery', 'dispossessed', 'tackle',
            'take on', 'clearance', 'blocked pass', 'offside pass', 'attempt saved',
            'save', 'foul', 'miss', 'challenge', 'goal'

        """
        events_to_sync = [
            "pass",
            "aerial",
            "interception",
            "ball recovery",
            "dispossessed",
            "tackle",
            "take on",
            "clearance",
            "blocked pass",
            "offside pass",
            "attempt saved",
            "save",
            "foul",
            "miss",
            "challenge",
            "goal",
        ]

        tracking_data = self.tracking_data
        event_data = self.event_data

        start_datetime_period = {}
        start_frame_period = {}
        for _, row in self.periods.iterrows():
            start_datetime_period[row["period"]] = row["start_time_td"]
            start_frame_period[row["period"]] = row["start_frame"]

        tracking_data["datetime"] = [
            start_datetime_period[int(p)]
            + dt.timedelta(
                milliseconds=int((x - start_frame_period[p]) / self.frame_rate * 1000)
            )
            if p > 0
            else pd.to_datetime("NaT")
            for x, p in zip(tracking_data["timestamp"], tracking_data["period"])
        ]

        tracking_data["event"] = np.nan
        tracking_data["event_id"] = np.nan
        event_data["tracking_frame"] = np.nan

        mask_events_to_sync = event_data["event"].isin(events_to_sync)
        event_data = event_data[mask_events_to_sync]

        periods_played = self.periods[self.periods["start_frame"] > 0]["period"].values

        for p in periods_played:
            # create batches to loop over
            start_batch_frame = self.periods.loc[
                self.periods["period"] == p, "start_frame"
            ].iloc[0]
            start_batch_datetime = start_datetime_period[p] + dt.timedelta(
                milliseconds=int(
                    (start_batch_frame - start_frame_period[p]) / self.frame_rate * 1000
                )
            )
            delta = (
                self.periods.loc[self.periods["period"] == p, "end_frame"].iloc[0]
                - start_batch_frame
            )
            end_batches_frames = np.floor(
                np.arange(
                    delta / n_batches_per_half,
                    delta + delta / n_batches_per_half,
                    delta / n_batches_per_half,
                )
                + start_batch_frame
            )
            end_batches_datetime = [
                start_datetime_period[int(p)]
                + dt.timedelta(
                    milliseconds=int(
                        (x - start_frame_period[int(p)]) / self.frame_rate * 1000
                    )
                )
                for x in end_batches_frames
            ]

            tracking_data_period = tracking_data[tracking_data["period"] == p]
            event_data_period = event_data[event_data["period_id"] == p].copy()
            start_events = ["pass", "miss", "goal"]
            datetime_first_event = event_data_period[
                event_data_period["event"].isin(start_events)
            ].iloc[0]["datetime"]
            datetime_first_tracking_frame = tracking_data_period.iloc[0]["datetime"]
            diff_datetime = datetime_first_event - datetime_first_tracking_frame
            event_data_period["datetime"] -= diff_datetime

            if verbose:
                print(f"Syncing period {p}...")
                zip_batches = tqdm(
                    zip(end_batches_frames, end_batches_datetime),
                    total=len(end_batches_frames),
                )
            else:
                zip_batches = zip(end_batches_frames, end_batches_datetime)
            for end_batch_frame, end_batch_datetime in zip_batches:

                tracking_batch = tracking_data_period[
                    (tracking_data_period["timestamp"] <= end_batch_frame)
                    & (tracking_data_period["timestamp"] >= start_batch_frame)
                ].reset_index(drop=False)
                event_batch = event_data_period[
                    (event_data_period["datetime"] >= start_batch_datetime)
                    & (event_data_period["datetime"] <= end_batch_datetime)
                ].reset_index(drop=False)

                sim_mat = _create_sim_mat(tracking_batch, event_batch, self)
                event_frame_dict = _needleman_wunsch(sim_mat)

                for event, frame in event_frame_dict.items():
                    event_id = int(event_batch.loc[event, "event_id"])
                    event_type = event_batch.loc[event, "event"]
                    event_index = int(event_batch.loc[event, "index"])
                    tracking_frame = int(tracking_batch.loc[frame, "index"])
                    tracking_data.loc[tracking_frame, "event"] = event_type
                    tracking_data.loc[tracking_frame, "event_id"] = event_id
                    event_data.loc[event_index, "tracking_frame"] = tracking_frame

                start_batch_frame = tracking_data_period.loc[
                    tracking_frame, "timestamp"
                ]
                start_batch_datetime = event_data_period[
                    event_data_period["event_id"] == event_id
                ]["datetime"].iloc[0]
        tracking_data.drop("datetime", axis=1, inplace=True)
        self.tracking_data = tracking_data
        self.event_data = event_data

        return

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
        (Match): a Match object with all information of the match.
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


def _create_sim_mat(
    tracking_batch: pd.DataFrame, event_batch: pd.DataFrame, match: Match
) -> np.ndarray:
    """Function that creates similarity matrix between every frame and event in batch

    Args:
        tracking_batch (pd.DataFrame): batch of tracking data
        event_batch (pd.DataFrame): batch of event data
        match (Match): Match class containing full_name_to_player_column_id function

    Returns:
        np.ndarray: array containing similarity scores between every frame and events,
                    size is #frames, #events
    """
    sim_mat = np.zeros((len(tracking_batch), len(event_batch)))
    tracking_batch["datetime"] = tracking_batch["datetime"]

    for i, event in event_batch.iterrows():
        time_diff = (tracking_batch["datetime"] - event["datetime"]) / dt.timedelta(
            seconds=1
        )
        ball_loc_diff = np.hypot(
            tracking_batch["ball_x"] - event["start_x"],
            tracking_batch["ball_y"] - event["start_y"],
        )

        if not np.isnan(event["player_id"]):
            column_id_player = match.player_id_to_column_id(
                player_id=event["player_id"]
            )
            player_ball_diff = np.hypot(
                tracking_batch["ball_x"] - tracking_batch[f"{column_id_player}_x"],
                tracking_batch["ball_y"] - tracking_batch[f"{column_id_player}_y"],
            )
        else:
            player_ball_diff = 0
        # similarity function from: https://kwiatkowski.io/sync.soccer
        sim_mat[:, i] = np.abs(time_diff) + ball_loc_diff / 5 + player_ball_diff

    sim_mat[np.isnan(sim_mat)] = np.nanmax(
        sim_mat
    )  # replace nan values with highest value
    den = np.nanmax(np.nanmin(sim_mat, axis=1))  # scale similarity scores
    sim_mat = np.exp(-sim_mat / den)

    return sim_mat


def _needleman_wunsch(
    sim_mat: np.ndarray, gap_event: int = -1, gap_frame: int = 1
) -> dict:
    """
    Function that calculates the optimal alignment between events and frames
    given similarity scores between all frames and events
    Based on: https://gist.github.com/slowkow/06c6dba9180d013dfd82bec217d22eb5

    Args:
        sim_mat (np.ndarray): matrix with similarity between every frame and event
        gap_event (int): penalty for leaving an event unassigned to a frame
                         (not allowed), defaults to -1
        gap_frame (int): penalty for leaving a frame unassigned to a penalty
                         (very common), defaults to 1

    Returns:
       event_frame_dict (dict): dictionary with events as keys and frames as values
    """
    n_frames, n_events = np.shape(sim_mat)

    F = np.zeros((n_frames + 1, n_events + 1))
    F[:, 0] = np.linspace(0, n_frames * gap_frame, n_frames + 1)
    F[0, :] = np.linspace(0, n_events * gap_event, n_events + 1)

    # Pointer matrix
    P = np.zeros((n_frames + 1, n_events + 1))
    P[:, 0] = 3
    P[0, :] = 4

    t = np.zeros(3)
    for i in range(n_frames):
        for j in range(n_events):
            t[0] = F[i, j] + sim_mat[i, j]
            t[1] = F[i, j + 1] + gap_frame  # top + gap frame
            t[2] = F[i + 1, j] + gap_event  # left + gap event
            tmax = np.max(t)
            F[i + 1, j + 1] = tmax

            if t[0] == tmax:  # match
                P[i + 1, j + 1] += 2
            if t[1] == tmax:  # frame unassigned
                P[i + 1, j + 1] += 3
            if t[2] == tmax:  # event unassigned
                P[i + 1, j + 1] += 4

    # Trace through an optimal alignment.
    i = n_frames
    j = n_events
    frames = []
    events = []
    while i > 0 or j > 0:
        if P[i, j] in [2, 5, 6, 9]:  # 2 was added, match
            frames.append(i)
            events.append(j)
            i -= 1
            j -= 1
        elif P[i, j] in [3, 5, 7, 9]:  # 3 was added, frame unassigned
            frames.append(i)
            events.append(0)
            i -= 1
        elif P[i, j] in [4, 6, 7, 9]:  # 4 was added, event unassigned
            raise ValueError(
                "An event was left unassigned, check your gap penalty values"
            )

    frames = frames[::-1]
    events = events[::-1]

    idx_events = [idx for idx, i in enumerate(events) if i > 0]
    event_frame_dict = {}
    for i in idx_events:
        event_frame_dict[events[i] - 1] = frames[i] - 1

    return event_frame_dict


def get_open_match(provider: str = "metrica", verbose: bool = True) -> Match:
    """Function to load a match object from an open datasource

    Args:
        provider (str, optional): What provider to get the open data from.
        Defaults to "metrica".
        verbose (bool, optional): Whether or not to print info about progress
        in the terminal, Defaults to True.

    Returns:
        Match: All information about the match
    """
    assert provider in ["metrica"]

    if provider == "metrica":
        tracking_data, metadata = load_metrica_open_tracking_data(verbose=verbose)
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
