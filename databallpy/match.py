from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

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

    def synchronise_tracking_and_event_data(self, n_batches_per_half: int = 9):
        """Function that synchronises tracking and event data using Needleman-Wunsch
           algorithmn. Based on: https://kwiatkowski.io/sync.soccer

        Args:
            n_batches_per_half (int): the number of batches that are created per half.
            A higher number of batches reduces the time the code takes to load, but
            reduces the accuracy for events close to the splits.
            Default = 9

        Currently works for the following events:
            'pass', 'aerial', 'interception', 'ball recovery', 'dispossessed', 'tackle',
            'take on', 'clearance', 'blocked pass', 'offside pass', 'attempt saved',
            'save', 'foul', 'miss', 'challenge', 'goal'

        Returns:
            Match: The match class with synchronised tracking and event data

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
            start_datetime_period[row["period"]] = row["start_time_td"].to_datetime64()
            start_frame_period[row["period"]] = row["start_frame"]

        tracking_data["datetime"] = [
            start_datetime_period[int(p)]
            + np.timedelta64(
                int((x - start_frame_period[p]) / self.frame_rate * 1000), "ms"
            )
            if p > 0
            else np.timedelta64("NaT")
            for x, p in zip(tracking_data["timestamp"], tracking_data["period"])
        ]

        tracking_data["event"] = np.nan
        tracking_data["event_id"] = np.nan
        event_data["tracking_frame"] = np.nan

        mask_events_to_sync = event_data["event"].isin(events_to_sync)
        event_data = event_data[mask_events_to_sync]

        periods_played = self.periods[self.periods["start_frame"] >= 0]["period"].values
        for p in periods_played:
            # create batches to loop over
            start_batch_frame = self.periods.loc[
                self.periods["period"] == p, "start_frame"
            ].iloc[0]
            start_batch_datetime = start_datetime_period[p] + np.timedelta64(
                int(
                    (start_batch_frame - start_frame_period[p]) / self.frame_rate * 1000
                ),
                "ms",
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
                + np.timedelta64(
                    int((x - start_frame_period[int(p)]) / self.frame_rate * 1000), "ms"
                )
                for x in end_batches_frames
            ]

            print(f"Syncing period {p}...")
            for end_batch_frame, end_batch_datetime in tqdm(
                zip(end_batches_frames, end_batches_datetime),
                total=len(end_batches_frames),
            ):
                tracking_batch = tracking_data[
                    (tracking_data["timestamp"] <= end_batch_frame)
                    & (tracking_data["timestamp"] >= start_batch_frame)
                ].reset_index(drop=False)
                event_batch = event_data[
                    (event_data["datetime"] >= start_batch_datetime)
                    & (event_data["datetime"] <= end_batch_datetime)
                ].reset_index()

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

                start_batch_frame = tracking_data.iloc[tracking_frame]["timestamp"]
                start_batch_datetime = (
                    event_batch[event_batch["event_id"] == event_id]["datetime"]
                    .iloc[0]
                    .to_datetime64()
                )
        tracking_data.drop("datetime", axis=1, inplace=True)
        self.tracking_data = tracking_data
        self.event_data = event_data

        return self

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
    tracking_batch["datetime"] = tracking_batch["datetime"].astype("datetime64[ns]")

    for i, event in event_batch.iterrows():
        time_diff = (tracking_batch["datetime"] - event["datetime"]) / np.timedelta64(
            1, "s"
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

    den = np.nanmax(np.nanmin(sim_mat, axis=1))  # scale similarity scores
    sim_mat[np.isnan(sim_mat)] = np.nanmax(
        sim_mat
    )  # replace nan values with highest value
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
