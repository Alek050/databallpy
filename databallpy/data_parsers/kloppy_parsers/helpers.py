import warnings
from datetime import timedelta, timezone
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
import pandas as pd

from databallpy.data_parsers.tracking_data_parsers.utils import (
    _adjust_start_end_frames,
    _get_gametime,
    _insert_missing_rows,
)
from databallpy.schemas import EventData, TrackingData
from databallpy.utils.constants import MISSING_INT

if TYPE_CHECKING:
    from kloppy.domain import EventDataset, TrackingDataset


def _remove_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Remove timezone information from a timestamp or convert to UTC then remove timezone.

    If the timestamp has no timezone info, it's assumed to be UTC and timezone info is added
    then removed. If it has timezone info, it's converted to UTC then the timezone info is removed.
    This ensures consistent handling of timestamps regardless of their initial timezone state.

    Args:
        ts (pd.Timestamp): timestamp to process

    Returns:
        pd.Timestamp: timezone-naive timestamp in UTC
    """
    if ts is None:
        return
    return (
        ts.replace(tzinfo=timezone.utc)
        if ts.tzinfo is None
        else ts.astimezone(timezone.utc).replace(tzinfo=None)
    )


def _convert_datetime(
    kloppy_timestamp: timedelta,
    date: pd.Timestamp,
    period_id: int = None,
    verbose: bool = False,
) -> pd.Timestamp:
    """Convert a kloppy relative timestamp to an absolute datetime.

    Kloppy timestamps are relative to the start of each period. This function converts them to
    absolute timestamps by adding the game date and a period-based offset. The offset accounts
    for the duration of previous periods (60 min for period 2, 105 min for period 3, 120 min
    for period 4). Note that time between periods is not included in the offset.

    Args:
        kloppy_timestamp (timedelta): relative timestamp from kloppy (time since period start)
        date (pd.Timestamp): game date to use as base for absolute timestamp
        period_id (int, optional): period identifier (1, 2, 3, or 4). If None, no offset is applied.
            This is used when the date has already been adjusted for the period. Defaults to None.
        verbose (bool, optional): whether to print warnings. Defaults to False

    Returns:
        pd.Timestamp: absolute timestamp combining game date, kloppy timestamp, and period offset.
            Returns None if kloppy_timestamp is None. Falls back to Unix epoch ('1975-01-01') if date is None
    """

    # Note: this disregards the time in between periods
    timestamp_offset = (
        (
            pd.Timedelta(minutes=60)
            if period_id == 2
            else pd.Timedelta(minutes=105)
            if period_id == 3
            else pd.Timedelta(minutes=120)
            if period_id == 4
            else pd.Timedelta(0)
        )
        if period_id is not None
        else pd.Timedelta(0)
    )

    if kloppy_timestamp is None:
        return None

    date = _remove_utc(date)

    if date is not None:
        return kloppy_timestamp + date + timestamp_offset
    else:
        if verbose:
            warnings.warn(
                "Game date is None, using Unix epoch ('1975-01-01') as fall back date."
            )
        return kloppy_timestamp + pd.Timestamp("1975-01-01") + timestamp_offset


def players_from_kloppy(
    dataset: Union["EventDataset", "TrackingDataset"],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to get all information on players from a given kloppy dataset

    Args:
        dataset (kloppy.domain.TrackingDataset, kloppy.domain.EventDataset): Kloppy event or tracking dataset

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: All home and away player information in two DataFrames
    """
    from kloppy.domain import Ground

    home_players, away_players = [], []
    for player in dataset.metadata.teams[0].players + dataset.metadata.teams[1].players:
        p = {
            "id": player.player_id,
            "full_name": player.name,
            "shirt_num": player.jersey_no,
            "position": (
                player.starting_position.position_group.value[0]
                .lower()
                .replace("attacker", "forward")
                .replace("unknown", "unspecified")
            )
            if player.starting_position is not None
            else "unspecified",
            "start_frame": MISSING_INT,
            "end_frame": MISSING_INT,
            "starter": player.starting if player.starting is not None else False,
        }
        if player.team.ground == Ground.HOME:
            home_players.append(p)
        else:
            away_players.append(p)
    return pd.DataFrame(home_players), pd.DataFrame(away_players)


def periods_from_kloppy(
    event_dataset: "EventDataset" = None, tracking_dataset: "TrackingDataset" = None
) -> pd.DataFrame:
    """
    Function to get all information on period start and end times from kloppy datasets

    Args:
        tracking_dataset (kloppy.domain.TrackingDataset, optional): A Kloppy tracking dataset.
            Defaults to None.
        event_dataset (kloppy.domain.EventDataset, optional): A Kloppy event dataset.
            Defaults to None.

    Returns:
        periods (pd.DataFrame) All information about the periods
    """
    uses_tracking_data = False
    uses_event_data = False

    if tracking_dataset is not None:
        uses_tracking_data = True
    if event_dataset is not None:
        uses_event_data = True

    if event_dataset is None and tracking_dataset is None:
        raise ValueError(
            "At least one of event_dataset or tracking_dataset must be provided."
        )

    if event_dataset is not None and tracking_dataset is not None:
        if len(event_dataset.metadata.periods) != len(tracking_dataset.metadata.periods):
            min_periods = min(
                len(event_dataset.metadata.periods),
                len(tracking_dataset.metadata.periods),
            )
            event_dataset.metadata.periods = event_dataset.metadata.periods[
                0:min_periods
            ]
            tracking_dataset.metadata.periods = tracking_dataset.metadata.periods[
                0:min_periods
            ]
            warnings.warn(
                f"Number of periods in event and tracking dataset do not match. Using the minimum ({min_periods}) periods from both datasets."
            )

    game_date = (
        tracking_dataset.metadata.date
        if uses_tracking_data
        else event_dataset.metadata.date
    )
    periods = []

    # the Game.periods object must always have 5 enties
    for i in range(1, 6):
        period_info = {"period_id": i}

        period_records_td = (
            tracking_dataset.filter(lambda frame: frame.period.id == i)
            if uses_tracking_data
            else None
        )
        period_records_ed = (
            event_dataset.filter(lambda frame: frame.period.id == i)
            if uses_event_data
            else None
        )

        if uses_tracking_data:
            if len(period_records_td.records) == 0:
                start_frame = end_frame = MISSING_INT
                start_timestamp_td = end_timestamp_td = None
            else:
                period_td = tracking_dataset.metadata.periods[i - 1]
                if (
                    isinstance(period_td.start_timestamp, timedelta)
                    or period_td.start_timestamp is None
                ):
                    start_timestamp_td = _convert_datetime(
                        period_records_td[0].timestamp, game_date, period_id=i
                    )
                else:
                    start_timestamp_td = (
                        _remove_utc(period_td.start_timestamp)
                        if uses_tracking_data
                        else None
                    )

                if (
                    isinstance(period_td.end_timestamp, timedelta)
                    or period_td.end_timestamp is None
                ):
                    end_timestamp_td = _convert_datetime(
                        period_records_td[-1].timestamp, game_date, period_id=i
                    )
                else:
                    end_timestamp_td = (
                        _remove_utc(period_td.end_timestamp)
                        if uses_tracking_data
                        else None
                    )

                start_frame = period_records_td[0].frame_id
                end_frame = period_records_td[-1].frame_id

            period_info.update(
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_timestamp_td": start_timestamp_td,
                    "end_timestamp_td": end_timestamp_td,
                }
            )

        if uses_event_data:
            if (i - 1) >= len(event_dataset.metadata.periods):
                start_timestamp_ed = end_timestamp_ed = None
            else:
                period_ed = event_dataset.metadata.periods[i - 1]
                if (
                    isinstance(period_ed.start_timestamp, timedelta)
                    or period_ed.start_timestamp is None
                ):
                    start_timestamp_ed = _convert_datetime(
                        period_records_ed[0].timestamp, game_date, period_id=i
                    )
                else:
                    start_timestamp_ed = (
                        _remove_utc(period_ed.start_timestamp)
                        if uses_event_data
                        else None
                    )

                if (
                    isinstance(period_ed.end_timestamp, timedelta)
                    or period_ed.end_timestamp is None
                ):
                    end_timestamp_ed = _convert_datetime(
                        period_records_ed[-1].timestamp, game_date, period_id=i
                    )
                else:
                    end_timestamp_ed = (
                        _remove_utc(period_ed.end_timestamp) if uses_event_data else None
                    )

            period_info.update(
                {
                    "start_timestamp_ed": start_timestamp_ed,
                    "end_timestamp_ed": end_timestamp_ed,
                }
            )

        periods.append(period_info)

    return pd.DataFrame(periods)


def convert_kloppy_tracking_dataset(
    tracking_dataset: "TrackingDataset", periods: pd.DataFrame
) -> TrackingData:
    """
    Function to get all information of a game given kloppy dataset(s)

    Args:
        tracking_dataset (kloppy.domain.TrackingDataset, optional): a Kloppy tracking dataset.
        periods (pd.DataFrame): DataFrame containing information about the periods

    Returns:
        TrackingData: All tracking data in a TrackingData object
    """

    home_team, away_team = tracking_dataset.metadata.teams

    player_columns = {}
    for player in home_team.players + away_team.players:
        player_columns.update(
            {f"{player.player_id}_x": f"{player.team.ground}_{player.jersey_no}_x"}
        )
        player_columns.update(
            {f"{player.player_id}_y": f"{player.team.ground}_{player.jersey_no}_y"}
        )

    team_id_to_side = {home_team.team_id: "home", away_team.team_id: "away"}

    tracking_dataframe = (
        tracking_dataset.to_df(
            "frame_id",
            "period_id",
            "timestamp",
            "ball_state",
            "ball_owning_team_id",
            "ball_z",
            "*_x",
            "*_y",
            engine="pandas",
        )
        .assign(
            timestamp=lambda x: x.apply(
                lambda row: _convert_datetime(
                    row["timestamp"],
                    periods[periods["period_id"] == row["period_id"]][
                        "start_timestamp_td"
                    ].iloc[0],
                    period_id=None,
                    verbose=False,
                ),
                axis=1,
            ),
            team_possession=lambda x: x["ball_owning_team_id"].map(team_id_to_side),
        )
        .rename(
            columns={
                "frame_id": "frame",
                "ball_state": "ball_status",
                "timestamp": "datetime",
            }
            | player_columns
        )
        .drop(columns=["ball_owning_team_id"])
    )

    class SimplifiedMetada:
        def __init__(self, periods, frame_rate):
            self.periods_frames = periods
            self.frame_rate = frame_rate

    simplified_metadata = SimplifiedMetada(periods, tracking_dataset.metadata.frame_rate)
    tracking_dataframe = _insert_missing_rows(
        tracking_dataframe.reset_index(drop=True), "frame"
    )
    tracking_dataframe, simplified_metadata = _adjust_start_end_frames(
        tracking_dataframe, simplified_metadata
    )
    tracking_dataframe["gametime_td"] = _get_gametime(
        tracking_dataframe["frame"], tracking_dataframe["period_id"], simplified_metadata
    )

    return TrackingData(
        tracking_dataframe,
        provider=tracking_dataset.metadata.provider.value,
        frame_rate=tracking_dataset.metadata.frame_rate,
    )


def convert_kloppy_event_dataset(
    event_dataset: "EventDataset", periods: pd.DataFrame
) -> EventData:
    """
    Function to get all information of a game given kloppy dataset(s)

    Args:
        event_dataset (kloppy.domain.EventDataset, optional): A Kloppy event dataset.
        periods (pd.DataFrame): DataFrame containing information about the periods

    Returns:
        EventData: All event data in an EventData object
    """

    from kloppy.domain import (
        CarryResult,
        DuelResult,
        EventType,
        InterceptionResult,
        PassResult,
        ShotResult,
        TakeOnResult,
    )

    is_successful = [
        ShotResult.GOAL,
        ShotResult.OWN_GOAL,
        PassResult.COMPLETE,
        TakeOnResult.COMPLETE,
        CarryResult.COMPLETE,
        DuelResult.WON,
        InterceptionResult.SUCCESS,
    ]
    event_map = {
        EventType.PASS.value: "pass",
        EventType.SHOT.value: "shot",
        EventType.CARRY.value: "dribble",
        EventType.TAKE_ON.value: "dribble",
    }

    home_team, away_team = event_dataset.metadata.teams
    players = home_team.players + away_team.players

    player_id_to_name = {player.player_id: player.name for player in players}

    event_dataframe = (
        event_dataset.to_df(
            "period_id",
            "event_id",
            "timestamp",
            "player_id",
            "team_id",
            "coordinates_x",
            "coordinates_y",
            "event_type",
            "result",
            is_successful=lambda event: None
            if event.result is None
            else True
            if event.result in is_successful
            else False,
            minutes=lambda event: (int(event.timestamp.total_seconds()) % 3600 // 60)
            + (45 if event.period.id == 2 else 15 if event.period.id in [3, 4] else 0),
            seconds=lambda event: float(event.timestamp.total_seconds()) % 60,
            engine="pandas",
        )
        .sort_values(by=["period_id", "timestamp"], ascending=True)
        .reset_index(drop=True)
        .reset_index()
        .assign(
            timestamp=lambda x: x.apply(
                lambda row: _convert_datetime(
                    row["timestamp"],
                    periods[periods["period_id"] == row["period_id"]][
                        "start_timestamp_ed"
                    ].iloc[0],
                    period_id=None,
                    verbose=False,
                ),
                axis=1,
            ),
            databallpy_event=lambda x: np.where(
                x["result"] == ShotResult.OWN_GOAL,
                "own_goal",
                x["event_type"].map(event_map),
            ),
            player_name=lambda x: x["player_id"].map(player_id_to_name).astype(str),
            is_successful=lambda x: x["is_successful"].astype(pd.BooleanDtype()),
        )
        .rename(
            columns={
                "frame_id": "frame",
                "ball_state": "ball_status",
                "ball_owning_team_id": "team_possession",
                "timestamp": "datetime",
                "coordinates_x": "start_x",
                "coordinates_y": "start_y",
                "event_id": "original_event_id",
                "index": "event_id",
                "event_type": "original_event",
            }
        )
        .drop("result", axis=1)
    )

    # Because Pandera (when nullable=False) expects 'datetime' to have type datetime64[ns] and not datetime64[ns, UTC], remove UTC component
    event_dataframe["datetime"] = event_dataframe["datetime"].dt.tz_localize(None)

    return EventData(event_dataframe, provider=event_dataset.metadata.provider.value)
