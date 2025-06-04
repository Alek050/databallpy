import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

from ...schemas import EventData, TrackingData


def _convert_datetime(kloppy_timestamp: timedelta, game_date, verbose: bool = True) -> pd.Timestamp:
    if game_date is not None:
        return kloppy_timestamp + game_date
    else:
        if verbose:
            warnings.warn("Game date is None, using Unix epoch ('1970-01-01') as fall back date.")
        return kloppy_timestamp + pd.Timestamp('1970-01-01')

def players_from_kloppy(tracking_dataset):
    from kloppy.domain import Ground
    
    def __top_level_position_label(starting_position):
        if starting_position is not None:
            if starting_position.parent is not None:
                if starting_position.parent.parent is not None:
                    return str(starting_position.parent.parent).lower()
                else:
                    return str(starting_position.parent).lower()
                
        return "unspecified"
    
    home_players, away_players = [], []
    for player in tracking_dataset.metadata.teams[0].players + tracking_dataset.metadata.teams[1].players:
        p = {
            "id": player.player_id,
            "full_name": player.name,
            "shirt_num": player.jersey_no,
            "position": __top_level_position_label(player.starting_position).replace("attacker", "forward"),
            "start_frame": -999,
            "end_frame": -999,
            "starter": player.starting,
        }
        if player.team.ground == Ground.HOME:
            home_players.append(p)
        else:
            away_players.append(p)
    return pd.DataFrame(home_players), pd.DataFrame(away_players)

def periods_from_kloppy(event_dataset, tracking_dataset) -> pd.DataFrame:    
    assert len(event_dataset.metadata.periods) == len(tracking_dataset.metadata.periods)
    
    game_date = tracking_dataset.metadata.date
    periods = []
    for i in range(5):
        period_records_td = tracking_dataset.filter(lambda frame: frame.period.id == i + 1)
        period_records_ed = event_dataset.filter(lambda frame: frame.period.id == i + 1)
        
        if len(period_records_td.records) == 0:
            periods.append({
                "period_id": i + 1,
                "start_frame": -999,
                "end_frame": -999,
                "start_timestamp_td": None,
                "end_timestamp_td": None,
                "start_timestamp_ed": None,
                "end_timestamp_ed": None,
            })
        else:
            periods.append({
                "period_id": i + 1,
                "start_frame": period_records_td[0].frame_id,
                "end_frame": period_records_td[-1].frame_id,
                "start_timestamp_td": _convert_datetime(period_records_td[0].timestamp, game_date, verbose=True if i == 0 else False),
                "end_timestamp_td": _convert_datetime(period_records_td[-1].timestamp, game_date, verbose=False),
                "start_timestamp_ed": _convert_datetime(period_records_ed[0].timestamp, game_date, verbose=False),
                "end_timestamp_ed": _convert_datetime(period_records_ed[-1].timestamp, game_date, verbose=False),
            })
            
    return pd.DataFrame(periods)

def convert_kloppy_tracking_dataset(tracking_dataset: "TrackingDataset") -> TrackingData:
    home_team, away_team = tracking_dataset.metadata.teams

    player_columns = {}
    for player in home_team.players + away_team.players:
        player_columns.update({f"{player.player_id}_x": f"{player.team.ground}_{player.jersey_no}_x"})
        player_columns.update({f"{player.player_id}_y": f"{player.team.ground}_{player.jersey_no}_y"})
        
    tracking_dataframe = (
        tracking_dataset
        .to_df(
            "frame_id",
            "period_id",
            "timestamp",
            "ball_state",
            "ball_owning_team_id",
            "ball_z",
            "*_x",
            "*_y",
            engine="pandas"
        ) 
        .assign(
            timestamp=lambda x: x['timestamp'].apply(lambda ts: _convert_datetime(ts, tracking_dataset.metadata.date, verbose=False)),
        )    
        .rename(columns={
            "frame_id": "frame",
            "ball_state": "ball_status",
            "ball_owning_team_id": "team_possession",
            "timestamp": "datetime",
        } | player_columns
        )
    )    

    return TrackingData(
        tracking_dataframe,
        provider=tracking_dataset.metadata.provider.value,
        frame_rate=tracking_dataset.metadata.frame_rate,
    )

def convert_kloppy_event_dataset(event_dataset: "EventDataset") -> EventData:
    from kloppy.domain import (
        CarryResult,
        DuelResult,
        EventType,
        InterceptionResult,
        PassResult,
        ShotResult,
        TakeOnResult,
    )

    IS_SUCCESSFUL = [
        ShotResult.GOAL, 
        ShotResult.OWN_GOAL,
        PassResult.COMPLETE,
        TakeOnResult.COMPLETE,
        CarryResult.COMPLETE,
        DuelResult.WON,
        InterceptionResult.SUCCESS
    ]
    EVENT_MAP = {
        EventType.PASS.value: "pass",
        EventType.SHOT.value: "shot",
        EventType.CARRY.value: "dribble",
        EventType.TAKE_ON.value: "dribble"
    }

    event_data = (
        event_dataset
        .to_df(
            "period_id",
            "event_id",
            "timestamp",
            "player_id",
            "player",
            "team_id",
            "coordinates_x",
            "coordinates_y",
            "event_type",
            "result",
            is_successful=lambda event: None if event.result is None else True if event.result in IS_SUCCESSFUL else False,
            minutes=lambda event: (int(event.timestamp.total_seconds()) % 3600 // 60) + (45 if event.period.id == 2 else 15 if event.period.id in [3, 4] else 0),
            seconds=lambda event: float(event.timestamp.total_seconds()) % 60,    
            engine="pandas"
        )
        .sort_values(by=['period_id','timestamp'], ascending=True)
        .reset_index(drop=True)
        .reset_index()
        .assign(
            timestamp=lambda x: x['timestamp'].apply(lambda ts: _convert_datetime(ts, event_dataset.metadata.date, verbose=False)),
            databallpy_event = lambda x: np.where(
                x['result'] == ShotResult.OWN_GOAL,
                'own_goal',
                x['event_type'].map(EVENT_MAP)
            ),
            player=lambda x: str(x['player']),
            is_successful=lambda x: x['is_successful'].astype(pd.BooleanDtype()),
        )   
        .rename(columns={
            "frame_id": "frame",
            "ball_state": "ball_status",
            "ball_owning_team_id": "team_possession",
            "timestamp": "datetime",
            "coordinates_x": "start_x",
            "coordinates_y": "start_y",
            "event_id": "original_event_id",
            "index": "event_id",
            "event_type": "original_event",
            "player": "player_name",
        })
        .drop("result", axis=1)
    )

    return EventData(
        event_data, provider=event_dataset.metadata.provider.value
    )
    
