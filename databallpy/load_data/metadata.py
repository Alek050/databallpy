from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Metadata:
    match_id: int
    pitch_dimensions: list
    match_start_datetime: np.datetime64
    periods_frames: pd.DataFrame

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

    def __eq__(self, other):
        result = all(
            [
                self.match_id == other.match_id,
                all(
                    [
                        s == o
                        for s, o in zip(self.pitch_dimensions, other.pitch_dimensions)
                    ]
                ),
                self.match_start_datetime == other.match_start_datetime,
                self.periods_frames.equals(other.periods_frames),
                self.frame_rate == other.frame_rate
                if not pd.isnull(self.frame_rate)
                else pd.isnull(other.frame_rate),
                self.home_team_id == other.home_team_id,
                self.home_team_name == other.home_team_name,
                self.home_players.equals(other.home_players),
                self.home_score == other.home_score,
                self.home_formation == other.home_formation,
                self.away_team_id == other.away_team_id,
                self.away_team_name == other.away_team_name,
                self.away_players.equals(other.away_players),
                self.away_score == other.away_score,
                self.away_formation == other.away_formation,
            ]
        )
        return result
