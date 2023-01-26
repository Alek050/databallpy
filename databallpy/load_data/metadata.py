from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Metadata:
    match_id: int
    pitch_size_x_m: float
    pitch_size_y_m: float
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
