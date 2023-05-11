import pandas as pd
import numpy as np
from databallpy.errors import DataBallPyError
from databallpy.warnings import DataBallPyWarning
from databallpy.features.velocity import get_velocity
import warnings 


def _quality_check_tracking_data(tracking_data: pd.DataFrame, framerate: float, periods: pd.DataFrame)-> None:
    print("Checking quality tracking data")
    _check_missing_ball_data(tracking_data, framerate)
    _check_ball_velocity(tracking_data, framerate)
    _check_player_velocity(tracking_data, framerate, periods)
    print("Check over")
    return 

def _check_missing_ball_data(tracking_data: pd.DataFrame, framerate: float)-> None:
    mask_ball_alive = tracking_data["ball_status"] == "alive"
    valid_frames = ~np.isnan(tracking_data.loc[mask_ball_alive, ["ball_x", "ball_y"]]).any(axis=1)
    sum_valid_frames = sum(valid_frames)
    n_total_frames = len(tracking_data.loc[mask_ball_alive])
    if sum_valid_frames < n_total_frames*0.99:
        warnings.warn(DataBallPyWarning("Ball data is not available for more than 1% of all frames"))

    max_nan_sequence = _max_sequence_invalid_frames(valid_frames)
    if max_nan_sequence >= 5*framerate:
        warnings.warn(DataBallPyWarning("There is a gap in the ball data for at least 5 seconds"))
    
    return 


def _check_ball_velocity(tracking_data: pd.DataFrame, framerate:float):
    initial_columns = tracking_data.columns
    input_columns = ["ball_x", "ball_y"]
    tracking_data = get_velocity(tracking_data, input_columns, framerate)
    velocity_ball = np.hypot(tracking_data["ball_x_v"], tracking_data["ball_y_v"])
    mask_ball_alive = tracking_data["ball_status"] == "alive"
    valid_frames = velocity_ball[mask_ball_alive][1:] < 50
    sum_valid_frames = sum(valid_frames)
    n_total_frames = len(valid_frames)
    #import pdb;pdb.set_trace()
    if sum_valid_frames < n_total_frames*0.99:
        warnings.warn(DataBallPyWarning(f"Ball velocity is unrealistic (> 50 m/s\N{SUPERSCRIPT TWO}) for more than 1% of all frames"))

    max_invalid_sequence = _max_sequence_invalid_frames(valid_frames)

    if max_invalid_sequence > 5*framerate:
        warnings.warn(DataBallPyWarning("There is a gap in the realistic ball data (velocity < 50 m/s\N{SUPERSCRIPT TWO}) for at least 5 seconds"))
    
    tracking_data = tracking_data[initial_columns]
    return

def _check_player_velocity(tracking_data: pd.DataFrame, framerate: float, periods: pd.DataFrame):
    initial_columns = tracking_data.columns
    player_columns = [x for x in tracking_data.columns if "home" in x or "away" in x]
    players = [x.replace("_x", "") for x in player_columns if "_x" in x]
    tracking_data = get_velocity(tracking_data, player_columns, framerate)

    mask_no_break = [False]*len(tracking_data)
    first_frame = periods.loc[0, "start_frame"]
    for _, row in periods.iterrows():
        if row["start_frame"] != -999:
            p_start = row["start_frame"] - first_frame
            p_end = row["end_frame"] - first_frame
            mask_no_break[p_start:p_end] = [True]*(p_end-p_start)

    percentages_valid_frames = []
    max_sequences_invalid_frames = []
    for player in players:
        velocity_player = np.hypot(tracking_data[f"{player}_x_v"], tracking_data[f"{player}_y_v"])
        velocity_player = velocity_player[mask_no_break][1:].reset_index(drop=True)
        valid_frames = velocity_player < 12.5
        sum_valid_frames = sum(valid_frames)
        player_specific_total_frames = valid_frames[::-1].idxmax() - valid_frames.idxmax()
        percentages_valid_frames.append(sum_valid_frames/player_specific_total_frames)
        max_sequences_invalid_frames.append(_max_sequence_invalid_frames(valid_frames, False))
    
    n_players_to_many_invalid_frames = sum([True for x in percentages_valid_frames if x<0.995])
    n_players_sequence_to_long = sum([True for x in max_sequences_invalid_frames if x>3*framerate])
    
    if n_players_to_many_invalid_frames > 0:
        warnings.warn(
            DataBallPyWarning(f"For {n_players_to_many_invalid_frames} players, the velocity is unrealistic for more than 0.5% of playing time")
        )

    if n_players_sequence_to_long > 0:
        warnings.warn(
            DataBallPyWarning(f"For {n_players_sequence_to_long} players, the velocity is unrealistic for at least 3 consecutive seconds")
        )
        

    tracking_data = tracking_data[initial_columns]
    return
    

def _max_sequence_invalid_frames(valid_frames: pd.Series, include_start_finish: bool = True) -> int:
    blocks = valid_frames.cumsum()
    sequences = (~valid_frames).groupby(blocks).sum()
    if not include_start_finish:
        sequences = sequences[1:-1]
    return sequences.max()
    

