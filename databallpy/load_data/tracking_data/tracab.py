import numpy as np
import pandas as pd

def _get_lines_from_dat(tracab_loc: str, verbose: bool):   
    """_summary_

    Args:
        tracab_loc (_type_): _description_
        verbose (_type_): _description_

    Returns:
        _type_: _description_
    """
   
    if verbose:
        print(f"Reading in {tracab_loc}:")

    file = open(tracab_loc, "r")
    lines = file.readlines()
    import pdb;pdb.set_trace()
    return lines

def _add_player_data_to_dict(player, data, idx):
    
    team_id, _, shirt_num, x, y, speed = player.split(",")
        
    team_ids = {0: "home", 1: "away"}
    team = team_ids.get(int(team_id))
    if team is None: #player is unknown or referee
        return data

    if f"{team}_{shirt_num}_x" not in data.keys():
        data[f"{team}_{shirt_num}_x"] = [np.nan] * len(data["timestamp"])
        data[f"{team}_{shirt_num}_y"] = [np.nan] * len(data["timestamp"])
        data[f"{team}_{shirt_num}_speed"] = [np.nan] * len(data["timestamp"])

    data[f"{team}_{shirt_num}_x"][idx] = int(x)
    data[f"{team}_{shirt_num}_y"][idx] = int(y)
    data[f"{team}_{shirt_num}_speed"][idx] = float(speed)

    return data
        
def _add_ball_data_to_dict(ball_info, data, idx):
    
    x, y, z, speed, posession, status = ball_info.split(";")[0].split(",")[:6]
    data["ball_x"][idx] = int(x)
    data["ball_y"][idx] = int(y)
    data["ball_z"][idx] = int(z)
    data["ball_speed"][idx] = float(speed)
    data["ball_posession"][idx] = posession
    data["ball_status"][idx] = status

    return data

def _insert_missing_rows(df):
    missing = np.where(df["timestamp"].diff() > 1)[0]
    for start_missing in missing:
        n_missing = int(df["timestamp"].diff()[start_missing] - 1)
        start_timestamp = df.loc[start_missing, "timestamp"] - n_missing
        to_add_data = {"timestamp": list(np.arange(start_timestamp, start_timestamp+n_missing))}
        to_add_df = pd.DataFrame(to_add_data)
        df = pd.concat((df, to_add_df)).sort_values(by="timestamp")

    df.reset_index(drop=True, inplace=True)
    
    return df

def _get_tracking_data(tracab_loc, verbose):
    
    lines = _get_lines_from_dat(tracab_loc, verbose)
    size_lines = len(lines)

    data = {
        "timestamp": [np.nan] * size_lines,
        "ball_x": [np.nan] * size_lines,
        "ball_y": [np.nan] * size_lines,
        "ball_z": [np.nan] * size_lines,
        "ball_speed": [np.nan] * size_lines,
        "ball_status": [None] * size_lines,
        "ball_posession": [None] * size_lines,
    }

    for idx, line in enumerate(lines):
        timestamp, players_info, ball_info, _ = line.split(":")
        data["timestamp"][idx] = int(timestamp)

        players = players_info.split(";")[:-1]
        for player in players: 
            data = _add_player_data_to_dict(player, data, idx)

        data = _add_ball_data_to_dict(ball_info, data, idx)

    df=pd.DataFrame(data)

    df = _insert_missing_rows(df)
    
    return df


def load_tracking_data_tracab(tracab_loc, verbose=True):
    tracking_data = _get_tracking_data(tracab_loc, verbose)
    #meta_data = _get_meta_data(tracab_loc)

    return tracking_data
