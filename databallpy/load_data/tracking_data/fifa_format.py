from databallpy.load_data.metadata import Metadata
from typing import Tuple
from bs4 import BeautifulSoup
from tqdm import tqdm

import numpy as np
import pandas as pd
import bs4

def load_fifa_format_data(fifa_loc:str, meta_data_loc, verbose:bool) -> Tuple[pd.DataFrame, Metadata]:
    """

    Args:
        fifa_loc (str): _description_
        verbose (bool): _description_

    Returns:
        Tuple[pd.DataFrame, Metadata]: _description_
    """
    #tracking_data = _get_tracking_data(fifa_loc, verbose)
    meta_data = _get_meta_data(meta_data_loc, verbose)

    return meta_data

def _get_tracking_data(fifa_loc:str, verbose:bool) -> pd.DataFrame:
    """_summary_

    Args:
        fifa_loc (str): _description_
        verbose (bool): _description_

    Returns:
        pd.DataFrame: _description_
    """
    if verbose:
        print(f"Reading in {tracab_loc}", end="")

    file = open(tracab_loc, "r")
    lines = file.readlines()
    
    if verbose:
        print(" - Completed")

    file.close()
    

def _get_meta_data(meta_data_loc:str, verbose:bool) -> Metadata:
    """_summary_

    Args:
        meta_data_loc (str): _description_
        verbose (bool): _description_

    Returns:
        Metadata: _description_
    """
    
    file = open(meta_data_loc, "r", encoding="UTF-8").read()
    soup = BeautifulSoup(file, "xml")
    
    periods_dict = {"period": np.arange(1,6),
                    "start_frame": [np.nan]*5,
                    "end_frame": [np.nan]*5}
    periods = soup.find_all("Session")
    i = 0
    
    for period in periods:
        if period.SessionType.text == "Period":
            values = [int(x.text) for x in period.find_all("Value")]
            periods_dict["start_frame"][i] = values[0]
            periods_dict["end_frame"][i] = values[1]
            i += 1
    periods_frames = pd.DataFrame(periods_dict)

    home_team = soup.Teams.find_all("Team")[0]
    home_team_id = home_team.attrs["id"]
    home_team_name = home_team.Name.text
    home_team_player_data = soup.find_all("Player", {"teamId": home_team_id})
    home_players = _get_player_data(home_team_player_data)
    home_score = int(soup.LocalTeamScore.text)

    away_team = soup.Teams.find_all("Team")[1]
    away_team_id = away_team.attrs["id"]
    away_team_name = away_team.Name.text
    away_team_player_data = soup.find_all("Player", {"teamId": away_team_id})
    away_players = _get_player_data(away_team_player_data)
    away_score = int(soup.VisitingTeamScore.text)
    
    metadata = Metadata(
        match_id = int(soup.Session.attrs["id"]),
        pitch_dimensions = [float(soup.MatchParameters.FieldSize.Length.text),
                            float(soup.MatchParameters.FieldSize.Width.text)],
        match_start_datetime = np.datetime64(soup.Session.Start.text),
        periods_frames = periods_frames,

        frame_rate = int(soup.FrameRate.text),

        home_team_id = home_team_id,
        home_team_name = home_team_name,
        home_players = home_players,
        home_score = home_score, 
        home_formation = None,

        away_team_id = away_team_id,
        away_team_name = away_team_name,
        away_players = away_players,
        away_score = away_score,
        away_formation = None,
    )
    import pdb;pdb.set_trace()

def _get_player_data(team: bs4.element.Tag) -> pd.DataFrame:
    """Function that creates a df containing info on all players for a team
    Args:
        team (bs4.element.Tag): containing info no all players of a team
    Returns:
        pd.DataFrame: contains all player information for a team
    """

    player_dict = {
        "id": [],
        "full_name": [],
        "shirt_num": [],
        "player_type": [],
        "start_frame": [],
        "end_frame": []
    }
    for player in team:
        player_dict["id"].append(player["id"])
        player_dict["full_name"].append(player.Name.text)
        player_dict["shirt_num"].append(int(player.ShirtNumber.text))
        values = [x.text for x in player.find_all("Value")]
        player_dict["player_type"].append(values[0])
        player_dict["start_frame"].append(int(values[1]))
        player_dict["end_frame"].append(int(values[2]))
    df = pd.DataFrame(player_dict)

    return df

    