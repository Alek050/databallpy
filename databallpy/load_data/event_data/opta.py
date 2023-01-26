from typing import Tuple, List
import pandas as pd
import bs4
import os
from dataclasses import dataclass

@dataclass
class EventMetaData:
    home_formation: str
    home_score: int
    home_team_id: int
    home_team_name: str
    home_players: pd.DataFrame
    away_formation: str
    away_score: int
    away_team_id: int
    away_team_name: str
    away_players: pd.DataFrame

def load_opta_event_data(f7_loc:str, f24_loc:str) -> Tuple[pd.Series, pd.DataFrame]:
    
    assert f7_loc[-4:] == ".xml", "f7 opta file should by of .xml format!"
    assert f24_loc[-4:] == ".xml", "f24 opta file should be of .xml format!"

    metadata = _load_metadata(f7_loc)
    event_data = _load_event_data(f24_loc)
    return metadata, event_data

def _load_metadata(f7_loc:str) -> pd.Series:
    file = open(f7_loc, "r").read()
    soup = bs4.BeautifulSoup(file, "xml")
    
    res_dict = {}

    # opta has a TeamData and Team attribute with information...
    team_datas = soup.find_all("TeamData")
    teams = soup.find_all("Team")
    for team_data, team in zip(team_datas, teams):
        
        # WIP: check for developing if opta is consistent, should not be in final version
        team_team_id = int(soup.find_all("Team")[1].attrs["uID"][1:])
        team_data_team_id = int(team_datas[0].attrs["TeamRef"][1:])
        assert team_team_id == team_data_team_id

        team_name = teams[0].findChildren("Name")[0].contents[0]
        team_info = _get_team_information(team_data.attrs, team_name)
        
        # get player info
        player_data = [player.attrs for player in team_data.findChildren("MatchPlayer")]
        player_names = {}
        for player in team.findChildren("Player"):
            player_id = int(player.attrs["uID"][1:])
            first_name = teams[0].findChildren("Player")[0].contents[1].contents[1].contents[0]
            last_name = teams[0].findChildren("Player")[0].contents[1].contents[3].contents[0]
            player_names[str(player_id)] = f"{first_name} {last_name}"
        
        player_info = _get_player_info(player_data, player_names)
        
    
    return pd.Series(res_dict)

def _get_team_information(team:dict, team_name:str) -> dict:p
    # get team info
    team_info = {}
    team_info["team_name"] = team_name
    team_info["side"] = team["Side"].lower()
    team_info["formation"] = team["Formation"]
    team_info["score"] = int(team["Score"])
    team_info["team_id"] = int(team["TeamRef"][1:])

    return team_info

def _get_player_info(player_data:dict, player_names:dict) -> pd.DataFrame:
    import pdb; pdb.set_trace()

   


    


def _load_event_data(f24_loc:str) -> pd.DataFrame:

    pass