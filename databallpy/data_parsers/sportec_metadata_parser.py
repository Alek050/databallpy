import chardet
import pandas as pd
from bs4 import BeautifulSoup

from databallpy.data_parsers.metadata import Metadata
from databallpy.utils.constants import MISSING_INT

SPORTEC_BASE_URL = "https://springernature.figshare.com/ndownloader/files"
FILE_ID_MAP = {
    "J03WPY": {"metadata": 51643487, "event_data": 51643505, "tracking_data": 51643526},
    "J03WN1": {"metadata": 51643472, "event_data": 51643496, "tracking_data": 51643517},
    "J03WMX": {"metadata": 51643475, "event_data": 51643493, "tracking_data": 51643514},
    "J03WOH": {"metadata": 51643478, "event_data": 51643499, "tracking_data": 51643520},
    "J03WQQ": {"metadata": 51643484, "event_data": 51643508, "tracking_data": 51643529},
    "J03WOY": {"metadata": 51643481, "event_data": 51643502, "tracking_data": 51643523},
    "J03WR9": {"metadata": 51643490, "event_data": 51643511, "tracking_data": 51643532},
}


def _get_sportec_open_data_url(game_id: str, data_type: str) -> str:
    if game_id not in FILE_ID_MAP:
        raise ValueError(
            f"Unknown game id {game_id}, please specify one of "
            f"{list(FILE_ID_MAP.keys())}"
        )

    if data_type not in ["metadata", "event_data", "tracking_data"]:
        raise ValueError(
            f"Unknown data type {data_type}, please specify one of "
            "['metadata', 'tracking_data', 'event_data']"
        )

    return f"{SPORTEC_BASE_URL}/{FILE_ID_MAP[game_id][data_type]}"


DFB_POSITIONS = {
    "Sub": "unspecified",
    "TW": "goalkeeper",
    "RV": "defender",  # right back
    "IVR": "defender",  # right center back
    "IVL": "defender",  # left center back
    "IVZ": "defender",
    "LV": "defender",  # left back
    "ZD": "midfielder",  # central defensive midfielder
    "ORM": "midfielder",  # right midfield offensive
    "OLM": "midfielder",  # left midfield offsenive
    "DML": "midfielder",  # left midfield defensive
    "DMR": "midfielder",  # right midfield defensive
    "DLM": "midfielder",
    "DRM": "midfielder",
    "RM": "midfielder",
    "LM": "midfielder",
    "HR": "midfielder",
    "HL": "midfielder",
    "DMZ": "midfielder",
    "ZO": "midfielder",  # central offensive midfielder
    "STL": "forward",  # left striker
    "STZ": "forward",  # central striker
    "STR": "forward",  # right striker
    "RA": "forward",
    "LA": "forward",
}


def _get_sportec_metadata(metadata_loc: str, only_event_data: bool = False) -> Metadata:
    """Get the metadata of sportec. This metadata is used for gamees form the
    German Bundesliga, the data is assumed to be distributed by the shell company
    SporTec Solutions under the DFL.

    Args:
        metadata_loc (str): The location of the metadata xml
        only_event_data (bool, optional): Wheter the metadata is rendered for the event
            data or for the tracking data. Defaults to False.

    Returns:
        Metadata: The metadata object from sportec
    """

    with open(metadata_loc, "rb") as file:
        encoding = chardet.detect(file.read())["encoding"]
    with open(metadata_loc, "r", encoding=encoding) as file:
        lines = file.read()

    lines = lines.replace("ï»¿", "")
    soup = BeautifulSoup(lines, "xml")
    teams_info = {}
    for team in soup.find_all("Team"):
        players = team.find_all("Player")
        player_dict = {
            "id": [""] * len(players),
            "full_name": [""] * len(players),
            "shirt_num": [MISSING_INT] * len(players),
            "position": [""] * len(players),
            "start_frame": [MISSING_INT] * len(players),
            "end_frame": [MISSING_INT] * len(players),
            "starter": [None] * len(players),
        }

        for i, player in enumerate(team.find_all("Player")):
            player_dict["id"][i] = player["PersonId"]
            player_dict["full_name"][i] = player["FirstName"] + " " + player["LastName"]
            player_dict["shirt_num"][i] = int(player["ShirtNumber"])
            player_dict["position"][i] = DFB_POSITIONS[
                player.get("PlayingPosition", "Sub")
            ]
            player_dict["starter"][i] = player["Starting"] == "true"

        team_side = "home" if team["Role"] == "home" else "away"
        teams_info[f"{team_side}_team_id"] = team["TeamId"]
        teams_info[f"{team_side}_team_name"] = team["TeamName"]
        teams_info[f"{team_side}_players"] = pd.DataFrame(player_dict)
        teams_info[f"{team_side}_score"] = int(
            soup.find("General")["Result"].split(":")[team_side == "away"]
        )
        teams_info[f"{team_side}_formation"] = (
            team["LineUp"].split(" ")[0].replace("-", "")
        )

    pitch_size_x = float(soup.find("Environment")["PitchX"])
    pitch_size_y = float(soup.find("Environment")["PitchY"])

    if only_event_data:
        frames_df = pd.DataFrame(
            {
                "period_id": [1, 2, 3, 4, 5],
                "start_datetime_ed": [pd.to_datetime("NaT")] * 5,
                "end_datetime_ed": [pd.to_datetime("NaT")] * 5,
            }
        )
    else:
        frames_df = pd.DataFrame(
            {
                "period_id": [1, 2, 3, 4, 5],
                "start_frame": [MISSING_INT] * 5,
                "end_frame": [MISSING_INT] * 5,
                "start_datetime_td": [pd.to_datetime("NaT")] * 5,
                "end_datetime_td": [pd.to_datetime("NaT")] * 5,
            }
        )

    return Metadata(
        game_id=soup.find("General")["MatchId"],
        pitch_dimensions=[pitch_size_x, pitch_size_y],
        periods_frames=frames_df,
        frame_rate=MISSING_INT,
        country="Germany",
        **teams_info,
    )
