import chardet
import pandas as pd
from bs4 import BeautifulSoup

from databallpy.data_parsers.metadata import Metadata
from databallpy.utils.constants import MISSING_INT

SPORTEC_BASE_URL = "https://figshare.com/ndownloader/files"
SPORTEC_PRIVATE_LINK = "1f806cb3e755c6b54e05"
SPORTEC_METADATA_ID_MAP = {
    "J03WMX": "48392485",
    "J03WN1": "48392491",
    "J03WPY": "48392497",
    "J03WOH": "48392515",
    "J03WQQ": "48392488",
    "J03WOY": "48392503",
    "J03WR9": "48392494",
}
SPORTEC_EVENT_DATA_ID_MAP = {
    "J03WMX": "48392524",
    "J03WN1": "48392527",
    "J03WPY": "48392542",
    "J03WOH": "48392500",
    "J03WQQ": "48392521",
    "J03WOY": "48392518",
    "J03WR9": "48392530",
}
SPORTEC_TRACKING_DATA_ID_MAP = {
    "J03WMX": "48392539",
    "J03WN1": "48392512",
    "J03WPY": "48392572",
    "J03WOH": "48392578",
    "J03WQQ": "48392545",
    "J03WOY": "48392551",
    "J03WR9": "48392563",
}


def _get_sportec_open_data_url(game_id: str, data_type: str) -> str:
    if game_id not in SPORTEC_EVENT_DATA_ID_MAP:
        raise ValueError(
            f"Unknown game id {game_id}, please specify one of "
            f"{list(SPORTEC_EVENT_DATA_ID_MAP.keys())}"
        )

    data_type_map = {
        "metadata": SPORTEC_METADATA_ID_MAP,
        "event_data": SPORTEC_EVENT_DATA_ID_MAP,
        "tracking_data": SPORTEC_TRACKING_DATA_ID_MAP,
    }

    if data_type not in data_type_map:
        raise ValueError(
            f"Unknown data type {data_type}, please specify one of "
            "['metadata', 'tracking_data', 'event_data']"
        )

    return (
        f"{SPORTEC_BASE_URL}/{data_type_map[data_type][game_id]}"
        f"?private_link={SPORTEC_PRIVATE_LINK}"
    )


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
