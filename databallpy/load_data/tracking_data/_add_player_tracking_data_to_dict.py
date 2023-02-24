import numpy as np


def _add_player_tracking_data_to_dict(
    team: str, shirt_num: str, x: str, y: str, data: dict, idx: int
) -> dict:
    """Function that adds the data of one player to the data dict for one frame

    Args:
        team (str): team side, either "home" or "away"
        shirt_num (str): player's shirt number
        x (float): player's x coordinate at current idx
        y (float): player's y coordinate at current idx
        data (dict): data dictionary to write results to
        idx (int): indicates position in data dictionary

    Returns:
        dict: contains all tracking data
    """

    if f"{team}_{shirt_num}_x" not in data.keys():  # create keys for new player
        data[f"{team}_{shirt_num}_x"] = [np.nan] * len(data["timestamp"])
        data[f"{team}_{shirt_num}_y"] = [np.nan] * len(data["timestamp"])

    data[f"{team}_{shirt_num}_x"][idx] = float(x)
    data[f"{team}_{shirt_num}_y"][idx] = float(y)

    return data
