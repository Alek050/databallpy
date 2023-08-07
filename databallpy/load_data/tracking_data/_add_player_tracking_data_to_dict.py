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

    Note:
        Generally, if-statements are quicker than try-except blocks, but since
        this function is called for every player in every frame, it is quicker
        to use a try-except block here. The except block will only be reached
        for every player once, while the if-statement would be reached for every
        player in every frame.
    """

    try:
        data[f"{team}_{shirt_num}_x"][idx] = float(x)
        data[f"{team}_{shirt_num}_y"][idx] = float(y)
    except KeyError:
        data[f"{team}_{shirt_num}_x"] = [np.nan] * len(data["frame"])
        data[f"{team}_{shirt_num}_y"] = [np.nan] * len(data["frame"])
        data[f"{team}_{shirt_num}_x"][idx] = float(x)
        data[f"{team}_{shirt_num}_y"][idx] = float(y)

    return data
