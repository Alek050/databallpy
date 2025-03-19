from databallpy.utils.utils import _to_float


def _add_ball_data_to_dict(
    ball_x: str,
    ball_y: str,
    ball_z: str,
    possession: str,
    status: str,
    data: dict,
    idx: str,
) -> dict:
    """Function that adds the data of the ball to the data dict for one frame

    Args:
        ball_info (str): containing data from the ball
        data (dict): data dictionary to write results to
        idx (int): indicates position in data dictionary

    Returns:
        dict: contains all tracking data
    """

    data["ball_x"][idx] = _to_float(ball_x)
    data["ball_y"][idx] = _to_float(ball_y)
    data["ball_z"][idx] = _to_float(ball_z)
    data["team_possession"][idx] = possession
    data["ball_status"][idx] = status

    return data
