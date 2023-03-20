def _add_ball_data_to_dict(
    ball_x: str,
    ball_y: str,
    ball_z: str,
    posession: str,
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

    data["ball_x"][idx] = float(ball_x)
    data["ball_y"][idx] = float(ball_y)
    data["ball_z"][idx] = float(ball_z)
    data["ball_posession"][idx] = posession
    data["ball_status"][idx] = status

    return data
