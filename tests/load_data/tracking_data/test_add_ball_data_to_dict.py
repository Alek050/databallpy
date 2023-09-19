import unittest

import numpy as np

from databallpy.load_data.tracking_data._add_ball_data_to_dict import (
    _add_ball_data_to_dict,
)


class TestAddBallDataToDict(unittest.TestCase):
    def setUp(self):
        self.ball_info = [
            ["150", "-46", "7", "away", "Alive"],
            ["22", "17", "15", "home", "Dead"],
            ["124", np.nan, "7", "away", "Dead"],
        ]

        self.expected_dict = {
            "frame": [0, 1, 2],
            "ball_x": [150.0, 22.0, 124.0],
            "ball_y": [-46.0, 17.0, np.nan],
            "ball_z": [7.0, 15.0, 7.0],
            "ball_possession": ["away", "home", "away"],
            "ball_status": ["alive", "dead", "dead"],
        }

    def test_add_ball_data_to_dict(self):
        data = {
            "frame": [np.nan] * 3,
            "ball_x": [np.nan] * 3,
            "ball_y": [np.nan] * 3,
            "ball_z": [np.nan] * 3,
            "ball_possession": [np.nan] * 3,
            "ball_status": [np.nan] * 3,
        }

        for i, input in enumerate(self.ball_info):
            data["frame"][i] = int(i)
            ball_x, ball_y, ball_z, ball_possession, ball_status = input
            data = _add_ball_data_to_dict(
                ball_x, ball_y, ball_z, ball_possession, ball_status.lower(), data, i
            )

        assert data == self.expected_dict
