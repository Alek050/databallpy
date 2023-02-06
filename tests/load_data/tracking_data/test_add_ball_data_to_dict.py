import unittest

import numpy as np

from databallpy.load_data.tracking_data._add_ball_data_to_dict import (
    _add_ball_data_to_dict,
)


class TestAddBallDataToDict(unittest.TestCase):
    def setUp(self):
        self.ball_info = [
            ["150", "-46", "7", "A", "Alive"],
            ["22", "17", "15", "H", "Dead"],
            ["124", np.nan, "7", "A", "Dead"],
        ]

        self.expected_dict = {
            "timestamp": [0, 1, 2],
            "ball_x": [150.0, 22.0, 124.0],
            "ball_y": [-46.0, 17.0, np.nan],
            "ball_z": [7.0, 15.0, 7.0],
            "ball_posession": ["Away", "Home", "Away"],
            "ball_status": ["Alive", "Dead", "Dead"],
        }

    def test_add_ball_data_to_dict(self):
        data = {
            "timestamp": [np.nan] * 3,
            "ball_x": [np.nan] * 3,
            "ball_y": [np.nan] * 3,
            "ball_z": [np.nan] * 3,
            "ball_posession": [np.nan] * 3,
            "ball_status": [np.nan] * 3,
        }

        for i, input in enumerate(self.ball_info):
            data["timestamp"][i] = int(i)
            ball_x, ball_y, ball_z, ball_posession, ball_status = input
            data = _add_ball_data_to_dict(
                ball_x, ball_y, ball_z, ball_posession, ball_status, data, i
            )

        assert data == self.expected_dict
