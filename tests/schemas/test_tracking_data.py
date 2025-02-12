import os
import unittest

import pandas as pd

from databallpy.game import Game
from databallpy.utils.warnings import DataBallPyWarning
from databallpy.utils.get_game import get_game
from databallpy.schemas.tracking_data import TrackingDataSchema, TrackingData


class TestTrackingDataSchema(unittest.TestCase):
    def setUp(self):
        base_path = os.path.join("tests", "test_data")

        td_tracab_loc = os.path.join(base_path, "tracab_td_test.dat")
        md_tracab_loc = os.path.join(base_path, "tracab_metadata_test.xml")
        ed_opta_loc = os.path.join(base_path, "f24_test.xml")
        md_opta_loc = os.path.join(base_path, "f7_test.xml")
        self.td_provider = "tracab"
        self.ed_provider = "opta"

        self.expected_game_tracab_opta = get_game(
            tracking_data_loc=td_tracab_loc,
            tracking_metadata_loc=md_tracab_loc,
            tracking_data_provider="tracab",
            event_data_loc=ed_opta_loc,
            event_metadata_loc=md_opta_loc,
            event_data_provider="opta",
            check_quality=False,
        )

    def test_tracking_data_provider_setter(self):
        df = TrackingData()
        with self.assertRaises(AttributeError):
            df.provider = "new_provider"

    def test_frame_rate_setter(self):
        df = TrackingData()
        with self.assertRaises(AttributeError):
            df.frame_rate = "new_provider"

    def test_check_ball_status(self):
        td_changed = self.expected_game_tracab_opta.tracking_data.copy()
        td_changed["ball_status"] = ["dead", "dead", "dead", "dead", "alive"]

        with self.assertWarns(DataBallPyWarning):
            TrackingDataSchema.validate(td_changed)

    def test_check_first_frame(self):
        td_changed = self.expected_game_tracab_opta.tracking_data.copy()
        td_changed["ball_x"] += 10.0

        with self.assertWarns(DataBallPyWarning):
            TrackingDataSchema.validate(td_changed)
