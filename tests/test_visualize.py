import os
import unittest

import pandas as pd

from databallpy.match import get_match
from databallpy.visualize import save_match_clip


class TestVisualize(unittest.TestCase):
    def setUp(self):
        self.match = get_match(
            tracking_data_loc="tests/test_data/tracab_td_test.dat",
            tracking_metadata_loc="tests/test_data/tracab_metadata_test.xml",
            tracking_data_provider="tracab",
            event_data_loc="tests/test_data/f24_test.xml",
            event_metadata_loc="tests/test_data/f7_test.xml",
            event_data_provider="opta",
        )

    def test_save_match_clip(self):
        assert not os.path.exists("tests/test_data/test_clip.mp4")

        save_match_clip(
            self.match,
            0,
            2,
            save_folder="tests/test_data",
            title="test_clip",
            variable_of_interest=pd.Series(["1", "2", "3"]),
        )
        assert os.path.exists("tests/test_data/test_clip.mp4")

        if os.path.exists("tests/test_data/test_clip.mp4"):
            os.remove("tests/test_data/test_clip.mp4")
