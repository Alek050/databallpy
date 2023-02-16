import unittest
import os
import matplotlib.pyplot as plt
import pandas as pd

from databallpy.visualize import plot_soccer_pitch, requires_ffmpeg, save_match_clip
from databallpy.match import get_match


class TestVisualize(unittest.TestCase):
    def test_requires_ffmpeg_wrapper(self):
        @requires_ffmpeg
        def test_function():
            return "Hello World"

        self.assertEqual(test_function(), "Hello World")

    def test_plot_soccer_pitch(self):
        pitch, ax = plot_soccer_pitch()
        self.assertIsInstance(pitch, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(len(ax.lines), 27)
        self.assertEqual(len(ax.collections), 3)

        pitch, ax = plot_soccer_pitch(pitch_color="white")
        self.assertIsInstance(pitch, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(len(ax.lines), 27)
        self.assertEqual(len(ax.collections), 3)
    
    def test_save_match_clip(self):
        match = get_match(
            tracking_data_loc="tests/test_data/tracab_td_test.dat",
            tracking_metadata_loc="tests/test_data/tracab_metadata_test.xml",
            tracking_data_provider="tracab",
            event_data_loc="tests/test_data/f24_test.xml",
            event_metadata_loc="tests/test_data/f7_test.xml",
            event_data_provider="opta"
        )
        series = pd.Series([22, 23, 25], index=[1, 2, 3])
        with self.assertRaises(AssertionError):
            save_match_clip(match, 1, 3, save_folder="tests/test_data", title="test_clip", events=["pass"])
            save_match_clip(match, 0, 2, save_folder="tests/test_data", title="test_clip", variable_of_interest=series)
        
        assert not os.path.exists("tests/test_data/test_clip.mp4")

        save_match_clip(match, 1, 3, save_folder="tests/test_data", title="test_clip", variable_of_interest=series)

        assert os.path.exists("tests/test_data/test_clip.mp4")
        os.remove("tests/test_data/test_clip.mp4")

        

