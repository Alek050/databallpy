import os
import unittest

import matplotlib.pyplot as plt
import pandas as pd
from databallpy.utils.errors import DataBallPyError

from databallpy.get_match import get_match
from databallpy.visualize import (
    plot_events,
    plot_soccer_pitch,
    requires_ffmpeg,
    save_match_clip,
)


class TestVisualize(unittest.TestCase):
    def setUp(self):
        self.match = get_match(
            tracking_data_loc="tests/test_data/tracab_td_test.dat",
            tracking_metadata_loc="tests/test_data/tracab_metadata_test.xml",
            tracking_data_provider="tracab",
            event_data_loc="tests/test_data/f24_test.xml",
            event_metadata_loc="tests/test_data/f7_test.xml",
            event_data_provider="opta",
            check_quality=False,
        )

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

    def test_plot_events(self):
        match = self.match.copy()

        # Call plot_events function with different arguments
        fig, ax = plot_events(
            match,
            events=["pass", "dribble"],
            player_ids=[45849],
            team_id=3,
            pitch_color="green",
            color_by_col="team_id",
            team_colors=["blue", "red"],
            title="My Test Plot",
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)

        # Check plot elements
        self.assertEqual(ax.get_title(), "My Test Plot")
        self.assertEqual(ax.get_legend().get_texts()[0].get_text(), "TeamOne")
        self.assertEqual(ax.get_legend().get_texts()[1].get_text(), "TeamTwo")
        self.assertEqual(len(ax.collections), 5)
        
        fig, ax = plot_events(
            match,
            events=["pass", "dribble"],
            player_ids=[45849],
            outcome=1,
            team_id=3,
            pitch_color="green",
            color_by_col="team_id",
            team_colors=["blue", "red"],
            title="My Test Plot",
        )
        assert fig is None
        assert ax is None

        fig, ax = plot_events(
            match,
            events=["pass", "dribble"],
            player_ids=[45849],
            team_id=3,
            pitch_color="green",
            color_by_col="databallpy_event",
            team_colors=["blue", "red"],
            title="My Test Plot2",
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)

        # Check plot elements
        self.assertEqual(ax.get_title(), "My Test Plot2")
        self.assertEqual(ax.get_legend().get_texts()[0].get_text(), "dribble")
        self.assertEqual(len(ax.collections), 4)

        fig, ax = plot_events(
            match,
            events=["pass", "dribble"],
            player_ids=[45849],
            team_id=3,
            pitch_color="green",
            team_colors=["blue", "red"],
            title="My Test Plot3",
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)

        # Check plot elements
        self.assertEqual(ax.get_title(), "My Test Plot3")
        self.assertEqual(len(ax.collections), 4)

    def test_save_match_clip(self):
        match = self.match.copy()
        match.tracking_data["player_possession"] = [
            None,
            "home_34",
            None,
            None,
            "away_17",
        ]
        series = pd.Series([22, 23, 25], index=[1, 2, 3])
        with self.assertRaises(DataBallPyError):
            save_match_clip(
                match,
                1,
                3,
                save_folder="tests/test_data",
                title="test_clip",
                events=["pass"],
            )
        with self.assertRaises(DataBallPyError):
            save_match_clip(
                match,
                0,
                2,
                save_folder="tests/test_data",
                title="test_clip",
                variable_of_interest=series,
            )
        with self.assertRaises(DataBallPyError):
            save_match_clip(
                match,
                1,
                3,
                save_folder="tests/test_data",
                title="test_clip",
                variable_of_interest=series,
                player_possession_column="unknown_column",
            )

        assert not os.path.exists("tests/test_data/test_clip.mp4")

        save_match_clip(
            match,
            1,
            3,
            save_folder="tests/test_data",
            title="test_clip",
            variable_of_interest=series,
            player_possession_column="player_possession",
        )

        assert os.path.exists("tests/test_data/test_clip.mp4")
        os.remove("tests/test_data/test_clip.mp4")

    def test_save_match_clip_with_events(self):
        synced_match = get_match(
            tracking_data_loc="tests/test_data/sync/tracab_td_sync_test.dat",
            tracking_metadata_loc="tests/test_data/sync/tracab_metadata_sync_test.xml",
            tracking_data_provider="tracab",
            event_data_loc="tests/test_data/sync/opta_events_sync_test.xml",
            event_metadata_loc="tests/test_data/sync/opta_metadata_sync_test.xml",
            event_data_provider="opta",
            check_quality=False,
        )
        synced_match.allow_synchronise_tracking_and_event_data = True

        synced_match.synchronise_tracking_and_event_data(n_batches=2)
        events = [
            "pass",
            "dribble",
            "shot",
        ]

        assert not os.path.exists("tests/test_data/test_match_with_events.mp4")

        save_match_clip(
            synced_match,
            1,
            10,
            save_folder="tests/test_data",
            title="test_match_with_events",
            events=events,
        )

        assert os.path.exists("tests/test_data/test_match_with_events.mp4")
        os.remove("tests/test_data/test_match_with_events.mp4")
