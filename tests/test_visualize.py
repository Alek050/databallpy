import os
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databallpy.get_match import get_match
from databallpy.utils.errors import DataBallPyError
from databallpy.utils.warnings import DataBallPyWarning
from databallpy.visualize import (
    plot_events,
    plot_soccer_pitch,
    requires_ffmpeg,
    save_tracking_video,
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

        with self.assertRaises(ValueError):
            plot_soccer_pitch(pitch_color="invalid_color")

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

    def test_plot_events_wrong_input(self):
        with self.assertRaises(TypeError):
            plot_events(self.match.copy(), events="passes")
        with self.assertRaises(ValueError):
            plot_events(self.match.copy(), events=["tackle"])
        with self.assertRaises(ValueError):
            plot_events(self.match.copy(), outcome="false")
        with self.assertRaises(ValueError):
            plot_events(self.match.copy(), player_ids=[1, 2])
        with self.assertRaises(ValueError):
            plot_events(self.match.copy(), team_id="12345678")

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
            save_tracking_video(
                match,
                1,
                3,
                save_folder="tests/test_data",
                title="test_clip",
                events=["pass"],
            )
        with self.assertRaises(DataBallPyError):
            save_tracking_video(
                match,
                0,
                2,
                save_folder="tests/test_data",
                title="test_clip",
                variable_of_interest=series,
            )
        with self.assertRaises(DataBallPyError):
            save_tracking_video(
                match,
                1,
                3,
                save_folder="tests/test_data",
                title="test_clip",
                variable_of_interest=series,
                player_possession_column="unknown_column",
            )
        with self.assertRaises(DataBallPyError):
            save_tracking_video(
                match,
                1,
                3,
                save_folder="tests/test_data",
                title="test_clip",
                add_pitch_control=True,
            )

        if os.path.exists("tests/test_data/test_clip.mp4"):
            os.remove("tests/test_data/test_clip.mp4")

        save_tracking_video(
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

    @patch("databallpy.visualize.get_pitch_control_period")
    def test_save_match_clip_with_events(self, mock_get_pitch_control_period):
        synced_match = get_match(
            tracking_data_loc="tests/test_data/sync/tracab_td_sync_test.dat",
            tracking_metadata_loc="tests/test_data/sync/tracab_metadata_sync_test.xml",
            tracking_data_provider="tracab",
            event_data_loc="tests/test_data/sync/opta_events_sync_test.xml",
            event_metadata_loc="tests/test_data/sync/opta_metadata_sync_test.xml",
            event_data_provider="opta",
            check_quality=False,
        )
        mock_get_pitch_control_period.return_value = np.zeros(
            (len(synced_match.tracking_data), 240, 360)
        )
        for col in (
            synced_match.home_players_column_ids()
            + synced_match.away_players_column_ids()
        ):
            mask = synced_match.tracking_data[col + "_x"].notnull()
            synced_match.tracking_data.loc[mask, col + "_vx"] = 2
            synced_match.tracking_data.loc[mask, col + "_vy"] = 2
            synced_match.tracking_data.loc[mask, col + "_velocity"] = 2.67
        synced_match.allow_synchronise_tracking_and_event_data = True

        grid_size = [360, 240]
        x_range = np.linspace(
            -100 / 2 - 5,
            100 / 2 + 5,
            grid_size[0],
        )
        y_range = np.linspace(
            -50 / 2 - 5,
            50 / 2 + 5,
            grid_size[1],
        )
        grid = np.meshgrid(x_range, y_range)

        synced_match.synchronise_tracking_and_event_data(n_batches=2)
        events = [
            "pass",
            "dribble",
            "shot",
        ]

        if os.path.exists("tests/test_data/test_match_with_events.mp4"):
            os.remove("tests/test_data/test_match_with_events.mp4")

        with self.assertWarns(DataBallPyWarning):
            save_tracking_video(
                synced_match,
                1,
                10,
                save_folder="tests/test_data",
                title="test_match_with_events",
                events=events,
                add_pitch_control=True,
                add_velocities=True,
                verbose=False,
            )

        assert os.path.exists("tests/test_data/test_match_with_events.mp4")
        os.remove("tests/test_data/test_match_with_events.mp4")
        grid_res = mock_get_pitch_control_period.call_args.kwargs["grid"]
        td_res = mock_get_pitch_control_period.call_args.kwargs["tracking_data"]
        pd.testing.assert_frame_equal(synced_match.tracking_data.loc[1:10], td_res)
        np.testing.assert_allclose(grid_res, grid)
