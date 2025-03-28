import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databallpy.features import add_velocity
from databallpy.utils.errors import DataBallPyError
from databallpy.utils.get_game import get_game
from databallpy.visualize import (
    _pre_check_plot_td_inputs,
    plot_events,
    plot_soccer_pitch,
    plot_tracking_data,
    requires_ffmpeg,
    save_tracking_video,
)


class TestVisualize(unittest.TestCase):
    def setUp(self):
        self.game = get_game(
            tracking_data_loc="tests/test_data/tracab_td_test.dat",
            tracking_metadata_loc="tests/test_data/tracab_metadata_test.xml",
            tracking_data_provider="tracab",
            event_data_loc="tests/test_data/f24_test.xml",
            event_metadata_loc="tests/test_data/f7_test.xml",
            event_data_provider="opta",
            check_quality=False,
        )
        self.game.tracking_data["player_possession"] = "home_34"

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
        game = self.game.copy()

        # Call plot_events function with different arguments
        fig, ax = plot_events(
            game,
            events=["pass", "dribble"],
            player_ids=[45849],
            team_id=3,
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
            game,
            events=["pass", "dribble"],
            player_ids=[45849],
            outcome=1,
            team_id=3,
            color_by_col="team_id",
            team_colors=["blue", "red"],
            title="My Test Plot",
        )
        assert fig is None
        assert ax is None

        fig, ax = plot_events(
            game,
            events=["pass", "dribble"],
            player_ids=[45849],
            team_id=3,
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
            game,
            events=["pass", "dribble"],
            player_ids=[45849],
            team_id=3,
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
            plot_events(self.game.copy(), events="passes")
        with self.assertRaises(ValueError):
            plot_events(self.game.copy(), events=["wrong_event"])
        with self.assertRaises(ValueError):
            plot_events(self.game.copy(), outcome="false")
        with self.assertRaises(ValueError):
            plot_events(self.game.copy(), player_ids=[1, 2])
        with self.assertRaises(ValueError):
            plot_events(self.game.copy(), team_id="12345678")

    def test_plot_tracking_data(self):
        game = self.game.copy()
        idx = 1
        x, _ = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 10, 10))
        with self.assertRaises(DataBallPyError):
            fig, ax = plot_tracking_data(
                game,
                idx,
                title="My Test Plot",
                heatmap_overlay=x,
                overlay_cmap="viridis",
                add_velocities=True,
            )

        add_velocity(
            game.tracking_data,
            ["home_34", "away_17", "ball"],
            frame_rate=1.0,
            inplace=True,
        )
        game.tracking_data["databallpy_event"] = "pass"
        game.tracking_data["event_id"] = game.pass_events["event_id"].iloc[0]
        game._is_synchronised = True

        fig, ax = plot_tracking_data(
            game,
            idx,
            title="My Test Plot",
            add_player_possession=True,
            add_velocities=True,
            heatmap_overlay=x,
            overlay_cmap="viridis",
            events=["pass", "dribble"],
            variable_of_interest="12",
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)

        plt.close(fig)

    def test_save_game_clip(self):
        game = self.game.copy()
        game.home_players = game.home_players.iloc[0:1]
        game.home_players["shirt_num"] = 34
        game.away_players = game.away_players.iloc[0:1]
        game.away_players["shirt_num"] = 17
        game.tracking_data["player_possession"] = [
            None,
            "home_34",
            None,
            None,
            "away_17",
        ]
        series = pd.Series([22, 23, 25], index=[1, 2, 3])
        with self.assertRaises(DataBallPyError):
            save_tracking_video(
                game,
                1,
                3,
                save_folder="tests/test_data",
                title="test_clip",
                events=["pass"],
            )
        with self.assertRaises(DataBallPyError):
            save_tracking_video(
                game,
                0,
                2,
                save_folder="tests/test_data",
                title="test_clip",
                variable_of_interest=series,
            )
        with self.assertRaises(DataBallPyError):
            save_tracking_video(
                game,
                1,
                3,
                save_folder="tests/test_data",
                title="test_clip",
                heatmap_overlay=np.zeros((4, 10, 10)),
            )
        with self.assertRaises(DataBallPyError):
            save_tracking_video(
                game,
                1,
                3,
                save_folder="tests/test_data",
                title="test_clip",
                heatmap_overlay=np.zeros((3, 10, 10, 10)),
            )

        with self.assertRaises(DataBallPyError):
            save_tracking_video(
                game,
                1,
                3,
                save_folder="tests/test_data",
                title="test_clip",
                heatmap_overlay=np.zeros((3, 10, 10)),
                overlay_cmap="custom_cmap",
            )

        if os.path.exists("tests/test_data/test_clip.mp4"):
            os.remove("tests/test_data/test_clip.mp4")

        game.tracking_data["databallpy_event"] = None
        game.tracking_data["event_id"] = None
        game.tracking_data.loc[2, "databallpy_event"] = "pass"
        game.tracking_data.loc[2, "event_id"] = game.pass_events["event_id"].iloc[0]
        game._is_synchronised = True

        heatmap = np.zeros((3, 10, 10))

        save_tracking_video(
            game,
            1,
            3,
            save_folder="tests/test_data",
            title="test_clip",
            add_player_possession=True,
            variable_of_interest=series,
            heatmap_overlay=heatmap,
            overlay_cmap="viridis",
            events=["pass"],
        )

        assert os.path.exists("tests/test_data/test_clip.mp4")
        os.remove("tests/test_data/test_clip.mp4")

    def test_save_game_clip_with_events(self):
        synced_game = get_game(
            tracking_data_loc="tests/test_data/sync/tracab_td_sync_test.dat",
            tracking_metadata_loc="tests/test_data/sync/tracab_metadata_sync_test.xml",
            tracking_data_provider="tracab",
            event_data_loc="tests/test_data/sync/opta_events_sync_test.xml",
            event_metadata_loc="tests/test_data/sync/opta_metadata_sync_test.xml",
            event_data_provider="opta",
            check_quality=False,
        )
        synced_game.home_players = synced_game.home_players.iloc[1:2]
        synced_game.away_players = synced_game.away_players.iloc[0:1]
        for col in synced_game.get_column_ids():
            mask = synced_game.tracking_data[col + "_x"].notnull()
            synced_game.tracking_data.loc[mask, col + "_vx"] = 2
            synced_game.tracking_data.loc[mask, col + "_vy"] = 2
            synced_game.tracking_data.loc[mask, col + "_velocity"] = 2.67
        synced_game.allow_synchronise_tracking_and_event_data = True

        synced_game.synchronise_tracking_and_event_data(n_batches=2)
        events = [
            "pass",
            "dribble",
            "shot",
        ]

        if os.path.exists("tests/test_data/test_game_with_events.mp4"):
            os.remove("tests/test_data/test_game_with_events.mp4")

        save_tracking_video(
            synced_game,
            1,
            10,
            save_folder="tests/test_data",
            title="test_game_with_events",
            events=events,
            heatmap_overlay=np.zeros((10, 10, 10)),
            add_velocities=True,
            verbose=False,
        )

        assert os.path.exists("tests/test_data/test_game_with_events.mp4")
        os.remove("tests/test_data/test_game_with_events.mp4")

    def test_pre_check_plot_td_inputs(self):
        game = self.game.copy()
        _pre_check_plot_td_inputs(
            game, game.tracking_data, [], None, False, False, None, "viridis"
        )

        with self.assertRaises(DataBallPyError):
            _pre_check_plot_td_inputs(
                "game", game.tracking_data, [], None, False, False, None, "viridis"
            )

        with self.assertRaises(DataBallPyError):
            _pre_check_plot_td_inputs(
                game,
                game.tracking_data,
                [],
                [1] * (len(game.tracking_data) + 2),
                False,
                False,
                None,
                "viridis",
            )

        game.tracking_data.drop(columns=["player_possession"], inplace=True)
        with self.assertRaises(DataBallPyError):
            _pre_check_plot_td_inputs(
                game, game.tracking_data, [], None, True, False, None, "viridis"
            )

        with self.assertRaises(DataBallPyError):
            _pre_check_plot_td_inputs(
                game, game.tracking_data, [], None, False, True, None, "viridis"
            )

        with self.assertRaises(DataBallPyError):
            _pre_check_plot_td_inputs(
                game,
                game.tracking_data,
                [],
                None,
                False,
                False,
                [[2, 3, 4], [5, 6, 7]],
                "viridis",
            )
        with self.assertRaises(DataBallPyError):
            _pre_check_plot_td_inputs(
                game,
                game.tracking_data.iloc[0:1],
                [],
                None,
                False,
                False,
                np.array([[[2, 3, 4], [5, 6, 7]], [[2, 3, 4], [5, 6, 7]]]),
                "viridis",
            )

        with self.assertRaises(DataBallPyError):
            _pre_check_plot_td_inputs(
                game,
                game.tracking_data,
                [],
                None,
                False,
                False,
                [[2, 3, 4], [5, 6, 7]],
                "my_custom_cmap",
            )
