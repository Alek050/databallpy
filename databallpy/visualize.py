import subprocess
from functools import wraps

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap
from tqdm import tqdm

from databallpy.game import Game
from databallpy.utils.errors import DataBallPyError
from databallpy.utils.logging import logging_wrapper


def requires_ffmpeg(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception:
            raise FileNotFoundError(
                "Could not find the subprocess ffmpeg. Make sure ffmpeg is installed"
                " globally on you device and added to your python path."
            )

        return func(*args, **kwargs)

    return wrapper


@logging_wrapper(__file__)
def plot_soccer_pitch(
    field_dimen: tuple[float, float] = (106.0, 68.0),
    pitch_color: str = "mediumseagreen",
    linewidth: int = 2,
    markersize: int = 20,
    fig: plt.figure = None,
    ax: plt.axes = None,
) -> tuple[plt.figure, plt.axes]:
    """A function to plot a soccer pitch
    Note: relies heavily on https://github.com/Friends-of-Tracking-Data-FoTD/
    LaurieOnTracking/blob/master/Metrica_Viz.py

    Args:
        field_dimen (tuple, optional): x and y length of pitch in meters. Defaults to
            (106.0, 68.0).
        pitch_color (str, optional): Color of the pitch. Defaults to "mediumseagreen".
        linewidth (int, optional): Width of the lines on the pitch. Defaults to 2.
        markersize (int, optional): Size of the dots on the pitch. Defaults to 20.
        fig (plt.figure, optional): Figure to plot the pitch on. Defaults to None.
        ax (plt.axes, optional): Axes to plot the pitch on. Defaults to None.

    Returns:
        Tuple[plt.figure, plt.axes]: figure and axes with the pitch depicted on it
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Set pitch and line colors
    ax.set_facecolor(pitch_color)
    if pitch_color not in ["white", "w"]:
        lc = "whitesmoke"  # line color
        pc = "w"  # 'spot' colors
    else:
        lc = "k"
        pc = "k"

    # All dimensions in meters
    border_dimen = (3, 3)  # include a border arround of the field of width 3m
    half_pitch_length = field_dimen[0] / 2.0  # length of half pitch
    half_pitch_width = field_dimen[1] / 2.0  # width of half pitch

    # Soccer field dimensions are in yards, so we need to convert them to meters
    meters_per_yard = 0.9144  # unit conversion from yards to meters
    goal_line_width = 8 * meters_per_yard
    box_width = 20 * meters_per_yard
    box_length = 6 * meters_per_yard
    area_width = 44 * meters_per_yard
    area_length = 18 * meters_per_yard
    penalty_spot = 12 * meters_per_yard
    corner_radius = 1 * meters_per_yard
    box_circle_length = 8 * meters_per_yard
    box_circle_radius = 10 * meters_per_yard
    box_circle_pos = 12 * meters_per_yard
    centre_circle_radius = 10 * meters_per_yard

    zorder = -2

    # Plot half way line
    ax.plot(
        [0, 0],
        [-half_pitch_width, half_pitch_width],
        lc,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.scatter(
        0.0, 0.0, marker="o", facecolor=lc, linewidth=0, s=markersize, zorder=zorder
    )
    # Plot center circle
    y = np.linspace(-1, 1, 150) * centre_circle_radius
    x = np.sqrt(centre_circle_radius**2 - y**2)
    ax.plot(x, y, lc, linewidth=linewidth, zorder=zorder)
    ax.plot(-x, y, lc, linewidth=linewidth, zorder=zorder)

    signs = [-1, 1]
    for s in signs:  # plots each line seperately
        # Plot pitch boundary
        ax.plot(
            [-half_pitch_length, half_pitch_length],
            [s * half_pitch_width, s * half_pitch_width],
            lc,
            linewidth=linewidth,
            zorder=zorder,
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length],
            [-half_pitch_width, half_pitch_width],
            lc,
            linewidth=linewidth,
            zorder=zorder,
        )

        # Goal posts & line
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length],
            [-goal_line_width / 2.0, goal_line_width / 2.0],
            pc + "s",
            markersize=6 * markersize / 20.0,
            linewidth=linewidth,
            zorder=zorder - 1,
        )

        # 6 yard box
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * box_length],
            [box_width / 2.0, box_width / 2.0],
            lc,
            linewidth=linewidth,
            zorder=zorder,
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * box_length],
            [-box_width / 2.0, -box_width / 2.0],
            lc,
            linewidth=linewidth,
            zorder=zorder,
        )
        ax.plot(
            [
                s * half_pitch_length - s * box_length,
                s * half_pitch_length - s * box_length,
            ],
            [-box_width / 2.0, box_width / 2.0],
            lc,
            linewidth=linewidth,
            zorder=zorder,
        )

        # Penalty area
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * area_length],
            [area_width / 2.0, area_width / 2.0],
            lc,
            linewidth=linewidth,
            zorder=zorder,
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * area_length],
            [-area_width / 2.0, -area_width / 2.0],
            lc,
            linewidth=linewidth,
            zorder=zorder,
        )
        ax.plot(
            [
                s * half_pitch_length - s * area_length,
                s * half_pitch_length - s * area_length,
            ],
            [-area_width / 2.0, area_width / 2.0],
            lc,
            linewidth=linewidth,
            zorder=zorder,
        )

        # Penalty spot
        ax.scatter(
            s * half_pitch_length - s * penalty_spot,
            0.0,
            marker="o",
            facecolor=lc,
            linewidth=0,
            s=markersize,
            zorder=zorder,
        )

        # Corner flags
        y = np.linspace(0, 1, 50) * corner_radius
        x = np.sqrt(corner_radius**2 - y**2)
        ax.plot(
            s * half_pitch_length - s * x,
            -half_pitch_width + y,
            lc,
            linewidth=linewidth,
            zorder=zorder,
        )
        ax.plot(
            s * half_pitch_length - s * x,
            half_pitch_width - y,
            lc,
            linewidth=linewidth,
            zorder=zorder,
        )

        # Draw the half circles by the box: the D
        y = (
            np.linspace(-1, 1, 50) * box_circle_length
        )  # D_length is the chord of the circle that defines the D
        x = np.sqrt(box_circle_radius**2 - y**2) + box_circle_pos
        ax.plot(s * half_pitch_length - s * x, y, lc, linewidth=linewidth, zorder=zorder)

    # remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    # set axis limits
    xmax = field_dimen[0] / 2.0 + border_dimen[0]
    ymax = field_dimen[1] / 2.0 + border_dimen[1]
    ax.set_xlim([-xmax, xmax])
    ax.set_ylim([-ymax, ymax])
    ax.set_axisbelow(True)

    return fig, ax


@logging_wrapper(__file__)
def plot_events(
    game: Game,
    events: list[str] = [],
    outcome: bool | None = None,
    player_ids: list[str] = [],
    team_id: int | str | None = None,
    fig: plt.figure = None,
    ax: plt.axes = None,
    color_by_col: str = None,
    team_colors: list[str] = ["orange", "red"],
    title: str | None = None,
) -> tuple[plt.figure, plt.axes]:
    """Function to plot the locations of specific events

    Args:
        game (Game): All information about a game
        events (list, optional): Filter of events you want to plot, if empty,
            all events are plotted. Defaults to [].
        outcome (bool, optional): Filter if the event should have a succesfull
            outcome (True) or not (False), if None, all outcomes are included.
            Defaults to None.
        player_ids (list, optional): Filter for what players to include, if empty,
            all players are included. Defaults to [].
        team_id (int | str, optional): Filter for what team to include, if None,
            both teams are included. Defaults to None.
        fig (plt.figure, optional): Figure to plot the events on. Defaults to None.
        ax (plt.axes, optional): Axes to plot the events on. Defaults to None.
        color_by_col (str, optional): If specified, colors of scatter is specified by
            this colom in game.event_data. Defaults to None.
        team_colors (list, optional): Colors by which the teams should be represented.
            Defaults to ["orange", "red"].
        title (str, optional): Title of the plot. Defaults to None.

    Returns:
        Tuple[plt.figure, plt.axes]: figure and axes with the pitch and events depicted
        on it.
    """

    event_data = game.event_data
    mask = pd.Series(True, index=event_data.index)
    if not isinstance(events, (list, np.ndarray)):
        message = f"'events' should be a list, not a {type(events)}"
        raise TypeError(message)
    if len(events) > 0:
        for wrong_event in [
            event
            for event in events
            if event not in event_data["databallpy_event"].unique()
        ]:
            raise ValueError(
                f"{wrong_event} is not a valid event in the databallpy_events, "
            )
        mask = (mask) & (event_data["databallpy_event"].isin(events))
    if outcome is not None:
        if outcome not in [False, True]:
            raise ValueError(
                f"'is_successful' should be either True or False, not {outcome}"
            )

        mask = (mask) & (event_data["is_successful"] == outcome)
    if len(player_ids) > 0:
        for wrong_id in [
            x for x in player_ids if x not in event_data["player_id"].unique()
        ]:
            raise ValueError(
                f"'{wrong_id}' is not found in event_data.player_id, can not "
                "show events."
            )
        mask = (mask) & (event_data["player_id"].isin(player_ids))
    if team_id:
        if team_id not in [game.home_team_id, game.away_team_id]:
            raise ValueError(
                f"'{team_id}' is not the id of either teams, can not plot events."
            )
        mask = mask & (event_data["team_id"] == team_id)

    event_data = event_data.loc[mask]
    if len(event_data) == 0:
        print(
            "No events could be found that game your" "requirements, please try again."
        )
        return None, None

    if fig is None and ax is None:
        fig, ax = plot_soccer_pitch(field_dimen=game.pitch_dimensions)
    if title:
        ax.set_title(title)

    # Set game name
    ax.text(
        game.pitch_dimensions[0] / -2.0 + 2,
        game.pitch_dimensions[1] / 2.0 + 1.0,
        game.home_team_name,
        fontsize=14,
        c=team_colors[0],
        zorder=2.5,
    )
    ax.text(
        game.pitch_dimensions[0] / 2.0 - 15,
        game.pitch_dimensions[1] / 2.0 + 1.0,
        game.away_team_name,
        fontsize=14,
        c=team_colors[1],
        zorder=2.5,
    )

    # Check if color_by_col is specified and is a valid column name
    if color_by_col:
        assert color_by_col in game.event_data.columns

        # Color events by team if the specified column is "team_id"
        if color_by_col == "team_id":
            for id, c, team_name in zip(
                [game.home_team_id, game.away_team_id],
                team_colors,
                [game.home_team_name, game.away_team_name],
            ):
                temp_events = event_data[event_data[color_by_col] == id]
                ax.scatter(
                    temp_events["start_x"],
                    temp_events["start_y"],
                    marker="x",
                    label=team_name,
                    c=c,
                    zorder=2.5,
                )

        # Color events by unique values in the specified column
        else:
            for value in event_data[color_by_col].unique():
                temp_events = event_data[event_data[color_by_col] == value]
                ax.scatter(
                    temp_events["start_x"],
                    temp_events["start_y"],
                    marker="x",
                    label=value,
                    zorder=2.5,
                )

        # Add legend to the plot
        ax.legend(loc="upper center")

    # If color_by_col is not specified, color events using default settings
    else:
        ax.scatter(event_data["start_x"], event_data["start_y"], marker="x", zorder=2.5)

    return fig, ax


@logging_wrapper(__file__)
def plot_tracking_data(
    game: Game,
    idx: int,
    team_colors: list[str] = ["green", "red"],
    *,
    fig: plt.figure = None,
    ax: plt.axes = None,
    title: str = None,
    events: list = [],
    variable_of_interest: any = None,
    add_player_possession: bool = False,
    add_velocities: bool = False,
    heatmap_overlay: np.ndarray | None = None,
    overlay_cmap: Colormap | str = "viridis",
) -> tuple[plt.figure, plt.axes]:
    """Function to plot the tracking data of a specific index in the
    game.tracking_data.

    Args:
        game (Game): Game with tracking data and other info of the game.
        idx (int): Index of the tracking data you want to plot.
        team_colors (list[str], optional): The color of the teams.
            Defaults to ["green", "red"].
        fig (plt.figure, optional): The figure to plot on. Defaults to None.
        ax (plt.axes, optional): The axes to plot on. Defaults to None.
        title (str, optional): The title of the plot. Defaults to None.
        events (list, optional): The databallpy events to plot. Defaults to [].
        variable_of_interest (any, optional): The variable you want to plot.
            Defaults to None.
        add_player_possession (bool, optional): Whether to add a circle around the
            player that has possession over the ball. Defaults to False.
        add_velocities (bool, optional): Whether to add the velocities to the plot.
            Defaults to False.
        heatmap_overlay (np.ndarray, optional): A heatmap to overlay on the pitch.
            Defaults to None.
        overlay_cmap (Colormap | str, optional): The colormap to use for the heatmap
            overlay. Defaults to "viridis".

    Returns:
        tuple[plt.figure, plt.axes]: The figure and axes with the tracking data
    """

    td = game.tracking_data.loc[[idx]]
    td_ht = td[
        np.array(
            [[x + "_x", x + "_y"] for x in game.get_column_ids(team="home")]
        ).reshape(1, -1)[0]
    ]
    td_at = td[
        np.array(
            [[x + "_x", x + "_y"] for x in game.get_column_ids(team="away")]
        ).reshape(1, -1)[0]
    ]

    _pre_check_plot_td_inputs(
        game=game,
        td=td,
        events=events,
        variable_of_interest=variable_of_interest,
        add_player_possession=add_player_possession,
        add_velocities=add_velocities,
        heatmap_overlay=heatmap_overlay,
        cmap=overlay_cmap,
    )

    if fig is None and ax is None:
        fig, ax = plot_soccer_pitch(field_dimen=game.pitch_dimensions)
    if title:
        ax.set_title(title)

    # Set game name
    ax.text(
        game.pitch_dimensions[0] / -2.0 + 2,
        game.pitch_dimensions[1] / 2.0 + 1.0,
        game.home_team_name,
        fontsize=14,
        c=team_colors[0],
        zorder=2.5,
    )
    ax.text(
        game.pitch_dimensions[0] / 2.0 - 15,
        game.pitch_dimensions[1] / 2.0 + 1.0,
        game.away_team_name,
        fontsize=14,
        c=team_colors[1],
        zorder=2.5,
    )

    if heatmap_overlay is not None:
        _, ax = _plot_heatmap_overlay(ax, heatmap_overlay, game, overlay_cmap, [])

    if add_velocities:
        _, ax = _plot_velocities(ax, td, idx, game, [])

    _, ax = _plot_single_frame(ax, td_ht, td_at, idx, team_colors, [], td, game)

    if variable_of_interest is not None:
        _, ax = _plot_variable_of_interest(ax, variable_of_interest, [], game)

    if add_player_possession:
        column_id = game.tracking_data.loc[idx, "player_possession"]
        _, ax = _plot_player_possession(ax, column_id, idx, game, [])

    if len(events) > 0 and td.loc[idx, "databallpy_event"] in events:
        _, ax = _plot_events(ax, td, idx, game, [])
    return fig, ax


@logging_wrapper(__file__)
@requires_ffmpeg
def save_tracking_video(
    game: Game,
    start_idx: int,
    end_idx: int,
    save_folder: str,
    *,
    title: str = "test_clip",
    team_colors: list = ["green", "red"],
    events: list = [],
    variable_of_interest: pd.Series | list = None,
    add_player_possession: bool = False,
    add_velocities: bool = False,
    heatmap_overlay: np.ndarray | None = None,
    overlay_cmap: Colormap | str = "viridis",
    verbose: bool = True,
):
    """Function to save a subset of a game clip of the tracking data.

    Note that making animation is build with FFMPEG. You need to have
    FFMPEG installed on you device before being able to use this function.

    Args:
        game (Game): Game with tracking data and ohter info of the game.
        start_idx (int): Start index of what to save of the game.tracking_data df.
        end_idx (int): End index of what to save of the game.tracking_data df.
        save_folder (str): Location where to save the clip.
        title (str, optional): Title of the clip. Defaults to "test_clip".
        team_colors (list, optional): Colors of the home and away team. Defaults to
            ["green", "red"].
        events (list, optional): What events should be plotted as well. Defaults to [].
        variable_of_interest (pd.Series | list, optional): Variable you want to have
            plotted in the clip, this is a pd.Series that should have the same index
            (start_idx:end_idx) as the tracking data that will be plotted. Defaults to
            None.
        add_player_possession (bool, optional): Whether to add a mark around which
            player has possession over the ball. Defaults to False. If True, the
            column 'player_possession' should be in the game.tracking_data.
        add_velocities (bool, optional): Whether or not to add the velocities to the
            clip, will add arrows to show the velocity. Defaults to False.
        heatmap_overlay (np.ndarray, optional): A heatmap to overlay on the pitch.
            Defaults to None.
        overlay_cmap (Colormap | str, optional): The colormap to use for the heatmap
            overlay. Defaults to "viridis".
        verbose (bool, optional): Whether or not to print info in the terminal on the
            progress
    """
    td = game.tracking_data.loc[start_idx:end_idx]
    td_ht = td[
        np.array(
            [[x + "_x", x + "_y"] for x in game.get_column_ids(team="home")]
        ).reshape(1, -1)[0]
    ]
    td_at = td[
        np.array(
            [[x + "_x", x + "_y"] for x in game.get_column_ids(team="away")]
        ).reshape(1, -1)[0]
    ]

    _pre_check_plot_td_inputs(
        game=game,
        td=td,
        events=events,
        variable_of_interest=variable_of_interest,
        add_player_possession=add_player_possession,
        add_velocities=add_velocities,
        heatmap_overlay=heatmap_overlay,
        cmap=overlay_cmap,
    )

    writer = animation.FFMpegWriter(
        fps=game.tracking_data.frame_rate,
        metadata={
            "title": title,
            "artist": "Matplotlib",
            "comment": "Made with DataballPy",
        },
    )
    video_loc = f"{save_folder}/{title}.mp4"

    pitch_color = "white" if heatmap_overlay is not None else "mediumseagreen"
    fig, ax = plot_soccer_pitch(
        field_dimen=game.pitch_dimensions, pitch_color=pitch_color
    )

    # Set game name, non variable over time
    ax.text(
        game.pitch_dimensions[0] / -2.0 + 2,
        game.pitch_dimensions[1] / 2.0 + 1.0,
        game.home_team_name,
        fontsize=14,
        color=team_colors[0],
    )
    ax.text(
        game.pitch_dimensions[0] / 2.0 - 15,
        game.pitch_dimensions[1] / 2.0 + 1.0,
        game.away_team_name,
        fontsize=14,
        color=team_colors[1],
    )

    indexes = (
        td.index if not verbose else tqdm(td.index, desc="Making game clip", leave=False)
    )
    # Generate movie with variable info
    with writer.saving(fig, video_loc, dpi=300):
        for idx_loc, idx in enumerate(indexes):
            variable_fig_objs = []

            if heatmap_overlay is not None:
                variable_fig_objs, ax = _plot_heatmap_overlay(
                    ax,
                    heatmap_overlay[idx_loc],
                    game,
                    overlay_cmap,
                    variable_fig_objs,
                )

            if add_velocities:
                variable_fig_objs, ax = _plot_velocities(
                    ax, td, idx, game, variable_fig_objs
                )

            variable_fig_objs, ax = _plot_single_frame(
                ax, td_ht, td_at, idx, team_colors, variable_fig_objs, td, game
            )

            if variable_of_interest is not None:
                value = (
                    variable_of_interest.loc[idx]
                    if isinstance(variable_of_interest, pd.Series)
                    else variable_of_interest[idx_loc]
                )
                variable_fig_objs, ax = _plot_variable_of_interest(
                    ax, value, variable_fig_objs, game
                )

            if add_player_possession:
                column_id = game.tracking_data.loc[idx, "player_possession"]
                variable_fig_objs, ax = _plot_player_possession(
                    ax, column_id, idx, game, variable_fig_objs
                )

            if len(events) > 0 and td.loc[idx, "databallpy_event"] in events:
                variable_fig_objs, ax = _plot_events(
                    ax, td, idx, game, variable_fig_objs
                )

                # 'pause' the clip for 1 second on this event
                [writer.grab_frame() for _ in range(game.tracking_data.frame_rate)]

            # Save current frame
            writer.grab_frame()

            # Delete all variable axis objects
            for fig_obj in variable_fig_objs:
                fig_obj.remove()

    plt.clf()
    plt.close(fig)


@logging_wrapper(__file__)
def _pre_check_plot_td_inputs(
    game: Game,
    td: pd.DataFrame,
    events: list[str],
    variable_of_interest: str | None,
    add_player_possession: bool,
    add_velocities: bool,
    heatmap_overlay: np.ndarray | None,
    cmap: Colormap | str | None,
):
    """Function to check if the inputs for the save_game_clip function are correct."""
    if not isinstance(game, Game):
        raise DataBallPyError("game should be an instance of databallpy.Game")

    if variable_of_interest is not None:
        if isinstance(variable_of_interest, pd.Series):
            if not (variable_of_interest.index == td.index).all():
                raise DataBallPyError(
                    "index of the variable_of_interest should be equal to the index "
                    "of the start_idx:end_idx."
                )
        elif isinstance(variable_of_interest, list):
            if len(variable_of_interest) != len(td):
                raise DataBallPyError(
                    "Length of variable_of_interest should be equal to the length of "
                    "the start_idx:end_idx."
                )
    if add_player_possession:
        if "player_possession" not in game.tracking_data.columns:
            raise DataBallPyError(
                "Column 'player_possession' not found in " "game.tracking_data.columns"
            )

    if len(events) > 0:
        if not game.is_synchronised:
            raise DataBallPyError("Game needs to be synchronised to add events.")

    if add_velocities:
        for player in game.get_column_ids() + ["ball"]:
            if player + "_vx" not in td.columns or player + "_vy" not in td.columns:
                raise DataBallPyError(
                    f"Player vx and/or vy of {player} not found in "
                    "game.tracking_data.columns. Please run "
                    "databallpy.features.differentiat.add_velocity() first."
                )

    if heatmap_overlay is not None:
        if not isinstance(heatmap_overlay, np.ndarray):
            raise DataBallPyError(
                "heatmap_overlay should be a numpy array."
                f" Type of heatmap_overlay: {type(heatmap_overlay)}"
            )
        if len(td) == 1 and len(heatmap_overlay.shape) != 2:
            raise DataBallPyError(
                "heatmap_overlay should be a 2D array."
                f" Heatmap overlay shape: {heatmap_overlay.shape}"
            )
        if len(td) > 1:
            if len(heatmap_overlay.shape) != 3:
                raise DataBallPyError(
                    "heatmap_overlay should be a 3D array."
                    f" Heatmap overlay shape: {heatmap_overlay.shape}"
                )
            if len(td) != heatmap_overlay.shape[0]:
                raise DataBallPyError(
                    "heatmap_overlay should have the same length as the tracking data."
                    f" Heatmap overlay length: {heatmap_overlay.shape[0]}, "
                    f"tracking data length: {len(td)}"
                )

        if not isinstance(cmap, Colormap) and cmap not in mpl.colormaps:
            raise DataBallPyError("cmap should be a matplotlib.colors.Colomap.")


def _plot_heatmap_overlay(
    ax: plt.axes,
    heatmap_overlay: np.ndarray,
    game: Game,
    cmap: Colormap,
    variable_fig_objs: list,
) -> tuple[plt.figure, plt.axes]:
    """Helper function to plot the heatmap overlay of the current frame."""
    fig_obj = ax.imshow(
        heatmap_overlay,
        extent=[
            -game.pitch_dimensions[0] / 2.0,
            game.pitch_dimensions[0] / 2.0,
            -game.pitch_dimensions[1] / 2.0,
            game.pitch_dimensions[1] / 2.0,
        ],
        origin="lower",
        cmap=cmap,
        alpha=0.5,
        zorder=-5,
    )

    variable_fig_objs.append(fig_obj)

    return variable_fig_objs, ax


def _plot_velocities(
    ax: plt.axes, td: pd.DataFrame, idx: int, game: Game, variable_fig_objs: list
) -> tuple[list, plt.axes]:
    """Helper function to plot the velocities of the current frame."""
    # Player velocities
    for col_id in game.get_column_ids():
        if pd.isnull(td.loc[idx, [col_id + "_vx", col_id + "_x"]]).any():
            continue

        arrow = mpatches.FancyArrowPatch(
            td.loc[idx, [f"{col_id}_x", f"{col_id}_y"]].values,
            td.loc[idx, [f"{col_id}_x", f"{col_id}_y"]].values
            + td.loc[idx, [f"{col_id}_vx", f"{col_id}_vy"]].values,
            mutation_scale=10,
        )
        fig_obj = ax.add_patch(arrow)
        variable_fig_objs.append(fig_obj)

    # Ball velocity
    if not pd.isnull(td.loc[idx, ["ball_vx", "ball_x"]]).any():
        arrow = mpatches.FancyArrowPatch(
            td.loc[idx, ["ball_x", "ball_y"]].values,
            td.loc[idx, ["ball_x", "ball_y"]].values
            + td.loc[idx, ["ball_vx", "ball_vy"]].values,
            mutation_scale=10,
            color="black",
        )
        fig_obj = ax.add_patch(arrow)
        variable_fig_objs.append(fig_obj)

    return variable_fig_objs, ax


def _plot_single_frame(
    ax: plt.axes,
    td_ht: pd.DataFrame,
    td_at: pd.DataFrame,
    idx: int,
    team_colors: list[str],
    variable_fig_objs: list,
    td: pd.DataFrame,
    game: Game,
) -> tuple[list, plt.axes]:
    """Helper function to plot the single frame of the current frame."""
    # Scatter plot the teams
    for td_team, c in zip([td_ht.loc[idx], td_at.loc[idx]], team_colors):
        x_cols = [x for x in td_team.index if x[-2:] == "_x"]
        y_cols = [y for y in td_team.index if y[-2:] == "_y"]
        fig_obj = ax.scatter(
            td_team[x_cols],
            td_team[y_cols],
            c=c,
            alpha=0.9,
            s=90,
            zorder=2.5,
        )
        variable_fig_objs.append(fig_obj)

        # Add shirt number to every dot
        for x, y in zip(x_cols, y_cols):
            if pd.isnull(td_team[x]):
                continue

            correction = 0.5 if len(x.split("_")[1]) == 1 else 0.8
            fig_obj = ax.text(
                td_team[x] - correction,
                td_team[y] - 0.5,
                x.split("_")[1],  # player number
                fontsize=9,
                c="white",
                zorder=3.0,
            )
            variable_fig_objs.append(fig_obj)

    # Plot the ball
    fig_obj = ax.scatter(td.loc[idx, "ball_x"], td.loc[idx, "ball_y"], c="black")
    variable_fig_objs.append(fig_obj)

    # Add time info
    fig_obj = ax.text(
        -20.5,
        game.pitch_dimensions[1] / 2.0 + 1.0,
        td.loc[idx, "gametime_td"],
        c="k",
        fontsize=14,
    )
    variable_fig_objs.append(fig_obj)

    return variable_fig_objs, ax


def _plot_variable_of_interest(
    ax: plt.axes,
    value: any,
    variable_fig_objs: list,
    game: Game,
) -> tuple[list, plt.axes]:
    """Helper function to plot the variable of interest of the current frame."""
    fig_obj = ax.text(
        -7,
        game.pitch_dimensions[1] / 2.0 + 1.0,
        str(value),
        fontsize=14,
    )
    variable_fig_objs.append(fig_obj)

    return variable_fig_objs, ax


def _plot_player_possession(
    ax: plt.axes, column_id: str, idx: int, game: Game, variable_fig_objs: list
) -> tuple[list, plt.axes]:
    """Helper function to plot the player possession of the current frame."""
    if pd.isnull(column_id) or pd.isnull(game.tracking_data.loc[idx, f"{column_id}_x"]):
        return variable_fig_objs, ax

    circle = plt.Circle(
        (
            game.tracking_data.loc[idx, f"{column_id}_x"],
            game.tracking_data.loc[idx, f"{column_id}_y"],
        ),
        radius=1,
        color="gold",
        fill=False,
    )
    fig_obj = ax.add_artist(circle)
    variable_fig_objs.append(fig_obj)

    return variable_fig_objs, ax


def _plot_events(
    ax: plt.axes, td: pd.DataFrame, idx: int, game: Game, variable_fig_objs: list
) -> tuple[list, plt.axes]:
    """Helper function to plot the events of the current frame."""
    event = (
        game.event_data[game.event_data["event_id"] == td.loc[idx, "event_id"]].iloc[0].T
    )
    player_name = event["player_name"]
    event_name = event["databallpy_event"]

    # Add event text
    fig_obj = ax.text(
        5,
        game.pitch_dimensions[1] / 2.0 + 1,
        f"{player_name}: {event_name}",
        fontsize=14,
    )
    variable_fig_objs.append(fig_obj)

    # Highligh location on the pitch of the event
    fig_obj = ax.scatter(
        event["start_x"],
        event["start_y"],
        color="red",
        marker="x",
        s=50,
    )
    variable_fig_objs.append(fig_obj)

    return variable_fig_objs, ax
