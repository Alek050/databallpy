import subprocess
import time
import warnings
from functools import wraps
from typing import Tuple

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from tqdm import tqdm

from databallpy.features.pitch_control import get_pitch_control_period
from databallpy.match import Match
from databallpy.utils.errors import DataBallPyError
from databallpy.utils.logging import create_logger
from databallpy.utils.warnings import DataBallPyWarning

LOGGER = create_logger(__name__)


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
        except Exception as e:
            LOGGER.critical(
                "Could not find the subprocess ffmpeg. Make sure ffmpeg is installed"
                " globally on you device and added to your python path."
            )
            raise e

        return func(*args, **kwargs)

    return wrapper


def plot_soccer_pitch(
    field_dimen: tuple = (106.0, 68.0),
    pitch_color: str = "mediumseagreen",
    linewidth: int = 2,
    markersize: int = 20,
) -> Tuple[plt.figure, plt.axes]:
    """A function to plot a soccer pitch
    Note: relies heavily on https://github.com/Friends-of-Tracking-Data-FoTD/
    LaurieOnTracking/blob/master/Metrica_Viz.py

    Args:
        field_dimen (tuple, optional): x and y length of pitch in meters. Defaults to
        (106.0, 68.0).
        pitch_color (str, optional): Color of the pitch. Defaults to "mediumseagreen".
        linewidth (int, optional): Width of the lines on the pitch. Defaults to 2.
        markersize (int, optional): Size of the dots on the pitch. Defaults to 20.

    Returns:
        Tuple[plt.figure, plt.axes]: figure and axes with the pitch depicted on it
    """
    LOGGER.info("Trying to plot soccer pitch")
    try:
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
        D_length = 8 * meters_per_yard
        D_radius = 10 * meters_per_yard
        D_pos = 12 * meters_per_yard
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
                np.linspace(-1, 1, 50) * D_length
            )  # D_length is the chord of the circle that defines the D
            x = np.sqrt(D_radius**2 - y**2) + D_pos
            ax.plot(
                s * half_pitch_length - s * x, y, lc, linewidth=linewidth, zorder=zorder
            )

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

        LOGGER.info("Succesfully plotted pitch")
        return fig, ax
    except Exception as e:
        LOGGER.exception(f"Found an unexpected exception in plot_soccer_pitch(): \n{e}")
        raise e


def plot_events(
    match: Match,
    events: list = [],
    outcome: int = None,
    player_ids: list = [],
    team_id: int = None,
    pitch_color: str = "mediumseagreen",
    color_by_col: str = None,
    team_colors: list = ["orange", "red"],
    title: str = None,
) -> Tuple[plt.figure, plt.axes]:
    """Function to plot the locations of specific events

    Args:
        match (Match): All information about a match
        events (list, optional): Filter of events you want to plot, if empty,
        all events are plotted. Defaults to [].
        outcome (int, optional): Filter if the event should have a succesfull
        outcome (1) or not (0), if None, all outcomes are included. Defaults to None.
        player_ids (list, optional): Filter for what players to include, if empty, all
        players are included. Defaults to [].
        team_id (int, optional): Filter for what team to include, if None, both teams
        are included. Defaults to None.
        pitch_color (str, optional): Prefered color of the pitch. Defaults to
        "mediumseagreen".
        color_by_col (str, optional): If specified, colors of scatter is specified by
        this colom in match.event_data. Defaults to None.
        team_colors (list, optional): Colors by which the teams should be represented.
        Defaults to ["orange", "red"].
        title (str, optional): Title of the plot. Defaults to None.

    Returns:
        Tuple[plt.figure, plt.axes]: figure and axes with the pitch and events depicted
        on it.
    """
    LOGGER.info(f"Trying to plot {events} in plot_events().")
    try:
        event_data = match.event_data
        mask = pd.Series(True, index=event_data.index)
        if not isinstance(events, (list, np.ndarray)):
            message = f"'events' should be a list, not a {type(events)}"
            LOGGER.error(message)
            raise TypeError(message)
        if len(events) > 0:
            for wrong_event in [
                event
                for event in events
                if event not in event_data["databallpy_event"].unique()
            ]:
                message = (
                    f"{wrong_event} is not a valid event in the databallpy_events, "
                )
                f"choose from {event_data['databallpy_event'].unique()}"
                LOGGER.error(message)
                raise ValueError(message)
            mask = (mask) & (event_data["databallpy_event"].isin(events))
        if outcome is not None:
            if outcome not in [0, 1]:
                message = f"'outcome' should be either 0 or 1, not {outcome}"
                LOGGER.error(message)
                raise ValueError(message)
            mask = (mask) & (event_data["outcome"] == outcome)
        if len(player_ids) > 0:
            for wrong_id in [
                x for x in player_ids if x not in event_data["player_id"].unique()
            ]:
                message = (
                    f"'{wrong_id}' is not found in event_data.player_id, can not "
                    "show events."
                )
                LOGGER.error(message)
                raise ValueError(message)
            mask = (mask) & (event_data["player_id"].isin(player_ids))
        if team_id:
            if team_id not in [match.home_team_id, match.away_team_id]:
                message = (
                    f"'{team_id}' is not the id of either teams, can not plot events."
                )
                LOGGER.error(message)
                raise ValueError(message)
            mask = mask & (event_data["team_id"] == team_id)

        event_data = event_data.loc[mask]
        if len(event_data) == 0:
            LOGGER.info(
                "No matching events were found, returning None in plot_events()."
            )
            print(
                "No events could be found that match your"
                "requirements, please try again."
            )
            return None, None
        else:
            LOGGER.info(f"Found {len(event_data)} matching events in plot_events().")
            print(f"Found {len(event_data)} matching events")

        fig, ax = plot_soccer_pitch(
            field_dimen=match.pitch_dimensions, pitch_color=pitch_color
        )
        if title:
            ax.set_title(title)

        # Set match name
        ax.text(
            match.pitch_dimensions[0] / -2.0 + 2,
            match.pitch_dimensions[1] / 2.0 + 1.0,
            match.home_team_name,
            fontsize=14,
            c=team_colors[0],
            zorder=2.5,
        )
        ax.text(
            match.pitch_dimensions[0] / 2.0 - 15,
            match.pitch_dimensions[1] / 2.0 + 1.0,
            match.away_team_name,
            fontsize=14,
            c=team_colors[1],
            zorder=2.5,
        )

        # Check if color_by_col is specified and is a valid column name
        if color_by_col:
            assert color_by_col in match.event_data.columns

            # Color events by team if the specified column is "team_id"
            if color_by_col == "team_id":
                for id, c, team_name in zip(
                    [match.home_team_id, match.away_team_id],
                    team_colors,
                    [match.home_team_name, match.away_team_name],
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
            plt.legend(loc="upper center")

        # If color_by_col is not specified, color events using default settings
        else:
            ax.scatter(
                event_data["start_x"], event_data["start_y"], marker="x", zorder=2.5
            )

        LOGGER.info("Successfully plotted events in plot_events().")
        return fig, ax
    except Exception as e:
        LOGGER.exception(f"Found unexpected exception in plot_events(): \n{e}")
        raise e


@requires_ffmpeg
def save_match_clip(
    match: Match,
    start_idx: int,
    end_idx: int,
    save_folder: str,
    *,
    title: str = "test_clip",
    team_colors: list = ["green", "red"],
    pitch_color: str = "mediumseagreen",
    events: list = [],
    variable_of_interest: pd.Series = None,
    player_possession_column: str = None,
    add_velocities: bool = False,
    add_pitch_control: bool = False,
    verbose: bool = True,
):
    """Function to save a subset of a match clip of the tracking data.

    Note that making animation is build with FFMPEG. You need to have FFMPEG installed
    on you device before being able to use this function.

    Args:
        match (Match): Match with tracking data and ohter info of the match.
        start_idx (int): Start index of what to save of the match.tracking_data df.
        end_idx (int): End index of what to save of the match.tracking_data df.
        save_folder (str): Location where to save the clip.
        title (str, optional): Title of the clip. Defaults to "test_clip".
        team_colors (list, optional): Colors of the home and away team. Defaults to
            ["green", "red"].
        pitch_color (str, optional): Color of the pitch. Defaults to "mediumseagreen".
        events (list, optional): What events should be plotted as well. Defaults to [].
        variable_of_interest (pd.Series, optional): Variable you want to have plotted
            in the clip, this is a pd.Series that should have the same index
            (start_idx:end_idx) as the tracking data that will be plotted. Defaults to
            None.
        player_possession_column (str, optional): Column in match.tracking_data that
            contains the player id of the player that has possession of the ball.
            Defaults to None.
        add_velocities (bool, optional): Whether or not to add the velocities to the
            clip, will add arrows to show the velocity. Defaults to False.
        add_pitch_control (bool, optional): Whether or not to add pitch control to the
            clip, will add a heatmap to show the pitch control. Defaults to False.
        verbose (bool, optional): Whether or not to print info in the terminal on the
            progress
    """
    LOGGER.info(f"Trying to create save a match clip '{title}' in save_match_clip().")
    try:
        td = match.tracking_data.loc[start_idx:end_idx]
        td_ht = td[
            np.array(
                [[x + "_x", x + "_y"] for x in match.home_players_column_ids()]
            ).reshape(1, -1)[0]
        ]
        td_at = td[
            np.array(
                [[x + "_x", x + "_y"] for x in match.away_players_column_ids()]
            ).reshape(1, -1)[0]
        ]
        if variable_of_interest is not None:
            if not (variable_of_interest.index == td.index).all():
                message = (
                    "index of the variable_of_interest should be equal to the index "
                    "of the start_idx:end_idx."
                )
                LOGGER.error(message)
                raise DataBallPyError(message)
        if player_possession_column is not None:
            if player_possession_column not in match.tracking_data.columns:
                message = (
                    f"Column {player_possession_column} not found in "
                    "match.tracking_data.columns"
                )
                LOGGER.error(message)
                raise DataBallPyError(message)

        if len(events) > 0:
            if not match.is_synchronised:
                message = "Match needs to be synchronised to add events."
                LOGGER.error(message)
                raise DataBallPyError(message)

        if add_velocities or add_pitch_control:
            for player in (
                match.home_players_column_ids() + match.away_players_column_ids()
            ):
                if player + "_vx" not in td.columns or player + "_vy" not in td.columns:
                    message = (
                        f"Player vx and/or vy of {player} not found in "
                        "match.tracking_data.columns. Please run "
                        "databallpy.features.differentiat.add_velocity() first."
                    )
                    LOGGER.error(message)
                    raise DataBallPyError(message)

        animation_metadata = {
            "title": title,
            "artist": "Matplotlib",
            "comment": "Made with DataballPy",
        }
        writer = animation.FFMpegWriter(
            fps=match.frame_rate, metadata=animation_metadata
        )
        video_loc = f"{save_folder}/{title}.mp4"

        if pitch_color not in ["white", "w"] and add_pitch_control:
            message = (
                "Pitch control will not be shown properly if the pitch color"
                " is not white. Changing the pitch color to white."
            )
            LOGGER.warning(message)
            warnings.warn(message, category=DataBallPyWarning)
            pitch_color = "white"

        fig, ax = plot_soccer_pitch(
            field_dimen=match.pitch_dimensions, pitch_color=pitch_color
        )

        if add_pitch_control:
            grid_size = [360, 240]
            x_range = np.linspace(
                -match.pitch_dimensions[0] / 2 - 5,
                match.pitch_dimensions[0] / 2 + 5,
                grid_size[0],
            )
            y_range = np.linspace(
                -match.pitch_dimensions[1] / 2 - 5,
                match.pitch_dimensions[1] / 2 + 5,
                grid_size[1],
            )
            grid = np.meshgrid(x_range, y_range)

            c3 = to_rgba(team_colors[0])
            c2 = (1, 1, 1, 1)
            c1 = to_rgba(team_colors[1])

            all_colors = [c1, c2, c3]

            cmap = LinearSegmentedColormap.from_list(
                "custom_colormap", all_colors, N=22
            )

            start_time = time.time()
            pitch_control_array = get_pitch_control_period(
                tracking_data=td,
                grid=grid,
            )
            n_cells = grid_size[0] * grid_size[1]
            pitch_control_array = (
                1 / (1 + np.exp(-n_cells / 50.0 * pitch_control_array))
            ) * 2 - 1
            LOGGER.info(
                f"Pitch control took {time.time() - start_time} "
                f"seconds, len(td) = {len(td)}"
            )

        # Set match name, non variable over time
        ax.text(
            match.pitch_dimensions[0] / -2.0 + 2,
            match.pitch_dimensions[1] / 2.0 + 1.0,
            match.home_team_name,
            fontsize=14,
            color=team_colors[0],
        )
        ax.text(
            match.pitch_dimensions[0] / 2.0 - 15,
            match.pitch_dimensions[1] / 2.0 + 1.0,
            match.away_team_name,
            fontsize=14,
            color=team_colors[1],
        )

        # Generate movie with variable info
        with writer.saving(fig, video_loc, 100):
            if verbose:
                indexes = tqdm(td.index, desc="Making match clip", leave=False)
            else:
                indexes = td.index
            for i, idx in enumerate(indexes):
                variable_fig_objs = []

                if add_pitch_control:
                    pitch_control = pitch_control_array[i]
                    imshow = ax.imshow(
                        pitch_control,
                        extent=[
                            x_range.min(),
                            x_range.max(),
                            y_range.min(),
                            y_range.max(),
                        ],
                        cmap=cmap,
                        origin="lower",
                        aspect="auto",
                        vmin=-1.0,
                        vmax=1.0,
                        zorder=-5,
                    )
                    variable_fig_objs.append(imshow)

                if add_velocities:
                    for col_ids in [
                        match.home_players_column_ids(),
                        match.away_players_column_ids(),
                    ]:
                        for col_id in col_ids:
                            if pd.isnull(td.loc[idx, col_id + "_vx"]):
                                continue

                            arrow = mpatches.FancyArrowPatch(
                                td.loc[idx, [f"{col_id}_x", f"{col_id}_y"]].values,
                                td.loc[idx, [f"{col_id}_x", f"{col_id}_y"]].values
                                + td.loc[idx, [f"{col_id}_vx", f"{col_id}_vy"]].values,
                                mutation_scale=10,
                            )
                            fig_obj = ax.add_patch(arrow)
                            variable_fig_objs.append(fig_obj)

                # Scatter plot the teams
                for td_team, c in zip([td_ht.loc[idx], td_at.loc[idx]], team_colors):
                    x_cols = [x for x in td_team.index if x[-2:] == "_x"]
                    y_cols = [y for y in td_team.index if y[-2:] == "_y"]
                    fig_obj = ax.scatter(
                        td_team[x_cols],
                        td_team[y_cols],
                        c=c,
                        alpha=0.7,
                        s=90,
                        zorder=2.5,
                    )
                    variable_fig_objs.append(fig_obj)

                    # Add shirt number to every dot
                    for x, y in zip(x_cols, y_cols):
                        if pd.isnull(td_team[x]):
                            # Player not on the pitch currently
                            continue

                        # Slightly different place needed if the number has a len of 2
                        correction = 0.5 if len(x.split("_")[1]) == 1 else 0.8
                        fig_obj = ax.text(
                            td_team[x] - correction,
                            td_team[y] - 0.5,
                            x.split("_")[1],  # the shirt number
                            fontsize=9,
                            c="white",
                            zorder=3.0,
                        )
                        variable_fig_objs.append(fig_obj)

                # Plot the ball
                fig_obj = ax.scatter(
                    td.loc[idx, "ball_x"], td.loc[idx, "ball_y"], c="black"
                )
                variable_fig_objs.append(fig_obj)

                # Add time info
                fig_obj = ax.text(
                    -20.5,
                    match.pitch_dimensions[1] / 2.0 + 1.0,
                    td.loc[idx, "matchtime_td"],
                    c="k",
                    fontsize=14,
                )
                variable_fig_objs.append(fig_obj)

                # Add variable of interest
                if variable_of_interest is not None:
                    fig_obj = ax.text(
                        -7,
                        match.pitch_dimensions[1] / 2.0 + 1.0,
                        str(variable_of_interest[idx]),
                        fontsize=14,
                    )
                    variable_fig_objs.append(fig_obj)

                # add player possessions
                if player_possession_column is not None:
                    column_id = match.tracking_data.loc[idx, player_possession_column]
                    if not pd.isnull(column_id):
                        circle = plt.Circle(
                            (
                                match.tracking_data.loc[idx, f"{column_id}_x"],
                                match.tracking_data.loc[idx, f"{column_id}_y"],
                            ),
                            radius=1,
                            color="gold",
                            fill=False,
                        )
                        fig_obj = ax.add_artist(circle)
                        variable_fig_objs.append(fig_obj)

                # Add events
                # Note: this should be last in this function since it assumes that all
                # other info is already plotted in the axes
                if len(events) > 0:
                    if td.loc[idx, "databallpy_event"] in events:
                        event = (
                            match.event_data[
                                match.event_data["event_id"] == td.loc[idx, "event_id"]
                            ]
                            .iloc[0]
                            .T
                        )

                        player_name = event["player_name"]
                        event_name = event["databallpy_event"]

                        # Add event text
                        fig_obj = ax.text(
                            5,
                            match.pitch_dimensions[1] / 2.0 + 1,
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

                        # Grap frame match.frame_rate times to 'pause' the video at
                        # this moment
                        for _ in range(match.frame_rate):
                            writer.grab_frame()

                # Save current frame
                writer.grab_frame()

                # Delete all variable axis objects
                for fig_obj in variable_fig_objs:
                    fig_obj.remove()

        # Close figure
        plt.clf()
        plt.close(fig)
        LOGGER.info(
            f"Succesfully created new saved match clip: {title} in {save_folder}."
        )
    except Exception as e:
        LOGGER.exception(f"Found an unexpected exception in save_match_clip(): \n{e}")
        raise e
