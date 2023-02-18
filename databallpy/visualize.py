import subprocess
from functools import wraps
from typing import Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from databallpy.match import Match


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
        except FileNotFoundError:
            raise FileNotFoundError(
                "It seems like ffmpeg is not added to your python path, please install\
                     add ffmpeg to you python path to use this code."
            )

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

    # Plot half way line
    ax.plot([0, 0], [-half_pitch_width, half_pitch_width], lc, linewidth=linewidth)
    ax.scatter(0.0, 0.0, marker="o", facecolor=lc, linewidth=0, s=markersize)
    # Plot center circle
    y = np.linspace(-1, 1, 150) * centre_circle_radius
    x = np.sqrt(centre_circle_radius**2 - y**2)
    ax.plot(x, y, lc, linewidth=linewidth)
    ax.plot(-x, y, lc, linewidth=linewidth)

    signs = [-1, 1]
    for s in signs:  # plots each line seperately

        # Plot pitch boundary
        ax.plot(
            [-half_pitch_length, half_pitch_length],
            [s * half_pitch_width, s * half_pitch_width],
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length],
            [-half_pitch_width, half_pitch_width],
            lc,
            linewidth=linewidth,
        )

        # Goal posts & line
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length],
            [-goal_line_width / 2.0, goal_line_width / 2.0],
            pc + "s",
            markersize=6 * markersize / 20.0,
            linewidth=linewidth,
        )

        # 6 yard box
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * box_length],
            [box_width / 2.0, box_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * box_length],
            [-box_width / 2.0, -box_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            [
                s * half_pitch_length - s * box_length,
                s * half_pitch_length - s * box_length,
            ],
            [-box_width / 2.0, box_width / 2.0],
            lc,
            linewidth=linewidth,
        )

        # Penalty area
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * area_length],
            [area_width / 2.0, area_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * area_length],
            [-area_width / 2.0, -area_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            [
                s * half_pitch_length - s * area_length,
                s * half_pitch_length - s * area_length,
            ],
            [-area_width / 2.0, area_width / 2.0],
            lc,
            linewidth=linewidth,
        )

        # Penalty spot
        ax.scatter(
            s * half_pitch_length - s * penalty_spot,
            0.0,
            marker="o",
            facecolor=lc,
            linewidth=0,
            s=markersize,
        )

        # Corner flags
        y = np.linspace(0, 1, 50) * corner_radius
        x = np.sqrt(corner_radius**2 - y**2)
        ax.plot(
            s * half_pitch_length - s * x,
            -half_pitch_width + y,
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            s * half_pitch_length - s * x, half_pitch_width - y, lc, linewidth=linewidth
        )

        # Draw the half circles by the box: the D
        y = (
            np.linspace(-1, 1, 50) * D_length
        )  # D_length is the chord of the circle that defines the D
        x = np.sqrt(D_radius**2 - y**2) + D_pos
        ax.plot(s * half_pitch_length - s * x, y, lc, linewidth=linewidth)

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
    event_data = match.event_data

    if len(events) > 0:
        assert all([x in match.event_data["event"].unique() for x in events])
        event_data = event_data.loc[event_data["event"].isin(events)]
    if outcome is not None:
        assert outcome in [0, 1]
        event_data = event_data.loc[event_data["outcome"] == outcome]
    if len(player_ids) > 0:
        assert all(x in match.event_data["player_id"].unique() for x in player_ids)
        event_data = event_data.loc[event_data["player_id"].isin(player_ids)]
    if team_id:
        assert team_id in [match.home_team_id, match.away_team_id]
        event_data = event_data.loc[event_data["team_id"] == team_id]

    if len(event_data) == 0:
        print(
            "No events could be found that match your\
            requirements, please try again."
        )
        return None, None
    else:
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
        ax.scatter(event_data["start_x"], event_data["start_y"], marker="x", zorder=2.5)

    return fig, ax


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
        (start_idx:end_idx) as the tracking data that will be plotted. Defaults to None.
    """

    td = match.tracking_data.loc[start_idx:end_idx]
    td_ht = td[[x for x in match.home_players_column_ids if "_x" in x or "_y" in x]]
    td_at = td[[x for x in match.away_players_column_ids if "_x" in x or "_y" in x]]

    if variable_of_interest is not None:
        assert (
            variable_of_interest.index == td.index
        ).all(), (
            "Index of variable of interest and of the tracking data should be alike!"
        )

    if len(events) > 0:
        assert (
            "event" in match.tracking_data.columns
        ), "No event column found in match.tracking_data.columns, did you synchronize\
            event and tracking data?"

    animation_metadata = {
        "title": title,
        "artist": "Matplotlib",
        "comment": "Made with databallpy",
    }
    writer = animation.FFMpegWriter(fps=match.frame_rate, metadata=animation_metadata)
    video_loc = f"{save_folder}/{title}.mp4"

    fig, ax = plot_soccer_pitch(
        field_dimen=match.pitch_dimensions, pitch_color=pitch_color
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
        print("Making match clip...")
        for _, idx in enumerate(tqdm(td.index)):

            variable_fig_objs = []

            # Scatter plot the teams
            for td_team, c in zip([td_ht.loc[idx], td_at.loc[idx]], team_colors):
                x_cols = [x for x in td_team.index if x[-2:] == "_x"]
                y_cols = [y for y in td_team.index if y[-2:] == "_y"]
                fig_obj = ax.scatter(
                    td_team[x_cols], td_team[y_cols], c=c, alpha=0.7, s=90, zorder=2.5
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

            # This code will only work after match has a synchronise() function
            # # Add events
            # # Note: this should be last in this function since it assumes that all
            # # other info is already plotted in the axes
            # if len(events) > 0:
            #     if td.loc[idx, "event"] in events:
            #         event = (
            #             match.event_data[
            #                 match.event_data["event_id"] == td.loc[idx, "event_id"]
            #             ]
            #             .iloc[0]
            #             .T
            #         )

            #         player_name = event["player_name"]
            #         event_name = event["event"]

            #         # Add event text
            #         fig_obj = ax.text(
            #             15,
            #             match.pitch_dimension[1] / 2.0 + 2.0,
            #             f"{player_name}: {event_name}",
            #             fontsize=14,
            #         )
            #         variable_fig_objs.append(fig_obj)

            #         # Highligh location on the pitch of the event
            #         fig_obj = ax.scatter(
            #             event["start_x"],
            #             event["start_y"],
            #             color="red",
            #             marker="x",
            #             markersize=14,
            #         )
            #         variable_fig_objs.append(fig_obj)

            #         # Grap frame match.frame_rate times to 'pause' the video at
            #         # this moment
            #         for _ in range(match.frame_rate):
            #             writer.grab_frame()

            # Save current frame
            writer.grab_frame()

            # Delete all variable axis objects
            for fig_obj in variable_fig_objs:
                fig_obj.remove()

    # Close figure
    plt.clf()
    plt.close(fig)
