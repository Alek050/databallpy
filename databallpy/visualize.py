from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


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
        Tuple[plt.fig, plt.axes]: figure and axes with the pitch depicted on it
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
