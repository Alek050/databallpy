import numpy as np
import pandas as pd

from databallpy.utils.constants import MISSING_INT


def _get_period_start_frames(periods_frames: pd.DataFrame) -> set[int]:
    """Function to obtain the start frames of all periods of the game.

    These frames are needed to determine the playing direction of the teams, and
    are therefore always loaded, also when they fall outside of the selection.

    Args:
        periods_frames (pd.DataFrame): the start and end frames of the periods

    Returns:
        set[int]: the start frames of all periods that are played
    """
    return set(
        periods_frames.loc[
            periods_frames["start_frame"] != MISSING_INT, "start_frame"
        ].to_list()
    )


def _get_frame_selection(
    periods_frames: pd.DataFrame,
    period_id: int | list[int] | None,
    frames: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """Function to obtain the range of frames that should be loaded.

    Args:
        periods_frames (pd.DataFrame): the start and end frames of the periods
        period_id (int | list[int] | None): the period(s) to load
        frames (tuple[int, int] | None): the first and last frame to load, inclusive

    Raises:
        TypeError: if period_id or frames are of the wrong type
        ValueError: if period_id or frames have invalid values, or if the
            combination of both does not contain any frame

    Returns:
        tuple[int, int] | None: the first and last frame to load, inclusive.
            None if all frames should be loaded.
    """
    if period_id is None and frames is None:
        return None

    first_frame = periods_frames.loc[
        periods_frames["start_frame"] != MISSING_INT, "start_frame"
    ].min()
    last_frame = periods_frames.loc[
        periods_frames["end_frame"] != MISSING_INT, "end_frame"
    ].max()

    if period_id is not None:
        period_ids = (
            [period_id] if isinstance(period_id, (int, np.integer)) else period_id
        )
        if not isinstance(period_ids, list) or not all(
            isinstance(x, (int, np.integer)) for x in period_ids
        ):
            raise TypeError(
                f"period_id should be an int or a list of ints, not {type(period_id)}"
            )
        available = periods_frames.loc[
            periods_frames["start_frame"] != MISSING_INT, "period_id"
        ].to_list()
        unknown = [x for x in period_ids if x not in available]
        if len(unknown) > 0:
            raise ValueError(
                f"Period(s) {unknown} are not available in the tracking data,"
                f" available periods are {available}"
            )
        selected = periods_frames[periods_frames["period_id"].isin(period_ids)]
        first_frame = selected["start_frame"].min()
        last_frame = selected["end_frame"].max()

    if frames is not None:
        if not isinstance(frames, (tuple, list)) or len(frames) != 2:
            raise TypeError(
                "frames should be a tuple with the first and last frame to load,"
                f" not {frames}"
            )
        if not all(isinstance(x, (int, np.integer)) for x in frames):
            raise TypeError(f"frames should contain two ints, not {frames}")
        if frames[0] > frames[1]:
            raise ValueError(
                f"The first frame ({frames[0]}) should not be later than the last"
                f" frame ({frames[1]})"
            )
        first_frame = max(first_frame, frames[0])
        last_frame = min(last_frame, frames[1])

    if first_frame > last_frame:
        raise ValueError(
            "No frames left to load, please check the period_id and frames arguments"
        )

    return int(first_frame), int(last_frame)
