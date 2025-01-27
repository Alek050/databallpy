import pandas as pd

from databallpy.data_parsers.metadata import Metadata
from databallpy.utils.constants import MISSING_INT
from databallpy.utils.errors import DataBallPyError


def _adjust_start_end_frames(td: pd.DataFrame, metadata: Metadata) -> Metadata:
    """Function to check if the start and end frames as proposed by the metadata
    correspond with the tracking data. If not, tracking data frames are added to
    the metadata to make them correspond.

    Args:
        td (pd.DataFrame): tracking data of the game
        metadata (Metadata): metadata of the game

    Returns:
        Metadata: metadata of the game, may be with adjusted start and end frames
    """

    home_players_x_columns = [
        x for x in td.columns if x[:4] == "home" and x[-2:] == "_x"
    ]
    away_players_x_columns = [
        x for x in td.columns if x[:4] == "away" and x[-2:] == "_x"
    ]

    for i, period_row in metadata.periods_frames.iterrows():
        if period_row["start_frame"] == MISSING_INT:
            continue

        # first check if proposes frames could be right, if so assume it is right
        first_frame = td[td["frame"] == period_row["start_frame"]]
        # for the first frame, ball should be in the centre and players should be
        # on opposing halves
        if len(first_frame) == 1:
            first_frame = first_frame.iloc[0]
            ball_mask = (first_frame[["ball_x", "ball_y"]].abs() < 10).all()
            home_players_x = first_frame[home_players_x_columns].mean()
            away_players_x = first_frame[away_players_x_columns].mean()
            players_mask = (away_players_x < 0 and home_players_x > 0) or (
                away_players_x > 0 and home_players_x < 0
            )
            adjust_start_frame = False if (players_mask & ball_mask) else True
        else:
            adjust_start_frame = True

        end_frame = td[td["frame"] == period_row["end_frame"]]
        adjust_end_frame = False if len(end_frame) == 1 else True

        if adjust_start_frame:
            new_start_frame = _find_new_start_frame(
                td,
                period_row["period_id"],
                home_players_x_columns,
                away_players_x_columns,
                metadata.frame_rate,
            )

            new_start_dt = td.loc[td["frame"] == new_start_frame, "datetime"].iloc[0]
            metadata.periods_frames.loc[i, "start_datetime_td"] = new_start_dt
            metadata.periods_frames.loc[i, "start_frame"] = new_start_frame

            mask_to_del = (td["frame"] < new_start_frame) & (
                td["period_id"] == period_row["period_id"]
            )
            td.loc[mask_to_del, "period_id"] = MISSING_INT
            mask_to_add = (
                (td["frame"] >= new_start_frame)
                & (td["period_id"] < period_row["period_id"] + 1)
                & (td["period_id"] != MISSING_INT)
            )
            td.loc[mask_to_add, "period_id"] = period_row["period_id"]

        if adjust_end_frame:
            # this can only shorten the periods when tracking data stops earlier than
            # metadata implies that the game ends.
            new_end_frame = td.loc[td["period_id"] == period_row["period_id"]].iloc[-1][
                "frame"
            ]
            metadata.periods_frames.loc[i, "end_frame"] = new_end_frame
            new_end_dt = td.loc[td["frame"] == new_end_frame, "datetime"].iloc[0]
            metadata.periods_frames.loc[i, "end_datetime_td"] = new_end_dt

    # if game starts later than indicated in period one, delete all unescesary rows
    mask = (
        td["frame"]
        < metadata.periods_frames.loc[
            metadata.periods_frames["period_id"] == 1, "start_frame"
        ].iloc[0]
    )
    new_tracking_data = td[~mask]
    new_tracking_data.reset_index(drop=True, inplace=True)
    return new_tracking_data, metadata


def _find_new_start_frame(
    td: pd.DataFrame,
    period_id: int,
    home_players_x_columns: list,
    away_players_x_columns: list,
    frame_rate: int,
) -> int:
    """Function to find the new start frame of a period. This is done by looking at
    the first frame of the period and checking if the ball is in the centre of the
    pitch and if the home players are on the other side of the away players. If not,
    the first frame is not the start of the period and the start frame is adjusted
    accordingly. If the first frame is not the start of the period, the start frame
    is adjusted by looking at the frames where the above mentioned conditions are met.
    Next, we use the ball acceleration. The first frame where the ball acceleration is
    above 12.5 m/s/s is assumed to be the start of the period.

    Args:
        td (pd.DataFrame): tracking data of the game
        period_id (int): period id of the period for which the start frame should be
            found
        home_players_x_columns (list): columns of the x positions of the home players
        away_players_x_columns (list): columns of the x positions of the away players
        frame_rate (int): frame rate of the tracking data

    Raises:
        DataBallPyError: if no tracking data is found for the period

    Returns:
        int: start frame of the period, potentially adjusted.
    """

    # create window arround first time period id is initiated:
    first_period_td = td[td["period_id"] == period_id]

    if len(first_period_td) == 0:
        raise DataBallPyError(
            f"Something went wrong when parsing period {period_id}. No tracking data"
            "was found for this period. Check your raw tracking and tracking metadata"
            "for unlogical values."
        )

    first_period_idx = first_period_td.index[0]

    first_window_idx = max(td.index[0], first_period_idx - (7 * frame_rate))
    last_window_idx = min(td.index[-1], first_period_idx + (7 * frame_rate))

    td_window = td.loc[first_window_idx:last_window_idx]

    # ball should be in/close to the centre of the pitch
    ball_mask = (td_window[["ball_x", "ball_y"]].abs() < 7).all(axis=1)
    ball_x_acc = td_window["ball_x"].diff().diff().abs()

    # home players should be on the other side of the away players
    home_x = td_window[home_players_x_columns].mean(axis=1)
    away_x = td_window[away_players_x_columns].mean(axis=1)
    players_mask = ((away_x < 0) & (home_x > 0)) | ((away_x > 0) & (home_x < 0))
    # 8 columns are never null, we expect at least 19 players, and 3 ball coordinates
    # added, thus 8+38+3 = 48 columns should be non-null
    non_null_mask = td_window.count(axis=1) > 48

    valid_options = td_window.loc[ball_mask & players_mask & non_null_mask]

    # select the optimal start frame
    if len(valid_options) == 1:
        return td.loc[valid_options.index[0], "frame"]
    elif len(valid_options) == 0:
        return td.loc[first_period_idx, "frame"]
    else:
        # find the index where the ball acceleration is maximal, probably the moment
        # of the first pass
        valid_ball_acc = ball_x_acc.loc[valid_options.index]
        acc_mask = valid_ball_acc > (12.5 / frame_rate)
        new_idx = (
            valid_options[acc_mask].index[0]
            if len(valid_options[acc_mask]) > 0
            else valid_options.index[0]
        )
        return td.loc[new_idx, "frame"]
