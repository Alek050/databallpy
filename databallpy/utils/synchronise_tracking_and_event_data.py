from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from databallpy.features.angle import get_smallest_angle
from databallpy.features.differentiate import _differentiate
from databallpy.utils.utils import MISSING_INT


def synchronise_tracking_and_event_data(
    match, n_batches_per_half: int = 100, verbose: bool = True
):
    """Function that synchronises tracking and event data using Needleman-Wunsch
       algorithmn. Based on: https://kwiatkowski.io/sync.soccer. The similarity
       function is used but information based on the specific event is added. For
       instance, when the event is a shot, the ball acceleration should be high and
       the ball should move towards the goal.

    Args:
        match (Match): Match object
        n_batches_per_half (int): the number of batches that are created per half.
            A higher number of batches reduces the time the code takes to load, but
            reduces the accuracy for events close to the splits. Default = 100
        verbose (bool, optional): Wheter or not to print info about the progress
            in the terminal. Defaults to True.

    Currently works for the following databallpy_events:
        'pass', 'shot', and 'dribble'
    """

    events_to_sync = [
        "pass",
        "shot",
        "dribble",
    ]

    tracking_data = match.tracking_data
    event_data = match.event_data

    tracking_data = pre_compute_synchronisation_variables(
        tracking_data, match.frame_rate, match.pitch_dimensions, match.periods
    )
    event_data["tracking_frame"] = MISSING_INT

    mask_events_to_sync = event_data["databallpy_event"].isin(events_to_sync)
    event_data_to_sync = event_data[mask_events_to_sync].copy()

    periods_played = match.periods[match.periods["start_frame"] > 0]["period"].values
    align_datetimes = not (
        match.tracking_timestamp_is_precise & match.event_timestamp_is_precise
    )
    for period_id in periods_played:
        (
            end_datetimes,
            tracking_data_full_period,
            event_data_full_period,
        ) = create_batches(
            period_id,
            match.frame_rate,
            n_batches_per_half,
            tracking_data,
            event_data_to_sync,
            align_datetimes,
        )

        if verbose:
            print(f"Syncing period {period_id}...")
            end_datetimes = tqdm(end_datetimes)

        batch_first_datetime = tracking_data_full_period["datetime"].iloc[0]

        # loop over batches
        for batch_end_datetime in end_datetimes:
            # create batches
            event_mask = event_data_full_period["datetime"].between(
                batch_first_datetime, batch_end_datetime, inclusive="left"
            )
            tracking_mask = tracking_data_full_period["datetime"].between(
                batch_first_datetime, batch_end_datetime, inclusive="left"
            )
            tracking_batch = tracking_data_full_period[tracking_mask].reset_index(
                drop=False
            )
            event_batch = event_data_full_period[event_mask].reset_index(drop=False)

            if len(event_batch) > 0:
                sim_mat = _create_sim_mat(tracking_batch, event_batch, match)
                event_frame_dict = _needleman_wunsch(sim_mat)
                # assign events to tracking data frames
                for event, frame in event_frame_dict.items():
                    event_id = int(event_batch.loc[event, "event_id"])
                    event_type = event_batch.loc[event, "databallpy_event"]
                    event_index = int(event_batch.loc[event, "index"])
                    tracking_frame = int(tracking_batch.loc[frame, "index"])
                    tracking_data.loc[tracking_frame, "databallpy_event"] = event_type
                    tracking_data.loc[tracking_frame, "event_id"] = event_id
                    event_data.loc[event_index, "tracking_frame"] = tracking_frame

            batch_first_datetime = batch_end_datetime

    tracking_data.drop(
        [
            "ball_acceleration_sqrt",
            "goal_angle_home_team",
            "goal_angle_away_team",
        ],
        axis=1,
        inplace=True,
    )

    match.tracking_data = tracking_data
    match.event_data = event_data

    match._is_synchronised = True


def _create_sim_mat(
    tracking_batch: pd.DataFrame, event_batch: pd.DataFrame, match
) -> np.ndarray:
    """Function that creates similarity matrix between every frame and event in batch

    Args:
        tracking_batch (pd.DataFrame): batch of tracking data
        event_batch (pd.DataFrame): batch of event data
        match (Match): Match class containing full_name_to_player_column_id function

    Returns:
        np.ndarray: array containing similarity scores between every frame and events,
                    size is #frames, #events
    """
    sim_mat = np.zeros((len(tracking_batch), len(event_batch)))

    # Pre-compute time_diff
    track_dt = tracking_batch["datetime"].values
    event_dt = event_batch["datetime"].values
    time_diff = (track_dt[:, np.newaxis] - event_dt) / pd.Timedelta(seconds=1)

    # Pre-compute ball_loc_diff
    track_bx = tracking_batch["ball_x"].values
    track_by = tracking_batch["ball_y"].values
    event_x = event_batch["start_x"].values
    event_y = event_batch["start_y"].values
    ball_x_diff = track_bx[:, np.newaxis] - event_x
    ball_y_diff = track_by[:, np.newaxis] - event_y
    ball_loc_diff = np.hypot(ball_x_diff, ball_y_diff)

    # Pre-compute time diff and ball diff for all events
    time_ball_diff = np.abs(time_diff) + ball_loc_diff / 5

    for row in event_batch.itertuples():
        i = row.Index

        if not np.isnan(row.player_id) and row.player_id != MISSING_INT:
            column_id_player = match.player_id_to_column_id(player_id=row.player_id)
            player_ball_diff = np.hypot(
                tracking_batch["ball_x"].values
                - tracking_batch[f"{column_id_player}_x"].values,
                tracking_batch["ball_y"].values
                - tracking_batch[f"{column_id_player}_y"].values,
            )

        else:
            player_ball_diff = [np.nan] * len(tracking_batch)

        # similarity function from: https://kwiatkowski.io/sync.soccer
        # Added information based in the type of event.

        # create array with all cost function variables
        if row.databallpy_event == "pass":
            total_shape = (len(tracking_batch), 4)
        elif row.databallpy_event == "shot":
            total_shape = (len(tracking_batch), 5)
        elif row.databallpy_event == "dribble":
            total_shape = (len(tracking_batch), 3)

        total = np.zeros(total_shape)
        total[:, :3] = np.array(
            [time_ball_diff[:, i] / 2, time_ball_diff[:, i] / 2, player_ball_diff]
        ).T

        if row.databallpy_event in ["shot", "pass"]:
            total[:, 3] = 1 / tracking_batch["ball_acceleration_sqrt"].values.clip(
                0.1
            )  # ball acceleration
            if row.databallpy_event == "shot":
                total[:, 4] = tracking_batch[
                    f"goal_angle_{['home', 'away'][row.team_id != match.home_team_id]}"
                    "_team"
                ].values

        # take the mean of all cost function variables.
        mask = np.isnan(total).all(axis=1)
        if total[mask].shape[0] > 0:
            total[mask] = np.nanmax(total)
        sim_mat[:, i] = np.nanmean(total, axis=1)
    # replace nan values with highest value, the algorithm will not assign these
    sim_mat[np.isnan(sim_mat)] = np.nanmax(sim_mat)
    den = np.nanmax(np.nanmin(sim_mat, axis=1))  # scale similarity scores
    sim_mat = np.exp(-sim_mat / den)

    return sim_mat


def _needleman_wunsch(
    sim_mat: np.ndarray, gap_event: int = -10, gap_frame: int = 1
) -> dict:
    """
    Function that calculates the optimal alignment between events and frames
    given similarity scores between all frames and events
    Based on: https://gist.github.com/slowkow/06c6dba9180d013dfd82bec217d22eb5

    Args:
        sim_mat (np.ndarray): matrix with similarity between every frame and event
        gap_event (int): penalty for leaving an event unassigned to a frame
                         (not allowed), defaults to -10
        gap_frame (int): penalty for leaving a frame unassigned to a penalty
                         (very common), defaults to 1

    Returns:
       event_frame_dict (dict): dictionary with events as keys and frames as values
    """
    n_frames, n_events = np.shape(sim_mat)

    F = np.zeros((n_frames + 1, n_events + 1))
    F[:, 0] = np.linspace(0, n_frames * gap_frame, n_frames + 1)
    F[0, :] = np.linspace(0, n_events * gap_event, n_events + 1)

    # Pointer matrix
    P = np.zeros((n_frames + 1, n_events + 1))
    P[:, 0] = 3
    P[0, :] = 4

    t = np.zeros(3)
    for i in range(n_frames):
        for j in range(n_events):
            t[0] = F[i, j] + sim_mat[i, j]
            t[1] = F[i, j + 1] + gap_frame  # top + gap frame
            t[2] = F[i + 1, j] + gap_event  # left + gap event

            # manually calculate tmax instead of using np.max() since it is
            # faster when using small arrays. On top of that, we can now fill in
            # the pointer matrix at the same time.
            if t[0] >= t[1] and t[0] >= t[2]:  # t[0] = tmax thus whe got a match
                tmax = t[0]
                P[i + 1, j + 1] += 2
            elif (
                t[1] >= t[0] and t[1] >= t[2]
            ):  # t[1] = tmax thus we got a frame unassigned
                tmax = t[1]
                P[i + 1, j + 1] += 3
            else:  # t[2] = tmax thus we got an event unassigned
                tmax = t[2]
                P[i + 1, j + 1] += 4

            F[i + 1, j + 1] = tmax

    # Trace through an optimal alignment.
    i = n_frames
    j = n_events
    frames = []
    events = []
    while i > 0 or j > 0:
        if P[i, j] in [2, 5, 6, 9]:  # 2 was added, match
            frames.append(i)
            events.append(j)
            i -= 1
            j -= 1
        elif P[i, j] in [3, 5, 7, 9]:  # 3 was added, frame unassigned
            frames.append(i)
            events.append(0)
            i -= 1
        elif P[i, j] in [4, 6, 7, 9]:  # 4 was added, event unassigned
            raise ValueError(
                "An event was left unassigned, check your gap penalty values"
            )
        else:
            raise ValueError(
                f"The algorithm got stuck due to an unexpected "
                f"value of P[{i}, {j}]: {P[i, j]}"
            )

    frames = frames[::-1]
    events = events[::-1]

    idx_events = [idx for idx, i in enumerate(events) if i > 0]
    event_frame_dict = {}
    for i in idx_events:
        event_frame_dict[events[i] - 1] = frames[i] - 1

    return event_frame_dict


def pre_compute_synchronisation_variables(
    tracking_data: pd.DataFrame,
    frame_rate: int,
    pitch_dimensions: tuple,
    periods: pd.DataFrame,
) -> pd.DataFrame:
    """Function that precomputes variables that are used in the synchronisation.
    The following variables are computed: ball_velocity, ball_acceleration,
    ball_acceleration_sqrt, goal_angle_home_team, goal_angle_away_team, and
    datetime.

    Args:
        tracking_data (pd.DataFrame): Tracking data of the match
        frame_rate (int): Frame rate of the tracking_data
        pitch_dimensions (tuple): Tuple containing the pitch dimensions
        periods (pd.DataFrame): Periods of the match, with corresponding datetime
            objects.

    Returns:
        pd.DataFrame: Tracking data with the precomputed variables.
    """
    # precompute ball acceleration
    if "ball_velocity" not in tracking_data.columns:
        tracking_data = _differentiate(
            df=tracking_data,
            max_val=50,
            new_name="velocity",
            metric="",  # differentiate the x and y values
            frame_rate=frame_rate,
            filter_type=None,
            column_ids=["ball"],
        )
    if "ball_acceleration" not in tracking_data.columns:
        tracking_data = _differentiate(
            df=tracking_data,
            new_name="acceleration",
            metric="v",  # differentiate the vx and vy values
            frame_rate=frame_rate,
            filter_type="savitzky_golay",
            max_val=150,
            column_ids=["ball"],
        )
    # take the square root to decrease the quadratic effect of the data
    tracking_data["ball_acceleration_sqrt"] = np.sqrt(
        tracking_data["ball_acceleration"]
    )

    # pre compute ball moving vector - ball goal vector angle
    goal_angle = get_smallest_angle(
        (
            tracking_data.loc[1:, ["ball_x", "ball_y"]]
            - tracking_data[["ball_x", "ball_y"]][:-1].values
        ).values,
        np.array([pitch_dimensions[0] / 2, 0])
        - tracking_data[["ball_x", "ball_y"]][:-1].values,
        angle_format="radian",
    )
    tracking_data["goal_angle_home_team"] = np.concatenate([goal_angle, [np.nan]])

    goal_angle = get_smallest_angle(
        (
            tracking_data.loc[1:, ["ball_x", "ball_y"]]
            - tracking_data[["ball_x", "ball_y"]][:-1].values
        ).values,
        np.array([pitch_dimensions[0] / 2, 0])
        - tracking_data[["ball_x", "ball_y"]][:-1].values,
        angle_format="radian",
    )

    tracking_data["goal_angle_away_team"] = np.concatenate([goal_angle, [np.nan]])

    # add datetime objects to tracking_data
    start_datetime_period = {}
    start_frame_period = {}

    for _, row in periods.iterrows():
        start_datetime_period[row["period"]] = row["start_datetime_td"]
        start_frame_period[row["period"]] = row["start_frame"]

    valid_start_frame_periods = np.array(
        [
            start_frame_period[p] if p != MISSING_INT else np.nan
            for p in tracking_data["period"]
        ]
    )

    datetime_values = pd.Series(
        [
            start_datetime_period[p] if p != MISSING_INT else pd.to_datetime("NaT")
            for p in tracking_data["period"]
        ]
    ) + pd.to_timedelta(
        (tracking_data["frame"] - valid_start_frame_periods) / frame_rate * 1000,
        "milliseconds",
    )

    # Combine the calculated values into a DataFrame
    new_columns = {
        "datetime": datetime_values,
        "databallpy_event": None,
        "event_id": MISSING_INT,
    }

    tracking_data = pd.concat([tracking_data, pd.DataFrame(new_columns)], axis=1)

    return tracking_data


def create_batches(
    period_id: int,
    frame_rate: int,
    n_batches_per_half: int,
    tracking_data: pd.DataFrame,
    event_data: pd.DataFrame,
    align_datetimes: bool,
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:
    """Function that creates batches to loop over. The batches are created based on
    the number of batches per half. The first batch starts at the first frame of the
    period. The last batch ends at the last frame of the period. The batches are
    created in such a way that the last frame of the batch is always a frame that
    contains tracking data.

    Args:
        period_id (int): Period id of the period that is being synchronised
        frame_rate (int): Frame rate of the tracking data
        n_batches_per_half (int):  the number of batches that are created per half.
        tracking_data (pd.DataFrame): Tracking data of the match
        event_data (pd.DataFrame): Event data of the match that needs to be synced
        align_datetimes (bool): Wheter or not to align the datetimes of the tracking
            and event data. This is done by substracting the difference between the
            first event and the first tracking frame from all event datetimes.

    Returns:
        Tuple[list, pd.DataFrame, pd.DataFrame]: Tuple containing the end datetimes
            of the batches, the tracking data of the period, and the event data of
            the period.
    """
    # create batches to loop over
    first_frame = tracking_data.loc[tracking_data["period"] == period_id, "frame"].iloc[
        0
    ]
    end_frame = tracking_data.loc[tracking_data["period"] == period_id, "frame"].iloc[
        -1
    ]
    first_datetime = tracking_data.loc[
        tracking_data["period"] == period_id, "datetime"
    ].iloc[0]

    delta2 = end_frame - first_frame
    end_frames = np.floor(
        np.arange(
            delta2 / n_batches_per_half,
            delta2 + delta2 / n_batches_per_half,
            delta2 / n_batches_per_half,
        )
        + first_frame
    )
    end_datetimes = [
        first_datetime + pd.to_timedelta((x - first_frame) / frame_rate, unit="s")
        for x in end_frames
    ]

    tracking_data_full_period = tracking_data[tracking_data["period"] == period_id]
    event_data_full_period = event_data[event_data["period_id"] == period_id].copy()

    # if both timestamps are precise, we do not need to calculdate the datetime diff
    if align_datetimes:
        start_events = ["pass", "shot"]
        datetime_first_event = event_data_full_period[
            event_data_full_period["databallpy_event"].isin(start_events)
        ].iloc[0]["datetime"]
        datetime_first_tracking_frame = tracking_data_full_period[
            tracking_data_full_period["ball_status"] == "alive"
        ].iloc[0]["datetime"]
        diff_datetime = datetime_first_event - datetime_first_tracking_frame
        event_data_full_period["datetime"] -= diff_datetime

    return end_datetimes, tracking_data_full_period, event_data_full_period
