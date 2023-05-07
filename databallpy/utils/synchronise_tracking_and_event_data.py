import datetime as dt

import numpy as np
import pandas as pd
from tqdm import tqdm


def synchronise_tracking_and_event_data(
    match, n_batches_per_half: int = 100, verbose: bool = True
):
    """Function that synchronises tracking and event data using Needleman-Wunsch
       algorithmn. Based on: https://kwiatkowski.io/sync.soccer

    Args:
        match (Match): Match object
        n_batches_per_half (int): the number of batches that are created per half.
        A higher number of batches reduces the time the code takes to load, but
        reduces the accuracy for events close to the splits. Default = 100
        verbose (bool, optional): Wheter or not to print info about the progress
        in the terminal. Defaults to True.

    Currently works for the following events:
        'pass', 'aerial', 'interception', 'ball recovery', 'dispossessed', 'tackle',
        'take on', 'clearance', 'blocked pass', 'offside pass', 'attempt saved',
        'save', 'foul', 'miss', 'challenge', 'goal'

    """
    events_to_sync = [
        "pass",
        "aerial",
        "interception",
        "ball recovery",
        "dispossessed",
        "tackle",
        "take on",
        "clearance",
        "blocked pass",
        "offside pass",
        "attempt saved",
        "save",
        "foul",
        "miss",
        "challenge",
        "goal",
        "shot",
    ]

    tracking_data = match.tracking_data
    event_data = match.event_data

    start_datetime_period = {}
    start_frame_period = {}
    for _, row in match.periods.iterrows():
        start_datetime_period[row["period"]] = row["start_datetime_td"]
        start_frame_period[row["period"]] = row["start_frame"]

    tracking_data["datetime"] = [
        start_datetime_period[int(p)]
        + dt.timedelta(
            milliseconds=int((x - start_frame_period[p]) / match.frame_rate * 1000)
        )
        if p > 0
        else pd.to_datetime("NaT")
        for x, p in zip(tracking_data["frame"], tracking_data["period"])
    ]

    tracking_data["event"] = np.nan
    tracking_data["event_id"] = np.nan
    event_data["tracking_frame"] = np.nan

    mask_events_to_sync = event_data["event"].isin(events_to_sync)
    event_data = event_data[mask_events_to_sync]

    periods_played = match.periods[match.periods["start_frame"] > 0]["period"].values

    for p in periods_played:
        # create batches to loop over
        start_batch_frame = match.periods.loc[
            match.periods["period"] == p, "start_frame"
        ].iloc[0]
        start_batch_datetime = start_datetime_period[p] + dt.timedelta(
            milliseconds=int(
                (start_batch_frame - start_frame_period[p]) / match.frame_rate * 1000
            )
        )
        delta = (
            match.periods.loc[match.periods["period"] == p, "end_frame"].iloc[0]
            - start_batch_frame
        )
        end_batches_frames = np.floor(
            np.arange(
                delta / n_batches_per_half,
                delta + delta / n_batches_per_half,
                delta / n_batches_per_half,
            )
            + start_batch_frame
        )
        end_batches_datetime = [
            start_datetime_period[int(p)]
            + dt.timedelta(
                milliseconds=int(
                    (x - start_frame_period[int(p)]) / match.frame_rate * 1000
                )
            )
            for x in end_batches_frames
        ]

        tracking_data_period = tracking_data[tracking_data["period"] == p]
        event_data_period = event_data[event_data["period_id"] == p].copy()
        start_events = ["pass", "miss", "goal"]
        datetime_first_event = event_data_period[
            event_data_period["event"].isin(start_events)
        ].iloc[0]["datetime"]
        datetime_first_tracking_frame = tracking_data_period[
            tracking_data_period["ball_status"] == "alive"
        ].iloc[0]["datetime"]
        diff_datetime = datetime_first_event - datetime_first_tracking_frame
        event_data_period["datetime"] -= diff_datetime

        if verbose:
            print(f"Syncing period {p}...")
            zip_batches = tqdm(
                zip(end_batches_frames, end_batches_datetime),
                total=len(end_batches_frames),
            )
        else:
            zip_batches = zip(end_batches_frames, end_batches_datetime)
        for end_batch_frame, end_batch_datetime in zip_batches:

            tracking_batch = tracking_data_period[
                (tracking_data_period["frame"] <= end_batch_frame)
                & (tracking_data_period["frame"] >= start_batch_frame)
            ].reset_index(drop=False)
            event_batch = event_data_period[
                (event_data_period["datetime"] >= start_batch_datetime)
                & (event_data_period["datetime"] <= end_batch_datetime)
            ].reset_index(drop=False)

            sim_mat = _create_sim_mat(tracking_batch, event_batch, match)
            event_frame_dict = _needleman_wunsch(sim_mat)
            for event, frame in event_frame_dict.items():
                event_id = int(event_batch.loc[event, "event_id"])
                event_type = event_batch.loc[event, "event"]
                event_index = int(event_batch.loc[event, "index"])
                tracking_frame = int(tracking_batch.loc[frame, "index"])
                tracking_data.loc[tracking_frame, "event"] = event_type
                tracking_data.loc[tracking_frame, "event_id"] = event_id
                event_data.loc[event_index, "tracking_frame"] = tracking_frame

            start_batch_frame = tracking_data_period.loc[tracking_frame, "frame"]
            start_batch_datetime = event_data_period[
                event_data_period["event_id"] == event_id
            ]["datetime"].iloc[0]

    tracking_data.drop("datetime", axis=1, inplace=True)
    match.tracking_data = tracking_data
    match.event_data = event_data

    match.is_synchronised = True


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
    tracking_batch["datetime"] = tracking_batch["datetime"]

    for i, event in event_batch.iterrows():
        time_diff = (tracking_batch["datetime"] - event["datetime"]) / dt.timedelta(
            seconds=1
        )
        ball_loc_diff = np.hypot(
            tracking_batch["ball_x"] - event["start_x"],
            tracking_batch["ball_y"] - event["start_y"],
        )

        if not np.isnan(event["player_id"]):
            column_id_player = match.player_id_to_column_id(
                player_id=event["player_id"]
            )
            player_ball_diff = np.hypot(
                tracking_batch["ball_x"] - tracking_batch[f"{column_id_player}_x"],
                tracking_batch["ball_y"] - tracking_batch[f"{column_id_player}_y"],
            )
        else:
            player_ball_diff = 0
        # similarity function from: https://kwiatkowski.io/sync.soccer
        sim_mat[:, i] = np.abs(time_diff) + ball_loc_diff / 5 + player_ball_diff

    sim_mat[np.isnan(sim_mat)] = np.nanmax(
        sim_mat
    )  # replace nan values with highest value
    den = np.nanmax(np.nanmin(sim_mat, axis=1))  # scale similarity scores
    sim_mat = np.exp(-sim_mat / den)

    return sim_mat


def _needleman_wunsch(
    sim_mat: np.ndarray, gap_event: int = -1, gap_frame: int = 1
) -> dict:
    """
    Function that calculates the optimal alignment between events and frames
    given similarity scores between all frames and events
    Based on: https://gist.github.com/slowkow/06c6dba9180d013dfd82bec217d22eb5

    Args:
        sim_mat (np.ndarray): matrix with similarity between every frame and event
        gap_event (int): penalty for leaving an event unassigned to a frame
                         (not allowed), defaults to -1
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
            tmax = np.max(t)
            F[i + 1, j + 1] = tmax

            if t[0] == tmax:  # match
                P[i + 1, j + 1] += 2
            if t[1] == tmax:  # frame unassigned
                P[i + 1, j + 1] += 3
            if t[2] == tmax:  # event unassigned
                P[i + 1, j + 1] += 4

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

    frames = frames[::-1]
    events = events[::-1]

    idx_events = [idx for idx, i in enumerate(events) if i > 0]
    event_frame_dict = {}
    for i in idx_events:
        event_frame_dict[events[i] - 1] = frames[i] - 1

    return event_frame_dict
