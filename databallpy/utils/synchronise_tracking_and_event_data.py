import inspect

import numpy as np
import pandas as pd
from tqdm import tqdm

from databallpy.features import get_smallest_angle
from databallpy.features.differentiate import _differentiate
from databallpy.utils.constants import DATABALLPY_EVENTS, MISSING_INT
from databallpy.utils.logging import logging_wrapper
from databallpy.utils.utils import sigmoid

FRAME_UNASSIGNED = 3
EVENT_FRAME_MATCH = 2
EVENT_UNASSIGNED = 4


@logging_wrapper(__file__)
def synchronise_tracking_and_event_data(
    tracking_data: pd.DataFrame,
    event_data: pd.DataFrame,
    home_players: pd.DataFrame,
    away_players: pd.DataFrame,
    cost_functions: dict = {},
    n_batches: int | str = "smart",
    optimize: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Function that synchronises tracking and event data using Needleman-Wunsch
       algorithmn. Based on: https://kwiatkowski.io/sync.soccer. The similarity
       function is used but information based on the specific event is added. For
       instance, when the event is a shot, the ball acceleration should be high and
       the ball should move towards the goal.

    Args:
        tracking_data (pd.DataFrame): Tracking data of the game
        event_data (pd.DataFrame): Event data of the game
        home_players (pd.DataFrame): Information about the home players
        away_players (pd.DataFrame): Information about the away_players
        cost_functions (dict, optional): Dictionary containing the cost functions that
            are used to calculate the similarity between the tracking and event data.
            The keys of the dictionary are the event types, the values are the cost
            functions. The cost functions will be called with the tracking data and the
            event as arguments. The cost functions should return an array containing
            the cost of the similarity between the tracking data and the event, scaled
            between 0 and 1. If no cost functions are passed, the default cost functions
            are used.
        n_batches (int, str): the number of batches that are created.
            A higher number of batches reduces the time the code takes to load, but
            reduces the accuracy for events close to the splits. Default = "smart".
            If n_batches is set to "smart", the number of batches is chosen based on
            the number of periods that the ball is set to alive. This way, the
            transition of the similarity matrix is always between two periods of
            active play. This method is optimised for events that are in active play,
            other events, such as yellow cards might not be perfectly synced.
        optimize (bool, optional): Whether or not to optimize the algorithm. If
                errors or warnings are raised, try if setting to False works. Defaults
                to True.
        verbose (bool, optional): Wheter or not to print info about the progress
            in the terminal. Defaults to True.

    Currently works for the following databallpy_events:
        'pass', 'shot', 'dribble', and 'tackle'

    Returns:
        tuple: Tuple containing two DataFrames. The first DataFrame contains
            information about the tracking data. The columns are:
            - databallpy_event: the event type
            - event_id: the event id
            - sync_certainty: the certainty of the synchronisation
            The second DataFrame contains information about the event data. The
            columns are:
            - tracking_frame: the frame in the tracking data that is synced to the event
            - sync_certainty: the certainty of the synchronisation
    """

    event_data_to_sync = event_data[
        event_data["databallpy_event"].isin(DATABALLPY_EVENTS)
    ]

    if n_batches == "smart":
        end_datetimes = create_smart_batches(tracking_data)
    else:
        end_datetimes = create_naive_batches(
            n_batches,
            tracking_data,
        )

    if verbose:
        end_datetimes = tqdm(
            end_datetimes,
            desc="Syncing event and tracking data",
            unit="batches",
            leave=False,
        )

    # loop over batches
    extra_tracking_info = pd.DataFrame(
        index=tracking_data.index,
        columns=["databallpy_event", "event_id", "sync_certainty"],
    )
    extra_event_info = pd.DataFrame(
        index=event_data.index, columns=["tracking_frame", "sync_certainty"]
    )
    batch_first_datetime = tracking_data["datetime"].iloc[0]
    for batch_end_datetime in end_datetimes:
        # create batches
        event_mask = event_data_to_sync["datetime"].between(
            batch_first_datetime, batch_end_datetime, inclusive="left"
        )
        tracking_mask = tracking_data["datetime"].between(
            batch_first_datetime, batch_end_datetime, inclusive="left"
        )
        tracking_batch = tracking_data[tracking_mask].reset_index(drop=False)
        event_batch = event_data_to_sync[event_mask].reset_index(drop=False)

        if len(event_batch) > 0:
            sim_mat = _create_sim_mat(
                tracking_batch,
                event_batch,
                home_players,
                away_players,
                cost_functions,
            )
            event_frame_dict = _needleman_wunsch(sim_mat, enable_optimization=optimize)

            # assign events to tracking data frames
            for event, frame in event_frame_dict.items():
                event_id = int(event_batch.loc[event, "event_id"])
                event_type = event_batch.loc[event, "databallpy_event"]
                event_index = int(event_batch.loc[event, "index"])
                tracking_frame = int(tracking_batch.loc[frame, "index"])
                extra_tracking_info.loc[tracking_frame, "databallpy_event"] = event_type
                extra_tracking_info.loc[tracking_frame, "event_id"] = event_id
                extra_tracking_info.loc[tracking_frame, "sync_certainty"] = sim_mat[
                    frame, event
                ]
                extra_event_info.loc[event_index, "tracking_frame"] = tracking_frame
                extra_event_info.loc[event_index, "sync_certainty"] = sim_mat[
                    frame, event
                ]

        batch_first_datetime = batch_end_datetime

    return extra_tracking_info, extra_event_info


@logging_wrapper(__file__)
def _create_sim_mat(
    tracking_batch: pd.DataFrame,
    event_batch: pd.DataFrame,
    home_players: pd.DataFrame,
    away_players: pd.DataFrame,
    cost_functions: dict = {},
) -> np.ndarray:
    """Function that creates similarity matrix between every frame and event in batch

    Args:
        tracking_batch (pd.DataFrame): batch of tracking data
        event_batch (pd.DataFrame): batch of event data
        home_players (pd.DataFrame): Information about the home players
        away_players (pd.DataFrame): Information about the away_players
        cost_functions (dict, optional): dictionary containing the cost functions that

    Returns:
        np.ndarray: array containing similarity scores between every frame and events,
            size is #frames, #events
    """
    sim_mat = np.zeros((len(tracking_batch), len(event_batch)))
    time_diff, ball_event_diff = pre_compute_cost_function_variables(
        tracking_batch, event_batch
    )

    for row in event_batch.itertuples():
        i = row.Index

        if row.databallpy_event == "pass":
            cost_function = cost_functions.get("pass", base_pass_cost_function)
        elif row.databallpy_event == "shot":
            cost_function = cost_functions.get("shot", base_shot_cost_function)
        else:  # dribble and tackle
            cost_function = cost_functions.get(
                row.databallpy_event, base_general_cost_ball_event
            )

        for side, players_df in zip(["home", "away"], [home_players, away_players]):
            if row.player_id in players_df["id"].values:
                team_side = side
                jersey = players_df.loc[
                    players_df["id"] == row.player_id, "shirt_num"
                ].iloc[0]
                break

        kwargs = {}
        sig = inspect.signature(cost_function)
        if "time_diff" in sig.parameters:
            kwargs["time_diff"] = time_diff[:, i]
        if "ball_event_distance" in sig.parameters:
            kwargs["ball_event_distance"] = ball_event_diff[:, i]

        cost = cost_function(
            tracking_data=tracking_batch,
            event=row,
            team_side=team_side,
            jersey=jersey,
            **kwargs,
        )

        _validate_cost(cost, len(tracking_batch))

        sim_mat[:, i] = cost

    sim_mat[np.isnan(sim_mat)] = 1
    sim_mat = -sim_mat + 1  # low cost is better similarity

    return sim_mat


@logging_wrapper(__file__)
def _needleman_wunsch(
    sim_mat,
    enable_optimization=True,
    gap_event=-10,
    gap_frame=0.2,
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
            (very common), defaults to 0.2

    Returns:
       event_frame_dict (dict): dictionary with events as keys and frames as values
    """
    n_frames, n_events = np.shape(sim_mat)

    function_matrix = np.zeros((n_frames + 1, n_events + 1), dtype=np.float32)
    function_matrix[:, 0] = np.linspace(0, n_frames * gap_frame, n_frames + 1)
    function_matrix[0, :] = np.linspace(0, n_events * gap_event, n_events + 1)

    pointer_matrix = np.zeros((n_frames + 1, n_events + 1), dtype=np.int16)
    pointer_matrix[:, 0] = 3
    pointer_matrix[0, :] = 4

    frames_high_sim_mat = np.where(sim_mat[:, 0] > 0.5)[0]
    for event_index in range(n_events):
        start_frame = (
            0
            if event_index == 0 or len(frames_high_sim_mat) == 0
            else max(0, frames_high_sim_mat[0] - 1)
        )
        if event_index + 1 == n_events:
            end_frame = n_frames
        else:
            frames_high_sim_mat = np.where(sim_mat[:, event_index + 1] > 0.5)[0]
            end_frame = (
                n_frames
                if len(frames_high_sim_mat) == 0 or event_index >= n_events - 2
                else min(n_frames, frames_high_sim_mat[-1] + 1)
            )
        start_frame, end_frame = (
            (0, n_frames) if not enable_optimization else (start_frame, end_frame)
        )

        for frame_index in range(start_frame, end_frame):
            match = (
                function_matrix[frame_index, event_index]
                + sim_mat[frame_index, event_index]
            )
            gap_f = (
                function_matrix[frame_index, event_index + 1] + gap_frame
            )  # top + gap frame
            gap_e = (
                function_matrix[frame_index + 1, event_index] + gap_event
            )  # left + gap event

            # Determine the maximum value and set the pointer matrix accordingly
            if gap_f >= match and gap_f >= gap_e:
                function_matrix[frame_index + 1, event_index + 1] = gap_f
                pointer_matrix[frame_index + 1, event_index + 1] = FRAME_UNASSIGNED
            elif match >= gap_e:
                function_matrix[frame_index + 1, event_index + 1] = match
                pointer_matrix[frame_index + 1, event_index + 1] = EVENT_FRAME_MATCH
            else:
                function_matrix[frame_index + 1, event_index + 1] = gap_e
                pointer_matrix[frame_index + 1, event_index + 1] = EVENT_UNASSIGNED

    # Solve
    frame_index = n_frames
    event_index = n_events
    frames = np.zeros(n_frames, dtype=np.int32)
    events = np.zeros(n_frames, dtype=np.int32)
    count = 0
    while frame_index > 0 or event_index > 0:
        if (
            pointer_matrix[frame_index, event_index] == FRAME_UNASSIGNED
        ):  # frame unassigned
            frames[count] = frame_index
            frame_index -= 1
        elif pointer_matrix[frame_index, event_index] == EVENT_FRAME_MATCH:  # match
            frames[count] = frame_index
            events[count] = event_index
            frame_index -= 1
            event_index -= 1
        elif (
            pointer_matrix[frame_index, event_index] == EVENT_UNASSIGNED
        ):  # event unassigned
            raise ValueError(
                "An event was left unassigned, check your gap penalty values"
            )
        else:
            raise ValueError(
                f"The algorithm got stuck due to an unexpected "
                f"value of P[{frame_index}, {event_index}]: {pointer_matrix[frame_index, event_index]}"
            )
        count += 1
    frames = frames[::-1]
    events = events[::-1]

    idx_events = np.where(events > 0)[0]
    event_frame_dict = {}
    for frame_index in idx_events:
        event_frame_dict[events[frame_index] - 1] = frames[frame_index] - 1

    return event_frame_dict


def pre_compute_cost_function_variables(
    tracking_batch: pd.DataFrame, event_batch: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Function that precomputes variables that are used in the cost functions. The
    following variables are computed: time_diff, and ball_event_diff.

    Args:
        tracking_batch (pd.DataFrame): Tracking data batch of the game
        event_batch (pd.DataFrame): Event data batch of the game

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing the precomputed variables
            time_diff, and ball_event_diff
    """
    # Pre-compute time_diff
    track_dt = tracking_batch["datetime"].values
    event_dt = event_batch["datetime"].values
    time_diff = (track_dt[:, np.newaxis] - event_dt) / pd.Timedelta(seconds=1)

    # Pre-compute ball_loc_diff bewteen tracking and event data
    track_bx = tracking_batch["ball_x"].values
    track_by = tracking_batch["ball_y"].values
    event_x = event_batch["start_x"].values
    event_y = event_batch["start_y"].values
    ball_x_diff = track_bx[:, np.newaxis] - event_x
    ball_y_diff = track_by[:, np.newaxis] - event_y
    ball_event_diff = np.hypot(ball_x_diff, ball_y_diff)

    return time_diff, ball_event_diff


def pre_compute_synchronisation_variables(
    tracking_data: pd.DataFrame,
    frame_rate: int,
    pitch_dimensions: tuple,
) -> pd.DataFrame:
    """Function that precomputes variables that are used in the synchronisation.
    The following variables are computed: ball_velocity, ball_acceleration,
    ball_acceleration_sqrt, goal_angle_home_team, and goal_angle_away_team.

    Args:
        tracking_data (pd.DataFrame): Tracking data of the game
        frame_rate (int): Frame rate of the tracking_data
        pitch_dimensions (tuple): Tuple containing the pitch dimensions

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

    # pre compute ball moving vector - ball goal vector angle
    goal_angle = get_smallest_angle(
        tracking_data.iloc[1:][["ball_x", "ball_y"]].values
        - tracking_data.iloc[:-1][["ball_x", "ball_y"]].values,
        np.array([pitch_dimensions[0] / 2, 0])
        - tracking_data[["ball_x", "ball_y"]][:-1].values,
        angle_format="radian",
    )
    tracking_data["goal_angle_home_team"] = np.concatenate([goal_angle, [np.nan]])

    goal_angle = get_smallest_angle(
        tracking_data.iloc[1:][["ball_x", "ball_y"]].values
        - tracking_data.iloc[:-1][["ball_x", "ball_y"]].values,
        np.array([pitch_dimensions[0] / 2, 0])
        - tracking_data[["ball_x", "ball_y"]][:-1].values,
        angle_format="radian",
    )

    tracking_data["goal_angle_away_team"] = np.concatenate([goal_angle, [np.nan]])

    return tracking_data


def create_naive_batches(
    n_batches: int,
    tracking_data: pd.DataFrame,
) -> list[pd.Timestamp]:
    """Function that creates batches to loop over. The batches are created based on
    the number of batches per half. The first batch starts at the first frame of the
    period. The last batch ends at the last frame of the period. The batches are
    created in such a way that the last frame of the batch is always a frame that
    contains tracking data.

    Args:
        n_batches_per_half (int):  the number of batches that are created per half.
        tracking_data (pd.DataFrame): Tracking data of the game

    Returns:
        list: The end datetimes of the batches.
    """
    len_periods = {
        period_id: len(tracking_data[tracking_data["period_id"] == period_id])
        for period_id in tracking_data["period_id"].unique()
    }
    len_periods = {
        period_id: len_period
        for period_id, len_period in len_periods.items()
        if period_id != MISSING_INT
    }
    tot_len = sum(len_periods.values())
    n_batches_per_period = {
        period_id: int(np.ceil(len_period / tot_len * n_batches))
        for period_id, len_period in len_periods.items()
    }

    end_datetimes_total = []
    for period_id, n_batches_period in n_batches_per_period.items():
        tracking_data_p = tracking_data[tracking_data["period_id"] == period_id]

        first_valid_frame_index = tracking_data_p[
            tracking_data_p["ball_status"] == "alive"
        ].index[0]
        last_valid_frame_index = tracking_data_p[
            tracking_data_p["ball_status"] == "alive"
        ].index[-1]

        # find the indexes where the batches end
        end_frames = np.floor(
            np.arange(
                first_valid_frame_index,
                last_valid_frame_index,
                (last_valid_frame_index - first_valid_frame_index) / n_batches_period,
            )
        ).astype(int)

        # drop the first datetime, is not a end datetime, and add the last
        end_frames = end_frames[1:]
        end_frames = np.concatenate([end_frames, np.array([last_valid_frame_index])])
        # find the datetimes where the batches end

        end_datetimes = [tracking_data_p.loc[x, "datetime"] for x in end_frames]
        end_datetimes_total += end_datetimes

    return end_datetimes_total


def create_smart_batches(tracking_data: pd.DataFrame) -> list[pd.Timestamp]:
    """Function that creates batches to loop over. The batches are created based on
    active periods of play. For every active period of play, it is checked when the
    last period of play ended. The split of the batches is chosen in such a way that
    it is exactly between two periods of active play. For example, if the first
    period of play ends at 10 seconds and the second period of play starts at 20
    seconds, the split is chosen at 15 seconds.

    Note: this method is optimised for events that are in active play, other events,
    such as yellow cards might not be perfectly synced.


    Args:
        tracking_data (pd.DataFrame): Tracking data of the game

    Returns:
        List: List containing the end datetimes of the batches.
    """

    first_valid_frame_index = tracking_data[
        tracking_data["ball_status"] == "alive"
    ].index[0]
    last_valid_frame_index = tracking_data[
        tracking_data["ball_status"] == "alive"
    ].index[-1]

    # find all the indexes where the ball switches from alive to dead
    # this is the last frame that the ball is alive in a batch
    end_alive_idxs = (
        np.where(
            (tracking_data.iloc[:-1]["ball_status"] == "alive").values
            & (tracking_data.iloc[1:]["ball_status"] == "dead").values
        )[0]
        + tracking_data.index[0]
    )
    if last_valid_frame_index not in end_alive_idxs:
        end_alive_idxs = np.concatenate(
            [end_alive_idxs, np.array([last_valid_frame_index])]
        )

    # find all the indexes where the ball switches from dead to alive
    # this is the first frame that the ball is alive in a batch
    start_alive_idxs = (
        np.where(
            (tracking_data.iloc[:-1]["ball_status"] == "dead").values
            & (tracking_data.iloc[1:]["ball_status"] == "alive").values
        )[0]
        + 1
        + tracking_data.index[0]
    )
    if first_valid_frame_index not in start_alive_idxs:
        start_alive_idxs = np.concatenate(
            [np.array([first_valid_frame_index]), start_alive_idxs]
        )

    # create batches to loop over
    last_end_dt = None
    end_datetimes = []
    for start_idx, end_idx in zip(start_alive_idxs, end_alive_idxs):
        start_dt = tracking_data.loc[start_idx]["datetime"]
        end_dt = tracking_data.loc[end_idx,]["datetime"]

        if last_end_dt is not None:
            difference = start_dt - last_end_dt
            last_end_dt + difference / 2
            end_datetimes.append(last_end_dt + difference / 2)

        last_end_dt = end_dt

    # add last datetime with some room for the last events
    end_datetimes.append(last_end_dt + pd.to_timedelta(1, unit="h"))

    return end_datetimes


def align_event_data_datetime(
    event_data: pd.DataFrame, tracking_data: pd.DataFrame, offset: float = 1.0
) -> pd.DataFrame:
    """Function that aligns the datetimes of the event data and tracking data. This
    is done by substracting the difference between the first event and the first
    tracking frame from all event datetimes.

    Args:
        event_data (pd.DataFrame): Event data of the game that needs to be synced
        tracking_data (pd.DataFrame): Tracking data of the game
        offset (float, optional): Offset in seconds that is added to the difference
            between the first event and the first tracking frame. This is done because
            this way the event is synced to the last frame the ball is close to a
            player. Which often corresponds with the event (pass and shots).
            Defaults to 1.0.

    Returns:
        pd.DataFrame: Event data with aligned datetimes

    """

    start_events = ["pass", "shot"]
    for period in tracking_data["period_id"].unique():
        if period == MISSING_INT:
            continue
        tracking_data_p = tracking_data[tracking_data["period_id"] == period]
        event_data_p = event_data[event_data["period_id"] == period]

        datetime_first_event = event_data_p[
            event_data_p["databallpy_event"].isin(start_events)
        ].iloc[0]["datetime"]
        datetime_first_tracking_frame = tracking_data_p[
            tracking_data_p["ball_status"] == "alive"
        ].iloc[0]["datetime"]
        diff_datetime = datetime_first_event - datetime_first_tracking_frame

        event_data.loc[event_data_p.index, "datetime"] -= diff_datetime
        # add offset in seconds to diff_datetime
        event_data.loc[event_data_p.index, "datetime"] += pd.to_timedelta(
            np.max([0.0, offset]), unit="seconds"
        )

    return event_data


def get_time_difference_cost(
    tracking_data: pd.DataFrame,
    event: pd.Series,
    time_diff: np.ndarray[float] | None = None,
    **kwargs: dict,
) -> np.ndarray[float]:
    """Function that calculates the cost of the time difference between the tracking
    and event data datetime. The cost is calculated using a sigmoid function. The
    sigmoid function is used to give a higher cost to large time differences. The
    sigmoid function is defined as:  a + b / (1 + c * np.exp(d * -(x - e))). The
    default values are a=0.0, b=1.0, c=1.0, d=1.0, e=0.0. The default values can be
    changed by passing them as keyword arguments. If no keyword arguments are passed,
    the default values are used, except for e, which is set to 5.


    Args:
        tracking_data (pd.DataFrame): Tracking data of the game
        event (pd.Series): Event that needs to be synced
        time_diff (np.ndarray[float], optional): Array containing the time difference
            between the tracking and event data datetime
        **kwargs: Keyword arguments that can be passed to the sigmoid function

    Returns:
        np.ndarray[float]: array containing the cost of the time difference
    """
    _validate_sigmoid_kwargs(kwargs)

    if time_diff is None:
        tracking_datetime = tracking_data["datetime"]
        event_datetime = event.datetime
        time_diff = (tracking_datetime - event_datetime).dt.total_seconds().values
    if len(kwargs) > 0:
        return sigmoid(np.abs(time_diff), **kwargs)
    return sigmoid(np.abs(time_diff), e=5)


def get_distance_ball_event_cost(
    tracking_data: pd.DataFrame,
    event: pd.Series,
    ball_event_distance: np.ndarray[float] | None = None,
    **kwargs: dict,
) -> np.ndarray[float]:
    """Function that calculates the cost of the distance between the ball and the
    event location. The cost is calculated using a sigmoid function. The sigmoid
    function is used to give a higher cost to larger euclidean distances. The
    sigmoid function is defined as:  a + b / (1 + c * np.exp(d * -(x - e))). The
    default values are a=0.0, b=1.0, c=1.0, d=1.0, e=0.0. The default values can be
    changed by passing them as keyword arguments. If no keyword arguments are passed,
    the default values are used, except for d and e, which are set to 5 and 6.

    Args:
        tracking_data (pd.DataFrame): Tracking data of the game
        event (pd.Series): Event that needs to be synced
        ball_event_distance (np.ndarray[float], optional): Array containing the distance
        **kwargs: Keyword arguments that can be passed to the sigmoid function

    Returns:
        np.ndarray[float]: array containing the cost of the distance between the ball
        and the event location
    """
    _validate_sigmoid_kwargs(kwargs)
    if ball_event_distance is None:
        ball_event_distance = np.hypot(
            tracking_data["ball_x"].values - event.start_x,
            tracking_data["ball_y"].values - event.start_y,
        )

    if len(kwargs) > 0:
        return sigmoid(ball_event_distance, **kwargs)
    return sigmoid(ball_event_distance, d=5, e=6)


def get_distance_ball_player_cost(
    tracking_data: pd.DataFrame,
    team_side: str,
    jersey: int,
    **kwargs: dict,
) -> np.ndarray[float]:
    """Function that calculates the cost of the difference between the ball and the
    player in the tracking data. The cost is calculated using a sigmoid function. The
    sigmoid function is used to give a higher cost to larger eucledian distances. The
    sigmoid function is defined as:  a + b / (1 + c * np.exp(d * -(x - e))). The
    default values are a=0.0, b=1.0, c=1.0, d=1.0, e=0.0. The default values can be
    changed by passing them as keyword arguments. If no keyword arguments are passed,
    the default values are used, except for d and e, which are set to 5 and 2.5.

    Args:
        tracking_data (pd.DataFrame): Tracking data of the game
        team_side (str): Either home or away, the side of the player performing the event
        jersey (int): The shirt number of the player performing the event
        **kwargs: Keyword arguments that can be passed to the sigmoid function

    Returns:
        np.ndarray[float]: array containing the cost of the distance between the ball
        and the player in the tracking data
    """
    _validate_sigmoid_kwargs(kwargs)
    col_id = f"{team_side}_{jersey}"
    distance = np.hypot(
        tracking_data["ball_x"].values - tracking_data[f"{col_id}" + "_x"].values,
        tracking_data["ball_y"].values - tracking_data[f"{col_id}" + "_y"].values,
    )

    if len(kwargs) > 0:
        return sigmoid(distance, **kwargs)
    return sigmoid(distance, d=5, e=2.5)


def get_ball_acceleration_cost(
    tracking_data: pd.DataFrame, **kwargs: dict
) -> np.ndarray[float]:
    """Function that calculates the cost of the ball acceleration in the tracking data.
    The cost is calculated using a sigmoid function. The sigmoid function is used to
    give a lower cost to higher ball accelerations. The sigmoid function is defined as:
    a + b / (1 + c * np.exp(d * -(x - e))). The default values are a=0.0, b=1.0, c=1.0,
    d=1.0, e=0.0. The default values can be changed by passing them as keyword
    arguments. If no keyword arguments are passed, the default values are used, except
    for d and e, which are set to 0.2 and -25.0. In this case the acceleration is also
    passed by multiplying it with -1. This will not be done if keyword arguments are
    passed.

    Args:
        tracking_data (pd.DataFrame): Tracking data of the game
        **kwargs: Keyword arguments that can be passed to the sigmoid function

    Returns:
        np.ndarray[float]: array containing the cost of the ball acceleration
    """
    _validate_sigmoid_kwargs(kwargs)
    acc = tracking_data["ball_acceleration"].values
    if len(kwargs) > 0:
        return sigmoid(acc, **kwargs)
    return sigmoid(-tracking_data["ball_acceleration"].values, d=0.2, e=-25.0)


def get_player_ball_distance_increase_cost(
    tracking_data: pd.DataFrame,
    team_side: str,
    jersey: str,
    **kwargs: dict,
) -> np.ndarray[float]:
    """Function that calculates the cost of the increase in distance between the player
    and the ball. When passing or shooting the ball, the distance between the player and
    the ball should increase. The cost is calculated using a sigmoid function. The
    sigmoid function is used to give a lower cost to larger increases in distance. The
    sigmoid function is defined as:  a + b / (1 + c * np.exp(d * -(x - e))). The default
    values are a=0.0, b=1.0, c=1.0, d=1.0, e=0.0. The default values can be changed by
    passing them as keyword arguments. If no keyword arguments are passed, the default
    values are used, except for d and e, which are set to -8.0.

    Args:
        tracking_data (pd.DataFrame): Tracking data of the game
        team_side (str): Either home or away, the side of the player performing the event
        jersey (int): The shirt number of the player performing the event
        **kwargs: Keyword arguments that can be passed to the sigmoid function

    Returns:
        np.ndarray[float]: array containing the cost of the increase in distance between
        the player and the ball
    """
    _validate_sigmoid_kwargs(kwargs)
    col_id = f"{team_side}_{jersey}"
    player_ball_diff = np.hypot(
        tracking_data["ball_x"].values - tracking_data[f"{col_id}" + "_x"].values,
        tracking_data["ball_y"].values - tracking_data[f"{col_id}" + "_y"].values,
    )
    if len(kwargs) > 0:
        return sigmoid(np.gradient(player_ball_diff), **kwargs)
    return sigmoid(np.gradient(player_ball_diff), d=-8.0)


def get_ball_goal_angle_cost(
    tracking_data: pd.DataFrame,
    team_side: str,
    pitch_length: float,
    **kwargs: dict,
) -> np.ndarray[float]:
    """Function that calculates the cost of the angle between the ball moving direction
    and the goal. The cost is calculated using a sigmoid function. The sigmoid function
    is used to give a higher cost to larger angles. The sigmoid function is defined as:
    a + b / (1 + c * np.exp(d * -(x - e))). The default values are a=0.0, b=1.0, c=1.0,
    d=1.0, e=0.0. The default values can be changed by passing them as keyword
    arguments. If no keyword arguments are passed, the default values are used, except
    for d and e, which are set to 6 and 0.2 * np.pi.

    Args:
        tracking_data (pd.DataFrame): Tracking data of the game
        team_side (str): Either home or away, the side of the player performing the event
        pitch_length (float): The length of the pitch in meters
        **kwargs: Keyword arguments that can be passed to the sigmoid function

    Returns:
        np.ndarray[float]: array containing the cost of the angle between the ball
            moving direction and the goal
    """
    _validate_sigmoid_kwargs(kwargs)
    if not all(
        x in tracking_data.columns
        for x in ["goal_angle_home_team", "goal_angle_away_team"]
    ):
        goal_loc = np.array([pitch_length / 2, 0])
        if team_side == "away":
            goal_loc[0] = -goal_loc[0]

        ball_moving_vectors = (
            tracking_data.iloc[1:][["ball_x", "ball_y"]].values
            - tracking_data.iloc[:-1][["ball_x", "ball_y"]].values
        )

        ball_goal_vectors = (
            goal_loc - tracking_data.iloc[:-1][["ball_x", "ball_y"]].values
        )
        goal_angle = get_smallest_angle(
            ball_moving_vectors,
            ball_goal_vectors,
            angle_format="radian",
        )

        goal_angle = np.concatenate([goal_angle, [goal_angle[-1]]])
    else:
        goal_angle = (
            tracking_data["goal_angle_home_team"].values
            if team_side == "home"
            else tracking_data["goal_angle_away_team"].values
        )

    if len(kwargs) > 0:
        return sigmoid(goal_angle, **kwargs)
    return sigmoid(goal_angle, d=6, e=0.2 * np.pi)


def combine_cost_functions(costs: list) -> np.ndarray[float]:
    """Function that combines multiple cost functions into one. The cost functions are
    passed as keyword arguments. The cost functions are combined by taking the mean of
    all the cost functions. The cost functions should return an array with the cost of
    each frame.

    Args:
        costs (list): List containing the cost values

    Returns:
        np.ndarray[float]: array containing the combined cost of all cost functions
    """
    total_array = np.array(costs)
    total_array[:, np.isnan(total_array).all(axis=0)] = 1
    return np.nanmean(total_array, axis=0)


def base_pass_cost_function(
    tracking_data: pd.DataFrame,
    event: pd.Series,
    team_side: str,
    jersey: int,
    time_diff: np.ndarray | None = None,
    ball_event_distance: np.ndarray | None = None,
) -> np.ndarray[float]:
    """Function that calculates the total cost of a pass event compared to each frame.
    The base pase cost function includes:
    1. Time difference between the tracking and event data datetime
    2. Distance between the ball and the event location
    3. Distance between the ball and the player in the tracking data
    4. Absolute ball acceleration
    5. The increase in distance between the player and the ball

    Args:
        tracking_data (pd.DataFrame): Tracking data of the game
        event (pd.Series): Pass event that needs to be synced
        team_side (str): Either home or away, the side of the player performing the event
        jersey (int): The shirt number of the player performing the event
        time_diff (np.ndarray[float], optional): Array containing the time difference
            between the tracking and event data datetime
        ball_event_distance (np.ndarray[float], optional): Array containing the
            distance between the ball and the event location

    Returns:
        np.ndarray[float]: array containing the total cost of the pass event compared
            to each frame
    """

    time_diff_cost = get_time_difference_cost(tracking_data, event, time_diff=time_diff)
    distance_ball_event_cost = get_distance_ball_event_cost(
        tracking_data, event, ball_event_distance=ball_event_distance
    )
    distance_ball_player_cost = get_distance_ball_player_cost(
        tracking_data, team_side, jersey
    )
    ball_acceleration_cost = get_ball_acceleration_cost(tracking_data)
    player_ball_diff_cost = get_player_ball_distance_increase_cost(
        tracking_data, team_side, jersey
    )

    return combine_cost_functions(
        [
            time_diff_cost,
            distance_ball_event_cost,
            distance_ball_player_cost,
            ball_acceleration_cost,
            player_ball_diff_cost,
        ]
    )


def base_shot_cost_function(
    tracking_data: pd.DataFrame,
    event: pd.Series,
    team_side: str,
    jersey: int,
    pitch_length: float = 105.0,
    time_diff: np.ndarray | None = None,
    ball_event_distance: np.ndarray | None = None,
) -> np.ndarray[float]:
    """Function that calculates the total cost of a shot event compared to each frame.
    The base shot cost function includes:
    1. Time difference between the tracking and event data datetime
    2. Distance between the ball and the event location
    3. Distance between the ball and the player in the tracking data
    4. Absolute ball acceleration
    5. The increase in distance between the player and the ball
    6. The angle between the ball moving direction and the goal

    Args:
        tracking_data (pd.DataFrame): Tracking data of the game
        event (pd.Series): Shot event that needs to be synced
        team_side (str): Either home or away, the side of the player performing the event
        jersey (int): The shirt number of the player performing the event
        time_diff (np.ndarray[float], optional): Array containing the time difference
            between the tracking and event data datetime
        ball_event_distance (np.ndarray[float], optional): Array containing the distance
            between the ball and the event location

    Returns:
        np.ndarray[float]: array containing the total cost of the shot event compared to
            each frame
    """

    time_diff_cost = get_time_difference_cost(tracking_data, event, time_diff=time_diff)
    distance_ball_event_cost = get_distance_ball_event_cost(
        tracking_data, event, ball_event_distance=ball_event_distance
    )
    distance_ball_player_cost = get_distance_ball_player_cost(
        tracking_data, team_side, jersey
    )
    ball_acceleration_cost = get_ball_acceleration_cost(tracking_data)
    player_ball_diff_cost = get_player_ball_distance_increase_cost(
        tracking_data, team_side, jersey
    )
    goal_angle_cost = get_ball_goal_angle_cost(tracking_data, team_side, pitch_length)

    return combine_cost_functions(
        [
            time_diff_cost,
            distance_ball_event_cost,
            distance_ball_player_cost,
            ball_acceleration_cost,
            player_ball_diff_cost,
            goal_angle_cost,
        ]
    )


def base_general_cost_ball_event(
    tracking_data: pd.DataFrame,
    event: pd.Series,
    team_side: str,
    jersey: int,
    time_diff: np.ndarray | None = None,
    ball_event_distance: np.ndarray | None = None,
) -> np.ndarray[float]:
    """Function that calculates the total cost of an event compared to each frame. The
    base general cost function includes:
    1. Time difference between the tracking and event data datetime
    2. Distance between the ball and the event location
    3. Distance between the ball and the player in the tracking data

    Args:
        tracking_data (pd.DataFrame): Tracking data of the game
        event (pd.Series): Event that needs to be synced
        team_side (str): Either home or away, the side of the player performing the event
        jersey (int): The shirt number of the player performing the event
        time_diff (np.ndarray[float], optional): Array containing the time difference
            between the tracking and event data datetime
        ball_event_distance (np.ndarray[float], optional): Array containing the distance
            between the ball and the event location

    Returns:
        np.ndarray[float]: array containing the total cost of the event compared to each
            frame
    """

    time_diff_cost = get_time_difference_cost(tracking_data, event, time_diff=time_diff)
    distance_ball_event_cost = get_distance_ball_event_cost(
        tracking_data, event, ball_event_distance=ball_event_distance
    )
    distance_ball_player_cost = get_distance_ball_player_cost(
        tracking_data, team_side, jersey
    )

    return combine_cost_functions(
        [time_diff_cost, distance_ball_event_cost, distance_ball_player_cost]
    )


def _validate_cost(
    cost: np.ndarray[float],
    expected_len: int,
) -> None:
    """
    Simple function to validate the output of the cost functions. The cost function
    should return a numpy array with the same length as the tracking data. The cost
    function should not return any NaN values, negative values, or values larger than 1.
    """
    if not isinstance(cost, np.ndarray):
        raise TypeError(
            f"Cost function should return a numpy array, got {type(cost)} instead"
        )
    if cost.ndim != 1:
        raise ValueError(
            f"Cost function should return a 1D numpy array, got {cost.ndim}D array"
            " instead"
        )

    if cost.shape[0] != expected_len:
        raise ValueError(
            "Cost function should return an array with the same length as the tracking "
            f"data, got {cost.shape[0]} instead of {expected_len}"
        )

    if np.isnan(cost).any():
        raise ValueError("Cost function should not return any NaN values")

    if np.min(cost) < 0:
        raise ValueError("Cost function should not return any negative values")

    if np.max(cost) > 1:
        raise ValueError("Cost function should not return any values larger than 1")


def _validate_sigmoid_kwargs(kwargs: dict[str, float]) -> None:
    """Function that validates the keyword arguments passed to the sigmoid function.
    The keyword arguments should only contain the following keys: a, b, c, d, e. The
    values should be integers or floats.

    Args:
        kwargs (dict[str, float]): Dictionary containing the keyword arguments passed to
            the sigmoid function

    Raises:
        ValueError: Invalid keyword argument _key_ passed to the sigmoid function
        ValueError: Invalid value _value_ passed to the sigmoid function for keyword
    """
    for key, value in kwargs.items():
        if key not in ["a", "b", "c", "d", "e"]:
            raise ValueError(
                f"Invalid keyword argument {key} passed to the sigmoid function"
            )
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(
                f"Invalid value {value} passed to the sigmoid function for keyword "
                f"argument {key}. Value should be an integer or float"
            )
