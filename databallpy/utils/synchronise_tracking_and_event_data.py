import numpy as np
import pandas as pd
from tqdm import tqdm

from databallpy.features import get_smallest_angle
from databallpy.features.differentiate import _differentiate
from databallpy.utils.constants import DATABALLPY_EVENTS, MISSING_INT
from databallpy.utils.logging import create_logger
from databallpy.utils.match_utils import player_id_to_column_id
from databallpy.utils.utils import sigmoid

logger = create_logger(__name__)


def synchronise_tracking_and_event_data(
    tracking_data: pd.DataFrame,
    event_data: pd.DataFrame,
    home_players: pd.DataFrame,
    away_players: pd.DataFrame,
    home_team_id: int | str,
    n_batches: int | str = "smart",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Function that synchronises tracking and event data using Needleman-Wunsch
       algorithmn. Based on: https://kwiatkowski.io/sync.soccer. The similarity
       function is used but information based on the specific event is added. For
       instance, when the event is a shot, the ball acceleration should be high and
       the ball should move towards the goal.

    Args:
        tracking_data (pd.DataFrame): Tracking data of the match
        event_data (pd.DataFrame): Event data of the match
        home_players (pd.DataFrame): DataFrame containing all the home players
        away_players (pd.DataFrame): DataFrame containing all the away players
        home_team_id (int | str): id of the home team
        n_batches (int, str): the number of batches that are created.
            A higher number of batches reduces the time the code takes to load, but
            reduces the accuracy for events close to the splits. Default = "smart".
            If n_batches is set to "smart", the number of batches is chosen based on
            the number of periods that the ball is set to alive. This way, the
            transition of the similarity matrix is always between two periods of
            active play. This method is optimised for events that are in active play,
            other events, such as yellow cards might not be perfectly synced.
        verbose (bool, optional): Wheter or not to print info about the progress
            in the terminal. Defaults to True.

    Currently works for the following databallpy_events:
        'pass', 'shot', and 'dribble'

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
    try:
        event_data_to_sync = event_data[
            event_data["databallpy_event"].isin(DATABALLPY_EVENTS)
        ]

        if n_batches == "smart":
            end_datetimes = create_smart_batches(tracking_data)
        else:
            end_datetimes = create_batches(
                n_batches,
                tracking_data,
            )
        logger.info(
            f"Succesfully created batches. Number of batches: {len(end_datetimes)}"
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
                    home_team_id,
                )
                event_frame_dict = _needleman_wunsch(sim_mat)

                # assign events to tracking data frames
                for event, frame in event_frame_dict.items():
                    event_id = int(event_batch.loc[event, "event_id"])
                    event_type = event_batch.loc[event, "databallpy_event"]
                    event_index = int(event_batch.loc[event, "index"])
                    tracking_frame = int(tracking_batch.loc[frame, "index"])
                    extra_tracking_info.loc[
                        tracking_frame, "databallpy_event"
                    ] = event_type
                    extra_tracking_info.loc[tracking_frame, "event_id"] = event_id
                    extra_tracking_info.loc[tracking_frame, "sync_certainty"] = sim_mat[
                        frame, event
                    ]
                    extra_event_info.loc[event_index, "tracking_frame"] = tracking_frame
                    extra_event_info.loc[event_index, "sync_certainty"] = sim_mat[
                        frame, event
                    ]

            batch_first_datetime = batch_end_datetime

        logger.info("Succesfully synchronised tracking and event data")
        return extra_tracking_info, extra_event_info
    except Exception as e:
        logger.exception(f"Failed to synchronise tracking and event data, error: {e}")
        raise e


def _create_sim_mat(
    tracking_batch: pd.DataFrame,
    event_batch: pd.DataFrame,
    home_players: pd.DataFrame,
    away_players: pd.DataFrame,
    home_team_id: int | str,
) -> np.ndarray:
    """Function that creates similarity matrix between every frame and event in batch

    Args:
        tracking_batch (pd.DataFrame): batch of tracking data
        event_batch (pd.DataFrame): batch of event data
        home_players (pd.DataFrame): DataFrame containing all the home players
        away_players (pd.DataFrame): DataFrame containing all the away players
        home_team_id (int | str): id of the home team

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

    time_diff = sigmoid(np.abs(time_diff), e=5)
    ball_diff = sigmoid(ball_loc_diff, d=5, e=6)

    for row in event_batch.itertuples():
        i = row.Index

        # create array with all cost function variables
        if row.databallpy_event == "pass":
            total_shape = (len(tracking_batch), 5)
        elif row.databallpy_event == "shot":
            total_shape = (len(tracking_batch), 6)
        elif row.databallpy_event == "dribble":
            total_shape = (len(tracking_batch), 3)

        total = np.zeros(total_shape)

        if not pd.isnull(row.player_id) and row.player_id != MISSING_INT:
            column_id_player = player_id_to_column_id(
                home_players, away_players, player_id=row.player_id
            )
            if f"{column_id_player}_x" not in tracking_batch.columns:
                player_ball_diff = [np.nan] * len(tracking_batch)
            else:
                player_ball_diff = np.hypot(
                    tracking_batch["ball_x"].values
                    - tracking_batch[f"{column_id_player}_x"].values,
                    tracking_batch["ball_y"].values
                    - tracking_batch[f"{column_id_player}_y"].values,
                )
        else:
            player_ball_diff = np.array([np.nan] * len(tracking_batch))

        total[:, :3] = np.array(
            [time_diff[:, i], ball_diff[:, i], sigmoid(player_ball_diff, d=5, e=2.5)]
        ).T

        if row.databallpy_event in ["shot", "pass"]:
            total[:, 3] = sigmoid(
                -tracking_batch["ball_acceleration"].values, d=0.2, e=-25.0
            )  # ball acceleration
            # the distance between player and ball should increase at the
            # moment of a pass or shot
            # negative is approaching, positive is moving away
            raw = np.gradient(player_ball_diff)

            total[:, 4] = sigmoid(raw, d=-8.0)

            if row.databallpy_event == "shot":
                angle = tracking_batch[
                    f"goal_angle_{['home', 'away'][row.team_id != home_team_id]}"
                    "_team"
                ].values
                total[:, 5] = sigmoid(angle, d=6, e=0.2 * np.pi)

        # take the mean of all cost function variables.
        mask = np.isnan(total).all(axis=1)
        if total[mask].shape[0] > 0:
            total[mask] = np.nanmax(total)

        sim_mat[:, i] = np.nanmean(total, axis=1)

    # replace nan values with highest value, the algorithm will not assign these
    sim_mat[np.isnan(sim_mat)] = np.nanmax(sim_mat)
    sim_mat = -sim_mat + 1  # low cost is better similarity

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
) -> pd.DataFrame:
    """Function that precomputes variables that are used in the synchronisation.
    The following variables are computed: ball_velocity, ball_acceleration,
    ball_acceleration_sqrt, goal_angle_home_team, and goal_angle_away_team.

    Args:
        tracking_data (pd.DataFrame): Tracking data of the match
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

    return tracking_data


def create_batches(
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
        tracking_data (pd.DataFrame): Tracking data of the match

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
        tracking_data (pd.DataFrame): Tracking data of the match

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
    end_alive_idxs = np.where(
        (tracking_data.iloc[:-1]["ball_status"] == "alive").values
        & (tracking_data.iloc[1:]["ball_status"] == "dead").values
    )[0]
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
    )
    if first_valid_frame_index not in start_alive_idxs:
        start_alive_idxs = np.concatenate(
            [np.array([first_valid_frame_index]), start_alive_idxs]
        )

    # create batches to loop over
    last_end_dt = None
    end_datetimes = []
    for start_idx, end_idx in zip(start_alive_idxs, end_alive_idxs):
        start_dt = tracking_data.iloc[start_idx]["datetime"]
        end_dt = tracking_data.iloc[end_idx,]["datetime"]

        if last_end_dt is not None:
            difference = start_dt - last_end_dt
            last_end_dt + difference / 2
            end_datetimes.append(last_end_dt + difference / 2)

        last_end_dt = end_dt

    # add last datetime with some room for the last events
    end_datetimes.append(last_end_dt + pd.to_timedelta(3, unit="s"))

    return end_datetimes


def align_event_data_datetime(
    event_data: pd.DataFrame, tracking_data: pd.DataFrame, offset: float = 1.0
) -> pd.DataFrame:
    """Function that aligns the datetimes of the event data and tracking data. This
    is done by substracting the difference between the first event and the first
    tracking frame from all event datetimes.

    Args:
        event_data (pd.DataFrame): Event data of the match that needs to be synced
        tracking_data (pd.DataFrame): Tracking data of the match
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
