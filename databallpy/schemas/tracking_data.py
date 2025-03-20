import math
import warnings
from warnings import simplefilter

import numpy as np
import pandas as pd
import pandera as pa
import pandera.extensions as extensions
from scipy.spatial import KDTree

from databallpy.features.covered_distance import (
    _add_covered_distance_interval,
    _parse_intervals,
    _validate_inputs,
)
from databallpy.features.differentiate import _differentiate
from databallpy.features.feature_utils import _check_column_ids
from databallpy.features.filters import savgol_filter
from databallpy.features.pitch_control import get_pitch_control_single_frame
from databallpy.features.player_possession import (
    get_ball_losses_and_updated_gain_idxs,
    get_distance_between_ball_and_players,
    get_initial_possessions,
    get_start_end_idxs,
    get_valid_gains,
)
from databallpy.features.pressure import (
    calculate_l,
    calculate_variable_dfront,
    calculate_z,
)
from databallpy.utils.constants import MISSING_INT
from databallpy.utils.logging import create_logger, logging_wrapper
from databallpy.utils.warnings import DataBallPyWarning

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

LOGGER = create_logger(__name__)
logging_wrapper(__file__)


@extensions.register_check_method()
def check_first_frame(df):
    ball_alive_mask = df["ball_status"] == "alive"
    first_frame = df.loc[ball_alive_mask, "ball_x"].first_valid_index()
    check_passed = (
        abs(df.loc[first_frame, "ball_x"]) < 7.0
        and abs(df.loc[first_frame, "ball_y"]) < 5.0
    )

    if not check_passed:
        x_start = df.loc[first_frame, "ball_x"]
        y_start = df.loc[first_frame, "ball_y"]
        message = (
            "The middle point of the pitch should be (0, 0), "
            f"now the kick-off is at ({x_start}, {y_start}). "
            "Either the recording has started too late or the ball_status "
            "is not set to 'alive' in the beginning. Please check and "
            " change the tracking data if desired."
            "\n NOTE: The quality of the synchronisation of the tracking "
            "and event data might be affected."
        )
        LOGGER.warning(message)
        warnings.warn(message=message, category=DataBallPyWarning)

    return True


@extensions.register_check_method()
def check_ball_status(df):
    frames_alive = df["ball_status"].value_counts()["alive"]
    check_passed = frames_alive > (len(df) / 2)

    if not check_passed:
        message = (
            "The ball status is alive for less than half of the"
            " full game. Ball status is uses for synchronisation; "
            "check the quality of the data before synchronising event and "
            "tracking data."
        )
        LOGGER.warning(message)
        warnings.warn(message=message, category=DataBallPyWarning)

    return True


@extensions.register_check_method()
def check_all_locations(df):
    cols = [x[:-2] for x in df.columns if x.endswith("_x")]

    message = None
    for col_id in cols:
        if f"{col_id}_y" not in df.columns:
            message = f"Missing column {col_id}_y. Please check the column names."
            break
        if not df[f"{col_id}_x"].abs().max() < 65:
            message = f"Column {col_id}_x has values outside the pitch dimensions."
            break
        if not df[f"{col_id}_y"].abs().max() < 45:
            message = f"Column {col_id}_y has values outside the pitch dimensions."
            break
    if message is not None:
        LOGGER.warning(message)
        warnings.warn(message=message, category=DataBallPyWarning)

    return True


class TrackingDataSchema(pa.DataFrameModel):
    frame: pa.typing.Series[int] = pa.Field(unique=True)
    datetime: pa.typing.Series[pd.Timestamp] = pa.Field(
        ge=pd.Timestamp("1975-01-01"), le=pd.Timestamp.now(), coerce=True, nullable=True
    )
    ball_x: pa.typing.Series[float] = pa.Field(ge=-60, le=60, nullable=True)
    ball_y: pa.typing.Series[float] = pa.Field(ge=-45, le=45, nullable=True)
    ball_z: pa.typing.Series[float] = pa.Field(ge=-5, le=45, nullable=True)
    ball_status: pa.typing.Series[str] = pa.Field(isin=["alive", "dead"], nullable=True)
    team_possession: pa.typing.Series[str] = pa.Field(nullable=True)

    class Config:
        check_first_frame = ()
        check_ball_status = ()
        check_all_locations = ()


class TrackingData(pd.DataFrame):
    """This is the tracking data class. It contains the tracking data for every
    frame as well as the provider and frame_rate. Additionaly it contains some
    basic functions to add columns to the tracking data or manipulate existing columns

    Args:
        tracking_data (pd.DataFrame): tracking data of the game
        provider (str): provider of the tracking data
        frame_rate (int): framerate of the tracking data
    """

    def __init__(
        self,
        *args,
        provider: str = "unspecified",
        frame_rate: int = MISSING_INT,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._provider = provider
        self._frame_rate = frame_rate

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_provider"] = self._provider
        state["_frame_rate"] = self._frame_rate
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._provider = state.get("_provider", "unspecified")
        self._frame_rate = state.get("_frame_rate", MISSING_INT)

    @property
    def _constructor(self):
        def wrapper(*args, provider=self.provider, frame_rate=self.frame_rate, **kwargs):
            return TrackingData(
                *args, provider=provider, frame_rate=frame_rate, **kwargs
            )

        return wrapper

    @property
    def provider(self):
        return self._provider

    @provider.setter
    def provider(self, _):
        raise AttributeError("Cannot set provider attribute of tracking data")

    @property
    def frame_rate(self):
        return self._frame_rate

    @frame_rate.setter
    def frame_rate(self, _):
        raise AttributeError("Cannot set frame rate attribute of tracking data")

    def add_velocity(
        self,
        column_ids: str | list[str],
        filter_type: str = None,
        window_length: int = 7,
        polyorder: int = 2,
        max_velocity: float = np.inf,
    ) -> None:
        """Function that adds velocity columns to the tracking data based on the position
           columns

        Args:
            self
            column_ids (str | list[str]): columns for which velocity should be calculated.
            filter_type (str, optional): filter type to use. Defaults to None.
                Options are `moving_average` and `savitzky_golay`.
            window_length (int, optional): window size for the filter. Defaults to 7.
            polyorder (int, optional): polynomial order for the filter. Defaults to 2.
            max_velocity (float, optional): maximum value for the velocity.
                Defaults to np.inf.

        Returns:
            None

        Raises:
            ValueError: if filter_type is not one of `moving_average`, `savitzky_golay`,
                or None.

        Note:
            The function will delete the columns in input_columns with the velocity if
            they already exist.
        """

        if isinstance(column_ids, str):
            column_ids = [column_ids]

        if filter_type not in ["moving_average", "savitzky_golay", None]:
            raise ValueError(
                "filter_type should be one of: 'moving_average', "
                f"'savitzky_golay', None, got: {filter_type}"
            )

        _differentiate(
            self,
            new_name="velocity",
            metric="",
            frame_rate=self.frame_rate,
            filter_type=filter_type,
            window=window_length,
            poly_order=polyorder,
            column_ids=column_ids,
            max_val=max_velocity,
            inplace=True,
        )

    def add_acceleration(
        self,
        column_ids: str | list[str],
        filter_type: str = None,
        window_length: int = 25,
        polyorder: int = 2,
        max_acceleration: float = np.inf,
    ) -> None:
        """Function that adds acceleration columns to the tracking data based on the
           position columns

        Args:
            self,
            column_ids (str | list[str]): columns for which acceleration should be
                calculated
            filter_type (str, optional): filter type to use. Defaults to None.
                Options are `moving_average` and `savitzky_golay`.
            window_length (int, optional): window size for the filter. Defaults to 25.
            polyorder (int, optional): polynomial order for the filter. Defaults to 2.
            max_acceleration (float, optional): maximum value for the acceleration.
                Defaults to np.inf.

        Returns:
            None

        Raises:
            ValueError: if filter_type is not one of `moving_average`, `savitzky_golay`,
                or None.
            ValueError: if velocity was not found in the DataFrame for the input_columns.

        Note:
            The function will delete the columns in input_columns with the acceleration if
            they already exist.
        """

        if isinstance(column_ids, str):
            column_ids = [column_ids]

        if filter_type not in ["moving_average", "savitzky_golay", None]:
            raise ValueError(
                "filter_type should be one of: 'moving_average', "
                f"'savitzky_golay', None, got: {filter_type}"
            )
        for column_id in column_ids:
            if (
                column_id + "_vx" not in self.columns
                or column_id + "_vy" not in self.columns
            ):
                raise ValueError(
                    f"Velocity was not found for {column_id} in the DataFrame. "
                    " Please calculate velocity first using get_velocity() function."
                )

        _differentiate(
            self,
            new_name="acceleration",
            metric="v",
            frame_rate=self.frame_rate,
            filter_type=filter_type,
            window=window_length,
            poly_order=polyorder,
            column_ids=column_ids,
            max_val=max_acceleration,
            inplace=True,
        )

    def add_individual_player_possession(
        self,
        pz_radius: float = 1.5,
        bv_threshold: float = 5.0,
        ba_threshold: float = 10.0,
        min_frames_pz: int = 0,
    ) -> None:
        """Function to calculate the individual player possession based on the tracking
        data. The method uses the methodology of the paper of  Vidal-Codina et al. (2022):
        "Automatic Event Detection in Football Using Tracking Data".

        Args:
            self.
            pz_radius (float, optional): The radius of the possession zone constant.
                Defaults to 1.5.
            bv_threshold (float, optional): The ball velocity threshold in m/s.
                Defaults to 5.0.
            ba_threshold (float, optional): The ball angle threshold in degrees.
                Defaults to 10.0.
            min_frames_pz (int, optional): The minimum number of frames that the ball
                has to be in the possession zone to be considered as a possession.
                Defaults to 0.

        Returns:
            None
        """
        if "ball_velocity" not in self.columns:
            raise ValueError(
                "The tracking data should have a column 'ball_velocity'. Use the "
                "add_velocity function to add the ball velocity."
            )

        distances_df = get_distance_between_ball_and_players(self)
        initial_possession = get_initial_possessions(pz_radius, distances_df)
        possession_start_idxs, possession_end_idxs = get_start_end_idxs(
            initial_possession
        )
        valid_gains = get_valid_gains(
            self,
            possession_start_idxs,
            possession_end_idxs,
            bv_threshold,
            ba_threshold,
            min_frames_pz,
        )
        valid_gains_start_idxs, ball_losses_idxs = get_ball_losses_and_updated_gain_idxs(
            possession_start_idxs, possession_end_idxs, valid_gains, initial_possession
        )

        possession = np.full(len(self), None, dtype=object)
        for start, end in zip(valid_gains_start_idxs, ball_losses_idxs):
            possession[start:end] = initial_possession[start]

        alive_mask = self["ball_status"] == "alive"
        possession[~alive_mask] = None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
            self["player_possession"] = possession

    def get_covered_distance(
        self,
        column_ids: list[str],
        velocity_intervals: tuple[float, ...] | tuple[tuple[float, ...], ...] = (),
        acceleration_intervals: tuple[float, ...] | tuple[tuple[float, ...], ...] = (),
        start_idx: int | None = None,
        end_idx: int | None = None,
    ) -> pd.DataFrame:
        """Calculates the distance covered based on the velocity magnitude at each frame.
            This function requires the `add_velocity` function to be called. Optionally,
            it can also calculate the distance covered within specified velocity and/or
            acceleration intervals.

        Args:
            self.
            column_ids (list[str]): columns for which covered distance should be
                calculated
            velocity_intervals (optional): tuple that contains the velocity interval(s).
                Defaults to ()
            acceleration_intervals (optional): tuple that contains the acceleration
                interval(s). Defaults to ()
            start_idx (int, optional): start index of the tracking data. Defaults to None.
            end_idx (int, optional): end index of the tracking data. Defaults to None

        Returns:
            pd.DataFrame: DataFrame with the covered distance for each player. The
            columns are the player_ids and the rows are the covered distance for each
            player. If velocity_intervals or acceleration_intervals are provided, the
            columns will be the player_ids and the intervals. The rows will be the
            covered distance for each player within the specified intervals.

        Notes:
            The function requires the velocity for every player calculated with the
            add_velocity function. The acceleration for every player depends on the
            presence of acceleration intervals in the input
        """

        _validate_inputs(
            self,
            column_ids,
            self.frame_rate,
            acceleration_intervals,
            start_idx,
            end_idx,
        )

        column_ids = sorted(column_ids)
        velocity_intervals = (
            _parse_intervals(velocity_intervals) if len(velocity_intervals) > 0 else []
        )
        acceleration_intervals = (
            _parse_intervals(acceleration_intervals)
            if len(acceleration_intervals) > 0
            else []
        )
        result_dict = (
            {"total_distance": []}
            | {
                f"total_distance_velocity_{interval[0]}_{interval[1]}": []
                for interval in velocity_intervals
            }
            | {
                f"total_distance_acceleration_{interval[0]}_{interval[1]}": []
                for interval in acceleration_intervals
            }
        )

        tracking_data_velocity = pd.concat(
            [self[player_id + "_velocity"] for player_id in column_ids], axis=1
        ).fillna(0)
        tracking_data_velocity.columns = tracking_data_velocity.columns.str.replace(
            "_velocity", ""
        )
        distance_per_frame = tracking_data_velocity / self.frame_rate

        start_idx = start_idx if start_idx is not None else self.index[0]
        end_idx = end_idx if end_idx is not None else self.index[-1]
        distance_per_frame = distance_per_frame.loc[start_idx:end_idx]
        self = self.loc[start_idx:end_idx]

        result_dict["total_distance"] = distance_per_frame.sum().values

        for intervals, interval_name in zip(
            [velocity_intervals, acceleration_intervals], ["velocity", "acceleration"]
        ):
            if len(intervals) > 0:
                result_dict = _add_covered_distance_interval(
                    result_dict,
                    interval_name,
                    self,
                    distance_per_frame,
                    intervals,
                    column_ids,
                )

        return pd.DataFrame(result_dict, index=column_ids)

    def filter_tracking_data(
        self,
        column_ids: str | list[str],
        filter_type: str = "savitzky_golay",
        window_length: int = 7,
        polyorder: int = 2,
    ) -> None:
        """Function to filter tracking data in specified DataFrame columns.

        Args:
            self.
            column_ids (str| list[str]): List of column IDs to apply the filter to.
            filter_type (str, optional): Type of filter to use. Defaults to
                "savitzky_golay". Options: {"moving_average", "savitzky_golay"}.
            window_length (int, optional): Window length of the filter. Defaults to 7.
            polyorder (int, optional): Polyorder to use when the savitzky_golay filter
                is selected. Defaults to 2.

        Returns:
            None

        """
        if isinstance(column_ids, str):
            column_ids = [column_ids]
        _check_column_ids(self, column_ids)
        if not isinstance(window_length, int):
            raise TypeError(
                f"window_length should be of type int, not {type(window_length)}"
            )
        if not isinstance(polyorder, int):
            raise TypeError(f"polyorder should be of type int, not {type(polyorder)}")
        if filter_type not in ["moving_average", "savitzky_golay"]:
            raise ValueError(
                "filter_type should be one of: 'moving_average', 'savitzky_golay'"
                f", got: {filter_type}"
            )

        xy_columns = [
            col
            for col in self.columns
            if "".join(col.split("_")[:-1]) in column_ids and col[-1] in ["x", "y"]
        ]
        for col in xy_columns:
            if filter_type == "savitzky_golay":
                self[col] = savgol_filter(
                    self[col].values,
                    window_length=window_length,
                    polyorder=polyorder,
                    mode="interp",
                )
            elif filter_type == "moving_average":
                self[col] = np.convolve(
                    self[col], np.ones(window_length) / window_length, mode="same"
                )

    def get_pressure_on_player(
        self,
        index: int,
        column_id: str,
        pitch_size: list[float, float],
        d_front: str | float = "variable",
        d_back: float = 3.0,
        q: float = 1.75,
    ) -> np.array:
        """
        Function to calculate the pressure in accordance with "Visual Analysis of Pressure
        in Soccer", Adrienko et al (2016). In short: pressure is determined as the sum of
        pressure of all opponents, which is a function of the angle and the distance to the
        player. This function calculates the pressure for a single player.

        Args:
            self.
            index: int, index of the frame for which to analyse pressure.
            column_id: str, column name of which player to analyse.
            pitch_size: list, length and width of the pitch.
            d_front: numeric or str, distance in meters of the front of the pressure oval
                     if "variable": d_front will be variable based on the location on
                     the field from the article of Mat Herold et al (2022).
            d_back: float, dinstance in meters of the back of the pressure oval.
            q: float, quotient of how fast pressure should increase/decrease as distance.
               to the player changes.
        Returns:
            np.array: pressure on player of the specified frame.
        """
        if index not in self.index:
            raise ValueError(f"index should be in game.tracking_data.index, not {index}")

        td_frame = self.loc[index, :]

        if d_front == "variable":
            d_front = calculate_variable_dfront(
                td_frame, column_id, pitch_length=pitch_size[0]
            )

        team = column_id[:4]
        opponent_team = "away" if team == "home" else "home"
        tot_pressure = 0
        player_xy = [td_frame[column_id + "_x"], td_frame[column_id + "_y"]]

        for opponent_column_id in [
            x[:-2] for x in td_frame.index if opponent_team in x and "_x" in x
        ]:
            opponent_xy = [
                td_frame[opponent_column_id + "_x"],
                td_frame[opponent_column_id + "_y"],
            ]
            player_opponent_distance = math.dist(player_xy, opponent_xy)
            # opponent not close enough to exert pressure on the player
            if player_opponent_distance > max([d_front, d_back]):
                continue

            z = calculate_z(
                td_frame, column_id, opponent_column_id, pitch_length=pitch_size[0]
            )
            variable_l = calculate_l(d_back, d_front, z)

            current_pressure = (
                pd.to_numeric(
                    (1 - player_opponent_distance / variable_l), errors="coerce"
                ).clip(0)
                ** q
                * 100
            )

            current_pressure = 0 if pd.isnull(current_pressure) else current_pressure
            tot_pressure += current_pressure

        return tot_pressure

    def get_pitch_control(
        self,
        pitch_dimensions: list[float, float],
        n_x_bins: int = 106,
        n_y_bins: int = 68,
        start_idx: int | None = None,
        end_idx: int | None = None,
    ) -> np.ndarray:
        """
        Calculate the pitch control surface for a given period of time. The pitch control
        surface is the sum of the team influences of the two teams. The team influence is
        the sum of the individual player influences of the team. The player influence is
        calculated using the statistical technique presented in the article "Wide Open
        Spaces" by Fernandez & Born (2018). It incorporates the position, velocity, and
        distance to the ball of a given player to determine the influence degree at each
        location on the field. The bivariate normal distribution is utilized to model the
        player's influence, and the result is normalized to obtain values within a [0, 1]
        range.
        The values are then passed through a sigmoid function to obtain the pitch control
        values within a [0, 1] range. Values near 1 indicate high pitch control by the home
        team, while values near 0 indicate high pitch control by the away team.

        Args:
            self.
            pitch_dimensions (list[float, float]): The dimensions of the pitch.
            n_x_bins (int, optional): The number of cells in the width (x) direction.
                Defaults to 106.
            n_y_bins (int, optional): The number of cells in the height (y) direction.
                Defaults to 68.
            start_idx (int, optional): The starting index of the period. Defaults to None.
            end_idx (int, optional): The ending index of the period. Defaults to None.

        Returns:
            np.ndarray: 3d pitch control values across the grid.
                Size is (len(tracking_data), grid[0].shape[0], grid[0].shape[1]).
        """

        start_idx = self.index[0] if start_idx is None else start_idx
        end_idx = self.index[-1] if end_idx is None else end_idx
        tracking_data = self.loc[start_idx:end_idx]

        pitch_control = np.zeros(
            (len(tracking_data), n_y_bins, n_x_bins), dtype=np.float32
        )

        # precompute player ball distances
        col_ids = [
            x[:-2]
            for x in tracking_data.columns
            if ("home" in x or "away" in x) and x[-2:] == "_x"
        ]
        player_ball_distances = pd.DataFrame(columns=col_ids, index=tracking_data.index)
        for col_id in col_ids:
            player_ball_distances[col_id] = np.linalg.norm(
                tracking_data[[f"{col_id}_x", f"{col_id}_y"]].values
                - tracking_data[["ball_x", "ball_y"]].values,
                axis=1,
            )

        for i, idx in enumerate(tracking_data.index):
            pitch_control[i] = get_pitch_control_single_frame(
                tracking_data.loc[idx],
                pitch_dimensions,
                n_x_bins,
                n_y_bins,
                player_ball_distances=player_ball_distances.loc[idx],
            )
        return np.array(pitch_control)

    def get_approximate_voronoi(
        self,
        pitch_dimensions: list[float, float],
        n_x_bins: int = 106,
        n_y_bins: int = 68,
        start_idx: int | None = None,
        end_idx: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the nearest player to each cell center in a grid of cells covering the
        pitch.

        Args:
            self.
            pitch_dimensions (list[float, float]): The dimensions of the pitch.
            n_x_bins (int, optional): The number of cells in the width (x) direction.
                Defaults to 106.
            n_y_bins (int, optional): The number of cells in the height (y) direction.
                Defaults to 68.
            start_idx (int, optional): The starting index of the period. Defaults to None.
            end_idx (int, optional): The ending index of the period. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]: The distances to the nearest player for each
                cell center and the column ids of the nearest player. If tracking_data is
                a pd.Series, the shape will be (n_y_bins x n_x_bins), otherwise
                (len(tracking_data) x n_y_bins x n_x_bins).
        """
        start_idx = self.index[0] if start_idx is None else start_idx
        end_idx = self.index[-1] if end_idx is None else end_idx
        tracking_data = self.loc[start_idx:end_idx]

        pitch_length, pitch_width = pitch_dimensions
        x_bins = np.linspace(-pitch_length / 2, pitch_length / 2, n_x_bins + 1)
        y_bins = np.linspace(-pitch_width / 2, pitch_width / 2, n_y_bins + 1)
        cell_centers_x, cell_centers_y = np.meshgrid(
            x_bins[:-1] + np.diff(x_bins) / 2, y_bins[:-1] + np.diff(y_bins) / 2
        )

        all_distances = np.empty(
            (len(tracking_data), n_y_bins, n_x_bins), dtype=np.float32
        )
        all_assigned_players = np.empty(
            (len(tracking_data), n_y_bins, n_x_bins), dtype="U7"
        )
        for i, (_, frame) in enumerate(tracking_data.iterrows()):
            player_column_ids = np.array(
                [
                    column[:-2]
                    for column in frame.index
                    if column[-2:] in ["_x", "_y"]
                    and not pd.isnull(frame[column])
                    and "ball" not in column
                ]
            )
            player_positions = np.array(
                [
                    [frame[column + "_x"], frame[column + "_y"]]
                    for column in player_column_ids
                ]
            ).astype(np.float64)

            tree = KDTree(player_positions)
            cell_centers = np.column_stack(
                (cell_centers_x.ravel(), cell_centers_y.ravel())
            )
            distances, nearest_player_indices = tree.query(cell_centers)

            all_assigned_players[i] = player_column_ids[nearest_player_indices].reshape(
                n_y_bins, n_x_bins
            )
            all_distances[i] = distances.reshape(n_y_bins, n_x_bins)

        if all_distances.shape[0] == 1:
            all_distances = all_distances[0]
            all_assigned_players = all_assigned_players[0]

        return all_distances, all_assigned_players

    def add_team_possession(
        self, event_data: pd.DataFrame, home_team_id: int, allow_overwrite: bool = False
    ) -> None | pd.DataFrame:
        """Function to add a column 'team_possession' to the tracking data, indicating
        which team has possession of the ball at each frame, either 'home' or 'away'.

        Raises:
            ValueError: If the tracking and event data are not synchronised.
            ValueError: If the home_team_id is not in the event data.

        Args:
            self
            event_data (EventData): Event data for a game
            home_team_id (int): The ID of the home team.
            allow_overwrite (bool, optional): If "team_possession" column has non null
                values, allow_overwrite should be set to true before the function is
                executed. Defaults to False.

        Returns:
            None
        """
        if not pd.isnull(self["team_possession"]).all() and not allow_overwrite:
            warnings.warn(
                "The 'team_possession' column is not empty. If you want to overwrite "
                "the column, set allow_overwrite=True.",
                category=DataBallPyWarning,
                stacklevel=2,
            )
            return

        if "event_id" not in self.columns:
            raise ValueError(
                "Tracking and event data are not synchronised, please synchronise the"
                " data first"
            )
        if home_team_id not in event_data.team_id.unique():
            raise ValueError(
                "The home team ID is not in the event data, please check"
                " the home team ID"
            )

        on_ball_events = ["pass", "dribble", "shot"]
        current_team_id = event_data.loc[
            ~pd.isnull(event_data["databallpy_event"]), "team_id"
        ].iloc[0]
        start_idx = 0
        self["team_possession"] = None
        for event_id in [x for x in self.event_id if x != MISSING_INT]:
            event = event_data[event_data.event_id == event_id].iloc[0]
            if (
                event["databallpy_event"] in on_ball_events
                and event.team_id != current_team_id
                and event.is_successful == 1
            ):
                end_idx = self[self.event_id == event_id].index[0]
                team = "home" if current_team_id == home_team_id else "away"
                self.loc[start_idx:end_idx, "team_possession"] = team

                current_team_id = event.team_id
                start_idx = end_idx

        last_team = "home" if current_team_id == home_team_id else "away"
        self.loc[start_idx:, "team_possession"] = last_team

    def to_long_format(self) -> pd.DataFrame:
        """Function that moves from the base format, with a row for every frame,
        to a long format, with a row for every frame/column_id combination

        The ball/team information will be added to every row

        returns: pd.DataFrame
        """
        df_players = []
        player_cols = [
            x[:-2]
            for x in self.columns
            if (x.startswith("home_") or x.startswith("away_")) and x.endswith("_x")
        ]
        for player in ["ball"] + player_cols:
            if player == "ball":
                value_cols = [
                    x.split("_")[1]
                    for x in self.columns
                    if player + "_" in x and "status" not in x
                ]
            else:
                value_cols = [x.split("_")[2] for x in self.columns if player + "_" in x]
            df_player = self[["frame"] + [player + "_" + x for x in value_cols]].copy()
            df_player.rename(
                columns={player + "_" + x: x for x in value_cols}, inplace=True
            )
            df_player.insert(1, "column_id", player)

            df_players.append(df_player)

        df_long = pd.concat(df_players, axis=0).reset_index(drop=True)

        used_cols = [
            player + "_" + value
            for player in player_cols + ["ball"]
            for value in df_long.columns[2:]
        ]
        unused_cols = [col for col in self.columns if col not in used_cols]
        return pd.DataFrame(df_long.merge(self[unused_cols], on="frame"))
