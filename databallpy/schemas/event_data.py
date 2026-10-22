from typing import Optional

import pandas as pd

try:
    import pandera.pandas as pa
except ModuleNotFoundError:
    import pandera as pa

from databallpy.utils.constants import (
    DATABALLPY_EVENTS,
    SPADL_ACTION_TYPES,
    SPADL_BODYPARTS,
    SPADL_RESULTS,
)


class EventDataSchema(pa.DataFrameModel):
    event_id: pa.typing.Series[int] = pa.Field(unique=True)
    databallpy_event: pa.typing.Series[str] = pa.Field(
        nullable=True, isin=DATABALLPY_EVENTS
    )
    period_id: pa.typing.Series[int] = pa.Field(ge=-1, le=5)
    minutes: pa.typing.Series[int] = pa.Field(ge=0, le=150)
    seconds: pa.typing.Series[float] = pa.Field(ge=0, lt=60)
    player_id: pa.typing.Series[object] = pa.Field(nullable=True, coerce=True)
    player_name: pa.typing.Series[str] = pa.Field(nullable=True)
    team_id: pa.typing.Series[object] = pa.Field(nullable=True, coerce=True)
    is_successful: pa.typing.Series[pd.BooleanDtype] = pa.Field(nullable=True)
    start_x: pa.typing.Series[float] = pa.Field(ge=-60, le=60, nullable=True)
    start_y: pa.typing.Series[float] = pa.Field(ge=-45, le=45, nullable=True)

    datetime: pa.typing.Series[object] = pa.Field(nullable=True, coerce=True)

    @pa.check("datetime")
    def is_timestamp(self, series: pa.typing.Series[object]) -> bool:
        return series.dropna().apply(lambda x: isinstance(x, pd.Timestamp)).all()

    @pa.check("datetime")
    def after_1975(self, series: pa.typing.Series[object]) -> bool:
        return (
            series.dropna()
            .apply(lambda x: x >= pd.Timestamp("1975-01-01", tz=x.tzinfo))
            .all()
        )

    @pa.check("datetime")
    def before_now(self, series: pa.typing.Series[object]) -> bool:
        return series.dropna().apply(lambda x: x <= pd.Timestamp.now(tz=x.tzinfo)).all()

    original_event_id: pa.typing.Series[object] = pa.Field(coerce=True)
    original_event: pa.typing.Series[str] = pa.Field(nullable=True)

    # SPADL vocabulary, null for events that are not on-ball actions
    action_type: pa.typing.Series[str] = pa.Field(nullable=True, isin=SPADL_ACTION_TYPES)
    action_result: pa.typing.Series[str] = pa.Field(nullable=True, isin=SPADL_RESULTS)
    action_bodypart: pa.typing.Series[str] = pa.Field(
        nullable=True, isin=SPADL_BODYPARTS
    )

    # optional
    end_x: Optional[pa.typing.Series[float]] = pa.Field(ge=-60, le=60, nullable=True)
    end_y: Optional[pa.typing.Series[float]] = pa.Field(ge=-45, le=45, nullable=True)
    to_player_id: Optional[pa.typing.Series[object]] = pa.Field(
        nullable=True, coerce=True
    )
    to_player_name: Optional[pa.typing.Series[str]] = pa.Field(nullable=True)
    event_type_id: Optional[pa.typing.Series[int]] = pa.Field(ge=-1)
    team_name: Optional[pa.typing.Series[str]] = pa.Field(nullable=True)


class EventData(pd.DataFrame):
    def __init__(self, *args, provider: str = "unspecified", **kwargs):
        super().__init__(*args, **kwargs)
        self._provider = provider

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_provider"] = self._provider
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._provider = state.get("_provider", "unspecified")

    @property
    def _constructor(self):
        def wrapper(*args, provider=self.provider, **kwargs):
            return EventData(*args, provider=provider, **kwargs)

        return wrapper

    @property
    def provider(self):
        return self._provider

    @provider.setter
    def provider(self, _):
        raise AttributeError("Cannot set provider attribute of event data")

    def to_spadl(
        self,
        home_team_id: int | str,
        pitch_dimensions: list[float],
        *,
        game_id: int | str | None = None,
        playing_direction: str = "home_oriented",
        coordinate_system: str = "databallpy",
    ) -> pd.DataFrame:
        """Convert the event data to SPADL actions (Decroos et al., 2019).

        The conversion follows socceraction as closely as possible. Only events
        with a SPADL ``action_type`` (other than ``non_action``) in periods 1 to 5
        are converted, actions without a start location, team or player are
        skipped with a warning. The end location of clearances is set to the start
        of the next action, interceptions followed by an action of the other team
        are failed, and dribbles are added between consecutive actions of the same
        team when the ball moved 3 to 60 meters in less than 10 seconds.

        Args:
            home_team_id (int | str): The id of the home team.
            pitch_dimensions (list[float]): The length and width of the pitch in
                meters.
            game_id (int | str | None, optional): The value of the game_id column.
                Defaults to None.
            playing_direction (str, optional): The playing direction of the actions.
                Defaults to "home_oriented", options are:

                - "home_oriented": the home team plays from left to right.
                - "away_oriented": the away team plays from left to right.
                - "possession_oriented": the team performing the action plays from
                  left to right, like ``socceraction.spadl.play_left_to_right``.
                - "out_of_possession_oriented": the team not performing the action
                  plays from left to right.
            coordinate_system (str, optional): Defaults to "databallpy", meters on
                the actual pitch with (0, 0) at the centre spot. With "spadl" the
                coordinates are rescaled to the 105x68 SPADL field with (0, 0) in
                the bottom left corner.

        Raises:
            NotImplementedError: if the event data is from InStat.
            ValueError: if the playing_direction or coordinate_system is not
                supported.

        Returns:
            pd.DataFrame: The SPADL actions sorted by period and time, with the
                columns game_id, original_event_id, event_id, action_id, period_id,
                time_seconds (since the start of the period), team_id, player_id,
                start_x, start_y, end_x, end_y, type_id, result_id and bodypart_id.
                The original_event_id and event_id columns link the actions to the
                event data and are null for added dribbles. The ids map to the
                names in ``databallpy.utils.constants.SPADL_ACTION_TYPES``,
                ``SPADL_RESULTS`` and ``SPADL_BODYPARTS``.

        Example:
            >>> from databallpy import get_open_game
            >>> game = get_open_game(provider="metrica")
            >>> actions = game.event_data.to_spadl(
            ...     home_team_id=game.home_team_id,
            ...     pitch_dimensions=game.pitch_dimensions,
            ...     coordinate_system="spadl",
            ... )
        """
        from databallpy.events.spadl import event_data_to_spadl

        if self.provider == "instat":
            raise NotImplementedError(
                "SPADL conversion is not supported for InStat event data, since "
                "InStat is deprecated and will be removed in version 0.10.0."
            )

        return event_data_to_spadl(
            self,
            home_team_id=home_team_id,
            pitch_dimensions=pitch_dimensions,
            game_id=game_id,
            playing_direction=playing_direction,
            coordinate_system=coordinate_system,
        )

    def to_video_analysis_xml(
        self,
        *,
        output: str = "string",
        team_id: int | str | list[int | str] | None = None,
        player_id: int | str | list[int | str] | None = None,
        min_minute: int | None = None,
        max_minute: int | None = None,
        databallpy_events: list[str] | None = None,
        original_events: list[str] | None = None,
        is_successful: bool | None = None,
        before_seconds: float = 3.0,
        after_seconds: float = 3.0,
        code_column: str = "databallpy_event",
        tag_period_starts: bool = True,
        time_decimals: int = 2,
    ) -> str:
        """Export filtered events to XML compatible with SportsCode/Longomatch.

        Parameters
        ----------
        output : str
            If ``"string"`` (default), return the XML as a string. Otherwise,
            treat the value as a file path, write the XML there, and return the
            path as a string.
        team_id : scalar or list, optional
            Filter to specific team(s).
        player_id : scalar or list, optional
            Filter to specific player(s).
        min_minute : int, optional
            Include only events at or after this minute.
        max_minute : int, optional
            Include only events at or before this minute.
        databallpy_events : list[str], optional
            Filter to specific databallpy event types.
        original_events : list[str], optional
            Filter to specific original event types.
        is_successful : bool, optional
            Filter by success status.
        before_seconds : float
            Seconds before the event to start the clip (default 3.0).
        after_seconds : float
            Seconds after the event to end the clip (default 3.0).
        code_column : str
            Column to use as the event code (default "databallpy_event").
        tag_period_starts : bool
            Whether to add period start markers (default True).
        time_decimals : int
            Decimal places for time values (default 2).

        Returns
        -------
        str
            Pretty-printed XML string when ``output="string"``, otherwise the
            path to the written file.
        """
        from databallpy.utils.constants import DATABALLPY_EVENTS
        from databallpy.utils.to_xml import Event, LabelDict, events_to_xml

        # --- Validation ---
        if before_seconds < 0:
            raise ValueError(f"before_seconds must be >= 0, got {before_seconds}")
        if after_seconds <= 0:
            raise ValueError(f"after_seconds must be > 0, got {after_seconds}")
        if code_column not in self.columns:
            raise ValueError(
                f"code_column '{code_column}' is not a column in the DataFrame. "
                f"Available columns: {list(self.columns)}"
            )
        if min_minute is not None and max_minute is not None and min_minute > max_minute:
            raise ValueError(
                f"min_minute ({min_minute}) must be <= max_minute ({max_minute})"
            )
        if databallpy_events is not None:
            unknown = [e for e in databallpy_events if e not in DATABALLPY_EVENTS]
            if unknown:
                import warnings

                warnings.warn(
                    f"Unknown databallpy_events values: {unknown}. "
                    f"Valid values are: {DATABALLPY_EVENTS}",
                    UserWarning,
                )

        # --- Filtering ---
        mask = pd.Series([True] * len(self), index=self.index)

        if team_id is not None:
            ids = [team_id] if not isinstance(team_id, list) else team_id
            mask &= self["team_id"].isin(ids)
        if player_id is not None:
            ids = [player_id] if not isinstance(player_id, list) else player_id
            mask &= self["player_id"].isin(ids)
        if min_minute is not None:
            mask &= self["minutes"] >= min_minute
        if max_minute is not None:
            mask &= self["minutes"] <= max_minute
        if databallpy_events is not None:
            mask &= self["databallpy_event"].isin(databallpy_events)
        if original_events is not None:
            mask &= self["original_event"].isin(original_events)
        if is_successful is not None:
            mask &= self["is_successful"] == is_successful

        filtered = self[mask]

        # --- Time computation ---
        # All times are absolute match seconds so that H2 follows H1 on the
        start_datetime = self.loc[
            (self["period_id"] == 1) & (self["databallpy_event"].isin(["pass", "shot"])),
            "datetime",
        ].min()

        # --- Build events dict ---
        events_dict: dict = {}

        # Period start markers
        if tag_period_starts:
            period_code_map = {1: "1H", 2: "2H", 3: "ET1", 4: "ET2", 5: "PK"}
            for pid in sorted(period_code_map.keys()):
                code = period_code_map.get(pid, f"ET{pid - 2}")
                if pid not in self["period_id"].to_list():
                    continue
                period_start_datetime = self.loc[
                    (self["period_id"] == pid)
                    & (self["databallpy_event"].isin(["pass", "shot"])),
                    "datetime",
                ].min()
                pid_t = (period_start_datetime - start_datetime).total_seconds()
                events_dict[f"period_start_{pid}"] = Event(
                    id=f"p{pid}",
                    code=code,
                    start_t=max(0.0, pid_t - float(before_seconds)),
                    end_t=pid_t + float(after_seconds),
                    labels=[LabelDict(group="Period", name=str(pid))],
                )

        # Regular events
        start_t = (
            filtered["datetime"]
            - start_datetime
            - pd.to_timedelta(before_seconds, unit="s")
        ).dt.total_seconds()
        end_t = (
            filtered["datetime"]
            - start_datetime
            + pd.to_timedelta(after_seconds, unit="s")
        ).dt.total_seconds()
        for idx, row in filtered.iterrows():
            code_val = row.get(code_column)
            if pd.isna(code_val):
                code_val = row.get("original_event")
            if pd.isna(code_val):
                code_val = "event"
            code_val = str(code_val)

            labels = []
            if not pd.isna(row.get("player_name")):
                labels.append(LabelDict(group="Player", name=str(row["player_name"])))
            team_display = (
                row.get("team_name")
                if not pd.isna(row.get("team_name"))
                else row.get("team_id")
            )
            if not pd.isna(team_display):
                labels.append(LabelDict(group="Team", name=str(team_display)))
            if code_column != "databallpy_event" and not pd.isna(
                row.get("databallpy_event")
            ):
                labels.append(
                    LabelDict(
                        group="Databallpy Event", name=str(row["databallpy_event"])
                    )
                )
            if code_column != "original_event" and not pd.isna(
                row.get("original_event")
            ):
                labels.append(
                    LabelDict(group="Original Event", name=str(row["original_event"]))
                )
            if not pd.isna(row.get("is_successful")):
                labels.append(LabelDict(group="Outcome", name=str(row["is_successful"])))

            events_dict[str(row["event_id"])] = Event(
                id=str(row["event_id"]),
                code=code_val,
                start_t=float(start_t[idx]),
                end_t=float(end_t[idx]),
                labels=labels,
            )

        xml_string = events_to_xml(events_dict, time_decimals=time_decimals)

        if output == "string":
            return xml_string

        with open(output, "w", encoding="utf-8") as f:
            f.write(xml_string)
        return output
