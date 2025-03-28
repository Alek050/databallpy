from typing import Optional

import pandas as pd
import pandera as pa

from databallpy.utils.constants import DATABALLPY_EVENTS


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
    datetime: pa.typing.Series[pd.Timestamp] = pa.Field(
        ge=pd.Timestamp("1975-01-01"), le=pd.Timestamp.now(), coerce=True
    )
    original_event_id: pa.typing.Series[object] = pa.Field(coerce=True)
    original_event: pa.typing.Series[str] = pa.Field(nullable=True)

    # optional
    end_x: Optional[pa.typing.Series[float]] = pa.Field(ge=-60, le=60, nullable=True)
    end_y: Optional[pa.typing.Series[float]] = pa.Field(ge=-45, le=45, nullable=True)
    to_player_id: Optional[pa.typing.Series[object]] = pa.Field(
        nullable=True, coerce=True
    )
    to_player_name: Optional[pa.typing.Series[str]] = pa.Field(nullable=True)
    event_type_id: Optional[pa.typing.Series[int]] = pa.Field(ge=-1)


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
