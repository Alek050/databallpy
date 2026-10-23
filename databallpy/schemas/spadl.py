import pandas as pd

try:
    import pandera.pandas as pa
except ModuleNotFoundError:
    import pandera as pa

from databallpy.utils.constants import (
    ATOMIC_SPADL_ACTION_TYPES,
    SPADL_ACTION_TYPES,
    SPADL_BODYPARTS,
    SPADL_FIELD_LENGTH,
    SPADL_FIELD_WIDTH,
    SPADL_RESULTS,
)


class SPADLSchema(pa.DataFrameModel):
    """Schema of SPADL actions in the SPADL coordinate system, mirrors the
    SPADLSchema of socceraction with the extra databallpy ``event_id`` column."""

    game_id: pa.typing.Series[object] = pa.Field(nullable=True)
    original_event_id: pa.typing.Series[object] = pa.Field(nullable=True)
    event_id: pa.typing.Series[pd.Int64Dtype] = pa.Field(nullable=True)
    action_id: pa.typing.Series[int] = pa.Field(unique=True)
    period_id: pa.typing.Series[int] = pa.Field(ge=1, le=5)
    time_seconds: pa.typing.Series[float] = pa.Field(ge=0)
    team_id: pa.typing.Series[object] = pa.Field()
    player_id: pa.typing.Series[object] = pa.Field()
    start_x: pa.typing.Series[float] = pa.Field(ge=0, le=SPADL_FIELD_LENGTH)
    start_y: pa.typing.Series[float] = pa.Field(ge=0, le=SPADL_FIELD_WIDTH)
    end_x: pa.typing.Series[float] = pa.Field(ge=0, le=SPADL_FIELD_LENGTH)
    end_y: pa.typing.Series[float] = pa.Field(ge=0, le=SPADL_FIELD_WIDTH)
    type_id: pa.typing.Series[int] = pa.Field(isin=list(range(len(SPADL_ACTION_TYPES))))
    result_id: pa.typing.Series[int] = pa.Field(isin=list(range(len(SPADL_RESULTS))))
    bodypart_id: pa.typing.Series[int] = pa.Field(isin=list(range(len(SPADL_BODYPARTS))))

    class Config:
        strict = True
        coerce = True


class AtomicSPADLSchema(pa.DataFrameModel):
    """Schema of Atomic-SPADL actions in the SPADL coordinate system, mirrors the
    AtomicSPADLSchema of socceraction with the extra databallpy ``event_id`` column."""

    game_id: pa.typing.Series[object] = pa.Field(nullable=True)
    original_event_id: pa.typing.Series[object] = pa.Field(nullable=True)
    event_id: pa.typing.Series[pd.Int64Dtype] = pa.Field(nullable=True)
    action_id: pa.typing.Series[int] = pa.Field(unique=True)
    period_id: pa.typing.Series[int] = pa.Field(ge=1, le=5)
    time_seconds: pa.typing.Series[float] = pa.Field(ge=0)
    team_id: pa.typing.Series[object] = pa.Field()
    player_id: pa.typing.Series[object] = pa.Field()
    x: pa.typing.Series[float] = pa.Field(ge=0, le=SPADL_FIELD_LENGTH)
    y: pa.typing.Series[float] = pa.Field(ge=0, le=SPADL_FIELD_WIDTH)
    dx: pa.typing.Series[float] = pa.Field(ge=-SPADL_FIELD_LENGTH, le=SPADL_FIELD_LENGTH)
    dy: pa.typing.Series[float] = pa.Field(ge=-SPADL_FIELD_WIDTH, le=SPADL_FIELD_WIDTH)
    type_id: pa.typing.Series[int] = pa.Field(
        isin=list(range(len(ATOMIC_SPADL_ACTION_TYPES)))
    )
    bodypart_id: pa.typing.Series[int] = pa.Field(isin=list(range(len(SPADL_BODYPARTS))))

    class Config:
        strict = True
        coerce = True
