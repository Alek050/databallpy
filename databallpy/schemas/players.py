from typing import Optional

import pandera as pa

from databallpy.utils.constants import DATABALLPY_POSITIONS


class PlayersSchema(pa.DataFrameModel):
    id: pa.typing.Series[object] = pa.Field(unique=True, coerce=True)
    full_name: pa.typing.Series[str] = pa.Field(unique=True)
    shirt_num: pa.typing.Series[int] = pa.Field(unique=True, ge=0, le=100)
    position: pa.typing.Series[str] = pa.Field(isin=DATABALLPY_POSITIONS)
    start_frame: Optional[pa.typing.Series[int]] = pa.Field()
    end_frame: Optional[pa.typing.Series[int]] = pa.Field()
    starter: pa.typing.Series[bool] = pa.Field()
