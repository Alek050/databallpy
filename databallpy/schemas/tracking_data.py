import warnings
import pandas as pd
import pandera as pa
import pandera.extensions as extensions
from databallpy.utils.logging import create_logger, logging_wrapper
from databallpy.utils.warnings import DataBallPyWarning

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
            f"The ball status is alive for less than half of the"
            " full game. Ball status is uses for synchronisation; "
            "check the quality of the data before synchronising event and "
            "tracking data."
        )
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
    ball_possession: pa.typing.Series[str] = pa.Field(nullable=True)

    class Config:
        check_first_frame = ()
        check_ball_status = ()
