from dataclasses import dataclass

import pandas as pd


@dataclass
class Event:
    """This is the base event class from which the specific event classes are inherited.
    It containts all the basic information that is available for every event.

    Args:
        event_id (int): distinct id of the event
        period_id (int): id of the period
        minutes (int): minute in which the event occurs
        seconds (int): seconds within the aforementioned minute where the event occurs
        datetime (pd.Timestamp): datetime at which the event occured
        x_start (float): x-coordinate where the event occured
        y_start (float): y-coordinate where the event occured
    """

    event_id: int
    period_id: int
    minutes: int
    seconds: int
    datetime: pd.Timestamp
    x_start: float
    y_start: float

    def __post_init__(self):
        if not isinstance(self.event_id, int):
            raise TypeError(f"event_id should be int, not {type(self.event_id)}")

        if not isinstance(self.period_id, int):
            raise TypeError(f"period_id should be int, not {type(self.period_id)}")

        if not isinstance(self.minutes, int):
            raise TypeError(f"minutes should be int, not {type(self.minutes)}")

        if not isinstance(self.seconds, int):
            raise TypeError(f"seconds should be int, not {type(self.seconds)}")

        if not isinstance(self.datetime, pd.Timestamp):
            raise TypeError(
                f"datetime should be pd.Timestamp, not {type(self.datetime)}"
            )

        if not isinstance(self.x_start, float):
            raise TypeError(f"x_start should be a float, not {type(self.x_start)}")

        if not isinstance(self.y_start, float):
            raise TypeError(f"y_start should be a float, not {type(self.y_start)}")
