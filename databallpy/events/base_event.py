from dataclasses import dataclass

import pandas as pd


@dataclass
class BaseEvent:
    """This is the base event class from which the specific event classes are inherited.
    It containts all the basic information that is available for every event.

    Args:
        event_id (int): distinct id of the event
        period_id (int): id of the period
        minutes (int): minute in which the event occurs
        seconds (int): seconds within the aforementioned minute where the event occurs
        datetime (pd.Timestamp): datetime at which the event occured
        start_x (float): x coordinate of the start location of the event
        start_y (float): y coordinate of the start location of the event
        team_id (int): id of the team that performed the event
    """

    event_id: int
    period_id: int
    minutes: int
    seconds: int
    datetime: pd.Timestamp
    start_x: float
    start_y: float
    team_id: int

    def __post_init__(self):
        if not isinstance(self.event_id, int):
            raise TypeError(f"event_id should be int, not {type(self.event_id)}")

        if not isinstance(self.period_id, int):
            raise TypeError(f"period_id should be int, not {type(self.period_id)}")

        if not isinstance(self.minutes, (int, float)):
            raise TypeError(f"minutes should be int, not {type(self.minutes)}")

        if not isinstance(self.seconds, (int, float)):
            raise TypeError(f"seconds should be int, not {type(self.seconds)}")

        if not isinstance(self.datetime, pd.Timestamp):
            raise TypeError(
                f"datetime should be pd.Timestamp, not {type(self.datetime)}"
            )

        if not isinstance(self.start_x, float):
            raise TypeError(f"x_start should be a float, not {type(self.start_x)}")

        if not isinstance(self.start_y, float):
            raise TypeError(f"y_start should be a float, not {type(self.start_y)}")

        if not isinstance(self.team_id, int) and not isinstance(self.team_id, str):
            raise TypeError(f"team_id should be int, not {type(self.team_id)}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseEvent):
            return False
        result = [
            self.event_id == other.event_id,
            self.period_id == other.period_id,
            self.minutes == other.minutes,
            round(self.seconds, 4) == round(other.seconds, 4),
            self.datetime == other.datetime,
            round(self.start_x, 4) == round(other.start_x, 4)
            if not pd.isnull(self.start_x)
            else pd.isnull(other.start_x),
            round(self.start_y, 4) == round(other.start_y, 4)
            if not pd.isnull(self.start_y)
            else pd.isnull(other.start_y),
            self.team_id == other.team_id,
        ]
        return all(result)
