from dataclasses import dataclass

import numpy as np
import pandas as pd

from databallpy.load_data.event_data.base_event import BaseEvent


@dataclass
class PassEvent(BaseEvent):
    """This is the pass event class. It contains all the information that is available
    for a pass event.

    Args:
        event_id (int): distinct id of the pass event
        period_id (int): id of the period
        minutes (int): minute in which the pass occurs
        seconds (int): seconds within the aforementioned minute where the pass occurs
        datetime (pd.Timestamp): datetime at which the pass occured
        start_x (float): x coordinate of the start location of the pass
        start_y (float): y coordinate of the start location of the pass
        team_id (int): id of the team that performed the pass
        outcome (str): outcome of the pass, options are:
            ['successful', 'unsuccessful', 'offside', 'results_in_shot',
            'assist', 'fair_play', 'not_specified']
        player_id (int): id of the player that performed the pass
        end_x (float): x coordinate of the end location of the pass
        end_y (float): y coordinate of the end location of the pass
        length (float): length of the pass
        angle (float): angle of the pass
        pass_type (str): type of the pass, options are:
            ['long_ball', 'cross', 'through_ball', 'chipped', 'lay-off', 'lounge',
            'flick_on', 'pull_back', 'switch_off_play', 'not_specified']
        set_piece (str): type of set piece, options are:
            ['goal_kick', 'free_kick', 'throw_in', 'corner_kick', 'kick_off',
            'penalty', 'no_set_piece', unspecified_set_piece]


    Raises:
        TypeError: If any of the inputtypes is not correct
    """

    outcome: str
    player_id: int
    end_x: float
    end_y: float
    pass_type: str
    set_piece: str
    length: float = np.nan
    angle: float = np.nan

    def copy(self):
        return PassEvent(
            event_id=self.event_id,
            period_id=self.period_id,
            minutes=self.minutes,
            seconds=self.seconds,
            datetime=self.datetime,
            start_x=self.start_x,
            start_y=self.start_y,
            team_id=self.team_id,
            outcome=self.outcome,
            player_id=self.player_id,
            end_x=self.end_x,
            end_y=self.end_y,
            length=self.length,
            angle=self.angle,
            pass_type=self.pass_type,
            set_piece=self.set_piece,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PassEvent):
            return False
        result = [
            super().__eq__(other),
            self.team_id == other.team_id,
            self.outcome == other.outcome,
            self.player_id == other.player_id,
            round(self.end_x, 4) == round(other.end_x, 4)
            if not pd.isnull(self.end_x)
            else pd.isnull(other.end_x),
            round(self.end_y, 4) == round(other.end_y, 4)
            if not pd.isnull(self.end_y)
            else pd.isnull(other.end_y),
            self.length == other.length
            if not pd.isnull(self.length)
            else pd.isnull(other.length),
            self.angle == other.angle
            if not pd.isnull(self.angle)
            else pd.isnull(other.angle),
            self.pass_type == other.pass_type,
            self.set_piece == other.set_piece,
        ]
        return all(result)

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.outcome, (str, type(None))):
            raise TypeError(f"outcome should be str, not {type(self.outcome)}")

        valid_outcomes = [
            "successful",
            "unsuccessful",
            "offside",
            "results_in_shot",
            "assist",
            "fair_play",
            "not_specified",
            None,
        ]
        if self.outcome not in valid_outcomes:
            raise ValueError(
                f"outcome should be one of {valid_outcomes}, not {self.outcome}"
            )

        if not isinstance(self.player_id, int):
            raise TypeError(f"player_id should be int, not {type(self.player_id)}")

        values = [self.end_x, self.end_y, self.length, self.angle]
        names = ["end_x", "end_y", "length", "angle"]
        for value, name in zip(values, names):
            if not isinstance(value, float):
                raise TypeError(f"{name} should be float, not {type(value)}")

        if not isinstance(self.pass_type, str):
            raise TypeError(f"pass_type should be str, not {type(self.pass_type)}")
        valid_pass_types = [
            "long_ball",
            "cross",
            "through_ball",
            "chipped",
            "lay-off",
            "lounge",
            "flick_on",
            "pull_back",
            "switch_off_play",
            "not_specified",
            "assist",
        ]
        if self.pass_type not in valid_pass_types:
            raise ValueError(
                f"pass_type should be one of {valid_pass_types}, not {self.pass_type}"
            )

        if not isinstance(self.set_piece, str):
            raise TypeError(f"set_piece should be str, not {type(self.set_piece)}")

        valid_set_pieces = [
            "goal_kick",
            "free_kick",
            "throw_in",
            "corner_kick",
            "kick_off",
            "penalty",
            "no_set_piece",
            "unspecified_set_piece",
        ]
        if self.set_piece not in valid_set_pieces:
            raise ValueError(
                f"set_piece should be one of {valid_set_pieces}, not {self.set_piece}"
            )
