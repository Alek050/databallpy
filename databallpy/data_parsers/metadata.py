from dataclasses import dataclass, fields

import pandas as pd

from databallpy.utils.constants import MISSING_INT
from databallpy.utils.logging import logging_wrapper
from databallpy.utils.utils import _copy_value_, _values_are_equal_


@dataclass
class Metadata:
    game_id: int
    pitch_dimensions: list
    periods_frames: pd.DataFrame
    frame_rate: int

    home_team_id: int
    home_team_name: str
    home_players: pd.DataFrame
    home_score: int
    home_formation: str

    away_team_id: int
    away_team_name: str
    away_players: pd.DataFrame
    away_score: int
    away_formation: str

    country: str
    periods_changed_playing_direction: list = None

    @logging_wrapper(__file__)
    def __post_init__(self):
        # game id
        if not isinstance(self.game_id, (int, str)):
            raise TypeError(
                f"game_id ({self.game_id}) should be an integer, not "
                f"{type(self.game_id)}"
            )

        # pitch_dimensions
        if not isinstance(self.pitch_dimensions, (tuple, list)):
            raise TypeError(
                f"pitch_dimensions ({self.pitch_dimensions}) should be a "
                f"list, not a {type(self.pitch_dimensions)}"
            )
        if not len(self.pitch_dimensions) == 2:
            raise ValueError(
                "pitch_dimensions should contain, two values: a length and a width "
                f"of the pitch, current input is {self.pitch_dimensions}"
            )
        if not all([isinstance(x, float) for x in self.pitch_dimensions]):
            raise TypeError(
                "Both values in pitch dimensions should by floats, current inputs "
                f"{[type(x) for x in self.pitch_dimensions]}"
            )

        # periods_frames
        if not isinstance(self.periods_frames, pd.DataFrame):
            raise TypeError(
                "periods_frames should be a pandas dataframe, not a "
                f" {type(self.periods_frames)}"
            )
        if "period_id" not in self.periods_frames.columns:
            raise ValueError("'period' should be one of the columns in period_frames")
        if any(
            [
                x not in self.periods_frames["period_id"].value_counts().index
                for x in [1, 2, 3, 4, 5]
            ]
        ) or not all(self.periods_frames["period_id"].value_counts() == 1):
            res = self.periods_frames["period_id"]
            raise ValueError(
                "'period' column in period_frames should contain only the values "
                f"[1, 2, 3, 4, 5]. Now it's {res}"
            )
        for col in [col for col in self.periods_frames if "datetime" in col]:
            if pd.isnull(self.periods_frames[col]).all():
                continue
            if self.periods_frames[col].dt.tz is None:
                raise ValueError(f"{col} column in period_frames should have a timezone")

        # frame_rate
        if not pd.isnull(self.frame_rate) and not self.frame_rate == MISSING_INT:
            if not isinstance(self.frame_rate, int):
                raise TypeError(
                    f"frame_rate should be an integer, not a {type(self.frame_rate)}"
                )
            if self.frame_rate < 1:
                raise ValueError(f"frame_rate should be positive, not {self.frame_rate}")

        # team id's
        for team, team_id in zip(
            ["home", "away"], [self.home_team_id, self.away_team_id]
        ):
            if (
                not isinstance(team_id, int)
                and not isinstance(team_id, str)
                and not team_id == MISSING_INT
            ):
                raise TypeError(
                    f"{team} team id should be an integer or string, not a "
                    f"{type(team_id)}"
                )

        # team names
        for team, name in zip(
            ["home", "away"], [self.home_team_name, self.away_team_name]
        ):
            if not isinstance(name, str):
                raise TypeError(
                    f"{team} team name should be a string, not a {type(name)}"
                )

        # team scores
        for team, score in zip(["home", "away"], [self.home_score, self.away_score]):
            if not pd.isnull(score) and not score == MISSING_INT:
                if not isinstance(score, int):
                    raise TypeError(
                        f"{team} team score should be an integer, not a {type(score)}"
                    )
                if score < 0:
                    raise ValueError(
                        f"{team} team score should be positive, not {score}"
                    )

        # team formations
        for team, form in zip(
            ["home", "away"], [self.home_formation, self.away_formation]
        ):
            if not pd.isnull(form):
                if not isinstance(form, str):
                    raise TypeError(
                        f"{team} team formation should be a string, not a {type(form)}"
                    )
                if len(form) > 5:
                    raise ValueError(
                        f"{team} team formation should be of length 5 or smaller, not "
                        f"{len(form)}"
                    )

        # team players
        for team, players in zip(
            ["home", "away"], [self.home_players, self.away_players]
        ):
            if not isinstance(players, pd.DataFrame):
                raise TypeError(
                    f"{team} team players should be a pandas dataframe, not a "
                    f"{type(players)}"
                )
            if not all([x in players.columns for x in ["id", "full_name", "shirt_num"]]):
                raise ValueError(
                    f"{team} team players should contain at least the column "
                    f"['id', 'full_name', 'shirt_num'], now its {players.columns}"
                )

        # country
        if not isinstance(self.country, str):
            raise TypeError(f"country should be a string, not a {type(self.country)}")

    @logging_wrapper(__file__)
    def __eq__(self, other):
        if not isinstance(other, Metadata):
            return False
        for field in fields(self):
            if not _values_are_equal_(
                getattr(self, field.name), getattr(other, field.name)
            ):
                return False
        return True

    @logging_wrapper(__file__)
    def copy(self):
        copied_kwargs = {
            f.name: _copy_value_(getattr(self, f.name)) for f in fields(self)
        }
        return Metadata(**copied_kwargs)
