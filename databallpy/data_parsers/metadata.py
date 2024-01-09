from dataclasses import dataclass

import numpy as np
import pandas as pd

from databallpy.utils.logging import create_logger
from databallpy.utils.utils import MISSING_INT

LOGGER = create_logger(__name__)


@dataclass
class Metadata:
    match_id: int
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

    def __post_init__(self):
        LOGGER.info("Created a Metadata instance, checking inputs...")
        # match id
        if not isinstance(self.match_id, int):
            message = (
                f"match_id ({self.match_id}) should be an integer, not "
                f"{type(self.match_id)}"
            )
            LOGGER.error(message)
            raise TypeError(message)

        # pitch_dimensions
        if not isinstance(self.pitch_dimensions, list):
            message = (
                f"pitch_dimensions ({self.pitch_dimensions}) should be a "
                f"list, not a {type(self.pitch_dimensions)}"
            )
            LOGGER.error(message)
            raise TypeError(message)
        if not len(self.pitch_dimensions) == 2:
            message = (
                "pitch_dimensions should contain, two values: a length and a width "
                f"of the pitch, current input is {self.pitch_dimensions}"
            )
            LOGGER.error(message)
            raise ValueError(message)
        if not all([isinstance(x, float) for x in self.pitch_dimensions]):
            message = (
                "Both values in pitch dimensions should by floats, current inputs "
                f"{[type(x) for x in self.pitch_dimensions]}"
            )
            LOGGER.error(message)
            raise TypeError(message)

        # periods_frames
        if not isinstance(self.periods_frames, pd.DataFrame):
            message = (
                "periods_frames should be a pandas dataframe, not a "
                f" {type(self.periods_frames)}"
            )
            LOGGER.error(message)
            raise TypeError(message)
        if "period_id" not in self.periods_frames.columns:
            message = "'period' should be one of the columns in period_frames"
            LOGGER.error(message)
            raise ValueError(message)
        if any(
            [
                x not in self.periods_frames["period_id"].value_counts().index
                for x in [1, 2, 3, 4, 5]
            ]
        ) or not all(self.periods_frames["period_id"].value_counts() == 1):
            res = self.periods_frames["period_id"]
            message = (
                "'period' column in period_frames should contain only the values "
                f"[1, 2, 3, 4, 5]. Now it's {res}"
            )
            LOGGER.error(message)
            raise ValueError(message)
        for col in [col for col in self.periods_frames if "datetime" in col]:
            if pd.isnull(self.periods_frames[col]).all():
                continue
            if self.periods_frames[col].dt.tz is None:
                message = f"{col} column in period_frames should have a timezone"
                LOGGER.error(message)
                raise ValueError(message)

        # frame_rate
        if not pd.isnull(self.frame_rate) and not self.frame_rate == MISSING_INT:
            if not isinstance(self.frame_rate, int):
                message = (
                    f"frame_rate should be an integer, not a {type(self.frame_rate)}"
                )
                LOGGER.error(message)
                raise TypeError(message)
            if self.frame_rate < 1:
                message = f"frame_rate should be positive, not {self.frame_rate}"
                LOGGER.error(message)
                raise ValueError(message)

        # team id's
        for team, team_id in zip(
            ["home", "away"], [self.home_team_id, self.away_team_id]
        ):
            if (
                not isinstance(team_id, int)
                and not isinstance(team_id, str)
                and not team_id == MISSING_INT
            ):
                message = (
                    f"{team} team id should be an integer or string, not a "
                    f"{type(team_id)}"
                )
                LOGGER.error(message)
                raise TypeError(message)

        # team names
        for team, name in zip(
            ["home", "away"], [self.home_team_name, self.away_team_name]
        ):
            if not isinstance(name, str):
                message = f"{team} team name should be a string, not a {type(name)}"
                LOGGER.error(message)
                raise TypeError(message)

        # team scores
        for team, score in zip(["home", "away"], [self.home_score, self.away_score]):
            if not pd.isnull(score) and not score == MISSING_INT:
                if not isinstance(score, int):
                    message = (
                        f"{team} team score should be an integer, not a {type(score)}"
                    )
                    LOGGER.error(message)
                    raise TypeError(message)
                if score < 0:
                    message = f"{team} team score should be positive, not {score}"
                    LOGGER.error(message)
                    raise ValueError(message)

        # team formations
        for team, form in zip(
            ["home", "away"], [self.home_formation, self.away_formation]
        ):
            if not pd.isnull(form):
                if not isinstance(form, str):
                    message = (
                        f"{team} team formation should be a string, not a {type(form)}"
                    )
                    LOGGER.error(message)
                    raise TypeError(message)
                if len(form) > 5:
                    message = (
                        f"{team} team formation should be of length 5 or smaller, not "
                        f"{len(form)}"
                    )
                    LOGGER.error(message)
                    raise ValueError(message)

        # team players
        for team, players in zip(
            ["home", "away"], [self.home_players, self.away_players]
        ):
            if not isinstance(players, pd.DataFrame):
                message = (
                    f"{team} team players should be a pandas dataframe, not a "
                    f"{type(players)}"
                )
                LOGGER.error(message)
                raise TypeError(message)
            if not all(
                [x in players.columns for x in ["id", "full_name", "shirt_num"]]
            ):
                message = (
                    f"{team} team players should contain at least the column "
                    f"['id', 'full_name', 'shirt_num'], now its {players.columns}"
                )
                LOGGER.error(message)
                raise ValueError(message)

        # country
        if not isinstance(self.country, str):
            message = f"country should be a string, not a {type(self.country)}"
            LOGGER.error(message)
            raise TypeError(message)

    LOGGER.info("Passed all Metadata input checks.")

    def __eq__(self, other):
        if isinstance(other, Metadata):
            result = [
                self.match_id == other.match_id,
                all(
                    [
                        s == o if not np.isnan(s) else np.isnan(o)
                        for s, o in zip(self.pitch_dimensions, other.pitch_dimensions)
                    ]
                ),
                self.periods_frames.equals(other.periods_frames),
                self.frame_rate == other.frame_rate
                if not pd.isnull(self.frame_rate)
                else pd.isnull(other.frame_rate),
                self.home_team_id == other.home_team_id,
                self.home_team_name == other.home_team_name,
                self.home_players.equals(other.home_players),
                self.home_score == other.home_score
                if not pd.isnull(self.home_score)
                else pd.isnull(other.home_score),
                self.home_formation == other.home_formation,
                self.away_team_id == other.away_team_id,
                self.away_team_name == other.away_team_name,
                self.away_players.equals(other.away_players),
                self.away_score == other.away_score
                if not pd.isnull(self.away_score)
                else pd.isnull(other.away_score),
                self.away_formation == other.away_formation,
                self.country == other.country,
                self.periods_changed_playing_direction
                == other.periods_changed_playing_direction
                if self.periods_changed_playing_direction is not None
                else other.periods_changed_playing_direction is None,
            ]
            return all(result)
        else:
            return False

    def copy(self):
        return Metadata(
            match_id=self.match_id,
            pitch_dimensions=list(self.pitch_dimensions),
            periods_frames=self.periods_frames.copy(),
            frame_rate=self.frame_rate,
            home_team_id=self.home_team_id,
            home_team_name=self.home_team_name,
            home_players=self.home_players.copy(),
            home_score=self.home_score,
            home_formation=self.home_formation,
            away_team_id=self.away_team_id,
            away_team_name=self.away_team_name,
            away_players=self.away_players.copy(),
            away_score=self.away_score,
            away_formation=self.away_formation,
            country=self.country,
            periods_changed_playing_direction=list(
                self.periods_changed_playing_direction
            )
            if self.periods_changed_playing_direction is not None
            else None,
        )
