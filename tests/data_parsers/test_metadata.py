import unittest

import pandas as pd

from databallpy.data_parsers.metadata import Metadata


class TestMetadata(unittest.TestCase):
    def setUp(self):
        self.match_id = 12
        self.pitch_dimensions = [100.0, 50.0]
        self.periods_frames = pd.DataFrame(
            {
                "period_id": [1, 2, 3, 4, 5],
                "start_datetime_ed": [
                    pd.to_datetime("2023-01-22T12:18:32.000", utc=True),
                    pd.to_datetime("2023-01-22T13:21:13.000", utc=True),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                ],
                "end_datetime_ed": [
                    pd.to_datetime("2023-01-22T13:04:32.000", utc=True),
                    pd.to_datetime("2023-01-22T14:09:58.000", utc=True),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                ],
            }
        )
        self.frame_rate = 10
        self.home_team_id = 10
        self.home_team_name = "Team One"
        self.home_players = pd.DataFrame({"id": [], "full_name": [], "shirt_num": []})
        self.home_score = 3
        self.home_formation = "1433"
        self.away_team_id = 11
        self.away_team_name = "Team Two"
        self.away_players = pd.DataFrame({"id": [], "full_name": [], "shirt_num": []})
        self.away_score = 2
        self.away_formation = "1442"
        self.country = "Netherlands"

    def test_metadata__eq__(self):
        metadata1 = Metadata(
            match_id=self.match_id,
            pitch_dimensions=self.pitch_dimensions,
            periods_frames=self.periods_frames,
            frame_rate=self.frame_rate,
            home_team_id=self.home_team_id,
            home_team_name=self.home_team_name,
            home_players=self.home_players,
            home_formation=self.home_formation,
            home_score=self.home_score,
            away_team_id=self.away_team_id,
            away_team_name=self.away_team_name,
            away_players=self.away_players,
            away_score=self.away_score,
            away_formation=self.away_formation,
            country=self.country,
        )

        metadata2 = Metadata(
            match_id=2,
            pitch_dimensions=self.pitch_dimensions,
            periods_frames=self.periods_frames,
            frame_rate=self.frame_rate,
            home_team_id=self.home_team_id,
            home_team_name=self.home_team_name,
            home_players=self.home_players,
            home_formation=self.home_formation,
            home_score=self.home_score,
            away_team_id=self.away_team_id,
            away_team_name=self.away_team_name,
            away_players=self.away_players,
            away_score=self.away_score,
            away_formation=self.away_formation,
            country=self.country,
        )
        assert metadata1 == metadata1
        assert metadata1 != metadata2
        assert metadata1 != "metadata"

    def test_metadata_post_init(self):
        # match id
        with self.assertRaises(TypeError):
            Metadata(
                match_id=["12"],
                pitch_dimensions=self.pitch_dimensions,
                periods_frames=self.periods_frames,
                frame_rate=self.frame_rate,
                home_team_id=self.home_team_id,
                home_team_name=self.home_team_name,
                home_players=self.home_players,
                home_formation=self.home_formation,
                home_score=self.home_score,
                away_team_id=self.away_team_id,
                away_team_name=self.away_team_name,
                away_players=self.away_players,
                away_score=self.away_score,
                away_formation=self.away_formation,
                country=self.country,
            )

        # pitch dimension
        with self.assertRaises(TypeError):
            Metadata(
                match_id=self.match_id,
                pitch_dimensions="12",
                periods_frames=self.periods_frames,
                frame_rate=self.frame_rate,
                home_team_id=self.home_team_id,
                home_team_name=self.home_team_name,
                home_players=self.home_players,
                home_formation=self.home_formation,
                home_score=self.home_score,
                away_team_id=self.away_team_id,
                away_team_name=self.away_team_name,
                away_players=self.away_players,
                away_score=self.away_score,
                away_formation=self.away_formation,
                country=self.country,
            )

        with self.assertRaises(ValueError):
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=[10.0],
                periods_frames=self.periods_frames,
                frame_rate=self.frame_rate,
                home_team_id=self.home_team_id,
                home_team_name=self.home_team_name,
                home_players=self.home_players,
                home_formation=self.home_formation,
                home_score=self.home_score,
                away_team_id=self.away_team_id,
                away_team_name=self.away_team_name,
                away_players=self.away_players,
                away_score=self.away_score,
                away_formation=self.away_formation,
                country=self.country,
            )
        with self.assertRaises(TypeError):
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=[10, 10.0],
                periods_frames=self.periods_frames,
                frame_rate=self.frame_rate,
                home_team_id=self.home_team_id,
                home_team_name=self.home_team_name,
                home_players=self.home_players,
                home_formation=self.home_formation,
                home_score=self.home_score,
                away_team_id=self.away_team_id,
                away_team_name=self.away_team_name,
                away_players=self.away_players,
                away_score=self.away_score,
                away_formation=self.away_formation,
                country=self.country,
            )

        # periods frames
        with self.assertRaises(TypeError):
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=self.pitch_dimensions,
                periods_frames=12,
                frame_rate=self.frame_rate,
                home_team_id=self.home_team_id,
                home_team_name=self.home_team_name,
                home_players=self.home_players,
                home_formation=self.home_formation,
                home_score=self.home_score,
                away_team_id=self.away_team_id,
                away_team_name=self.away_team_name,
                away_players=self.away_players,
                away_score=self.away_score,
                away_formation=self.away_formation,
                country=self.country,
            )

        with self.assertRaises(ValueError):
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=self.pitch_dimensions,
                periods_frames=pd.DataFrame({"test": []}),
                frame_rate=self.frame_rate,
                home_team_id=self.home_team_id,
                home_team_name=self.home_team_name,
                home_players=self.home_players,
                home_formation=self.home_formation,
                home_score=self.home_score,
                away_team_id=self.away_team_id,
                away_team_name=self.away_team_name,
                away_players=self.away_players,
                away_score=self.away_score,
                away_formation=self.away_formation,
                country=self.country,
            )

        with self.assertRaises(ValueError):
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=self.pitch_dimensions,
                periods_frames=pd.DataFrame({"period_id": [0, 1, 2, 3, 4]}),
                frame_rate=self.frame_rate,
                home_team_id=self.home_team_id,
                home_team_name=self.home_team_name,
                home_players=self.home_players,
                home_formation=self.home_formation,
                home_score=self.home_score,
                away_team_id=self.away_team_id,
                away_team_name=self.away_team_name,
                away_players=self.away_players,
                away_score=self.away_score,
                away_formation=self.away_formation,
                country=self.country,
            )

        with self.assertRaises(ValueError):
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=self.pitch_dimensions,
                periods_frames=pd.DataFrame({"period_id": [1, 1, 2, 3, 4, 5]}),
                frame_rate=self.frame_rate,
                home_team_id=self.home_team_id,
                home_team_name=self.home_team_name,
                home_players=self.home_players,
                home_formation=self.home_formation,
                home_score=self.home_score,
                away_team_id=self.away_team_id,
                away_team_name=self.away_team_name,
                away_players=self.away_players,
                away_score=self.away_score,
                away_formation=self.away_formation,
                country=self.country,
            )

        with self.assertRaises(ValueError):
            periods = self.periods_frames.copy()
            periods["start_datetime_ed"] = periods["start_datetime_ed"].dt.tz_localize(
                None
            )
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=self.pitch_dimensions,
                periods_frames=periods,
                frame_rate=self.frame_rate,
                home_team_id=self.home_team_id,
                home_team_name=self.home_team_name,
                home_players=self.home_players,
                home_formation=self.home_formation,
                home_score=self.home_score,
                away_team_id=self.away_team_id,
                away_team_name=self.away_team_name,
                away_players=self.away_players,
                away_score=self.away_score,
                away_formation=self.away_formation,
                country=self.country,
            )

        # frame rate
        with self.assertRaises(TypeError):
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=self.pitch_dimensions,
                periods_frames=self.periods_frames,
                frame_rate=12.0,
                home_team_id=self.home_team_id,
                home_team_name=self.home_team_name,
                home_players=self.home_players,
                home_formation=self.home_formation,
                home_score=self.home_score,
                away_team_id=self.away_team_id,
                away_team_name=self.away_team_name,
                away_players=self.away_players,
                away_score=self.away_score,
                away_formation=self.away_formation,
                country=self.country,
            )

        with self.assertRaises(ValueError):
            periods = self.periods_frames.copy()
            periods["start_datetime_ed"] = pd.to_datetime("NaT")
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=self.pitch_dimensions,
                periods_frames=periods,
                frame_rate=-10,
                home_team_id=self.home_team_id,
                home_team_name=self.home_team_name,
                home_players=self.home_players,
                home_formation=self.home_formation,
                home_score=self.home_score,
                away_team_id=self.away_team_id,
                away_team_name=self.away_team_name,
                away_players=self.away_players,
                away_score=self.away_score,
                away_formation=self.away_formation,
                country=self.country,
            )

        # team id
        with self.assertRaises(TypeError):
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=self.pitch_dimensions,
                periods_frames=self.periods_frames,
                frame_rate=self.frame_rate,
                home_team_id=999.0,
                home_team_name=self.home_team_name,
                home_players=self.home_players,
                home_formation=self.home_formation,
                home_score=self.home_score,
                away_team_id=self.away_team_id,
                away_team_name=self.away_team_name,
                away_players=self.away_players,
                away_score=self.away_score,
                away_formation=self.away_formation,
                country=self.country,
            )

        # team name
        with self.assertRaises(TypeError):
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=self.pitch_dimensions,
                periods_frames=self.periods_frames,
                frame_rate=self.frame_rate,
                home_team_id=self.home_team_id,
                home_team_name=self.home_team_name,
                home_players=self.home_players,
                home_formation=self.home_formation,
                home_score=self.home_score,
                away_team_id=self.away_team_id,
                away_team_name=["team one"],
                away_players=self.away_players,
                away_score=self.away_score,
                away_formation=self.away_formation,
                country=self.country,
            )

        # team score
        with self.assertRaises(TypeError):
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=self.pitch_dimensions,
                periods_frames=self.periods_frames,
                frame_rate=self.frame_rate,
                home_team_id=self.home_team_id,
                home_team_name=self.home_team_name,
                home_players=self.home_players,
                home_formation=self.home_formation,
                home_score=1.0,
                away_team_id=self.away_team_id,
                away_team_name=self.away_team_name,
                away_players=self.away_players,
                away_score=self.away_score,
                away_formation=self.away_formation,
                country=self.country,
            )

        with self.assertRaises(ValueError):
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=self.pitch_dimensions,
                periods_frames=self.periods_frames,
                frame_rate=self.frame_rate,
                home_team_id=self.home_team_id,
                home_team_name=self.home_team_name,
                home_players=self.home_players,
                home_formation=self.home_formation,
                home_score=self.home_score,
                away_team_id=self.away_team_id,
                away_team_name=self.away_team_name,
                away_players=self.away_players,
                away_score=-2,
                away_formation=self.away_formation,
                country=self.country,
            )

        # team formation
        with self.assertRaises(TypeError):
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=self.pitch_dimensions,
                periods_frames=self.periods_frames,
                frame_rate=self.frame_rate,
                home_team_id=self.home_team_id,
                home_team_name=self.home_team_name,
                home_players=self.home_players,
                home_formation=self.home_formation,
                home_score=self.home_score,
                away_team_id=self.away_team_id,
                away_team_name=self.away_team_name,
                away_players=self.away_players,
                away_score=self.away_score,
                away_formation=1433,
                country=self.country,
            )

        with self.assertRaises(ValueError):
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=self.pitch_dimensions,
                periods_frames=self.periods_frames,
                frame_rate=self.frame_rate,
                home_team_id=self.home_team_id,
                home_team_name=self.home_team_name,
                home_players=self.home_players,
                home_formation=self.home_formation,
                home_score=self.home_score,
                away_team_id=self.away_team_id,
                away_team_name=self.away_team_name,
                away_players=self.away_players,
                away_score=self.away_score,
                away_formation="132321",
                country=self.country,
            )

        # team players
        with self.assertRaises(TypeError):
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=self.pitch_dimensions,
                periods_frames=self.periods_frames,
                frame_rate=self.frame_rate,
                home_team_id=self.home_team_id,
                home_team_name=self.home_team_name,
                home_players={"players": ["player1", "player2"]},
                home_formation=self.home_formation,
                home_score=self.home_score,
                away_team_id=self.away_team_id,
                away_team_name=self.away_team_name,
                away_players=self.away_players,
                away_score=self.away_score,
                away_formation=self.away_formation,
                country=self.country,
            )

        with self.assertRaises(ValueError):
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=self.pitch_dimensions,
                periods_frames=self.periods_frames,
                frame_rate=self.frame_rate,
                home_team_id=self.home_team_id,
                home_team_name=self.home_team_name,
                home_players=self.home_players,
                home_formation=self.home_formation,
                home_score=self.home_score,
                away_team_id=self.away_team_id,
                away_team_name=self.away_team_name,
                away_players=self.away_players.drop("id", axis=1),
                away_score=self.away_score,
                away_formation=self.away_formation,
                country=self.country,
            )

        with self.assertRaises(TypeError):
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=self.pitch_dimensions,
                periods_frames=self.periods_frames,
                frame_rate=self.frame_rate,
                home_team_id=self.home_team_id,
                home_team_name=self.home_team_name,
                home_players=self.home_players,
                home_formation=self.home_formation,
                home_score=self.home_score,
                away_team_id=self.away_team_id,
                away_team_name=self.away_team_name,
                away_players=self.away_players,
                away_score=self.away_score,
                away_formation=self.away_formation,
                country=12,
            )
