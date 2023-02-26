import unittest

import pandas as pd

from databallpy.load_data.metadata import Metadata


class TestMetadata(unittest.TestCase):
    def setUp(self):
        self.match_id = 12
        self.pitch_dimensions = [100.0, 50.0]
        self.periods_frames = pd.DataFrame(
            {
                "period": [1, 2, 3, 4, 5],
                "start_datetime_opta": [
                    pd.to_datetime("20230122T121832+0000"),
                    pd.to_datetime("20230122T132113+0000"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                ],
                "end_datetime_opta": [
                    pd.to_datetime("20230122T130432+0000"),
                    pd.to_datetime("20230122T140958+0000"),
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
        )
        assert metadata1 == metadata1
        assert metadata1 != metadata2

    def test_metadata_post_init(self):
        # match id
        with self.assertRaises(TypeError):
            Metadata(
                match_id="12",
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
            )

        with self.assertRaises(ValueError):
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=self.pitch_dimensions,
                periods_frames=pd.DataFrame({"period": [0, 1, 2, 3, 4]}),
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
            )

        with self.assertRaises(ValueError):
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=self.pitch_dimensions,
                periods_frames=pd.DataFrame({"period": [1, 1, 2, 3, 4, 5]}),
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
            )

        with self.assertRaises(ValueError):
            Metadata(
                match_id=self.match_id,
                pitch_dimensions=self.pitch_dimensions,
                periods_frames=self.periods_frames,
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
                away_formation="13232",
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
            )
