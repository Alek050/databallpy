import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from databallpy.data_parsers.metadata import Metadata
from databallpy.utils.constants import MISSING_INT
from databallpy.utils.get_game import get_game


class TestStarterIntegration(unittest.TestCase):
    """Integration tests to verify starter information is added in get_game functions"""

    @patch("databallpy.utils.get_game.load_tracking_data")
    @patch("databallpy.utils.get_game._quality_check_tracking_data")
    def test_get_game_adds_starter_info_with_tracking_only(
        self, mock_quality_check, mock_load_tracking
    ):
        """Test that get_game adds starter information when only tracking data is provided"""

        # Create mock tracking data with players
        tracking_data = pd.DataFrame(
            {
                "frame": [0, 1, 2],
                "home_1_x": [10.0, 11.0, 12.0],
                "home_2_x": [15.0, 16.0, 17.0],
                "away_1_x": [-10.0, -11.0, -12.0],
                "away_2_x": [-15.0, -16.0, -17.0],
            }
        )

        # Create mock metadata without starter information
        home_players = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "full_name": ["Player 1", "Player 2", "Player 3"],
                "shirt_num": [1, 2, 3],
                "position": ["goalkeeper", "defender", "midfielder"],
                "start_frame": [0, 0, 100],
                "end_frame": [1000, 1000, 1000],
            }
        )
        away_players = pd.DataFrame(
            {
                "id": [11, 12, 13],
                "full_name": ["Player 11", "Player 12", "Player 13"],
                "shirt_num": [1, 2, 3],
                "position": ["goalkeeper", "defender", "midfielder"],
                "start_frame": [0, 0, 100],
                "end_frame": [1000, 1000, 1000],
            }
        )

        periods_frames = pd.DataFrame(
            {
                "period_id": [1, 2, 3, 4, 5],
                "start_frame": [0, 500, MISSING_INT, MISSING_INT, MISSING_INT],
                "end_frame": [499, 999, MISSING_INT, MISSING_INT, MISSING_INT],
            }
        )

        metadata = Metadata(
            game_id=1,
            pitch_dimensions=[105.0, 68.0],
            periods_frames=periods_frames,
            frame_rate=10,
            home_team_id=1,
            home_team_name="Home Team",
            home_players=home_players.copy(),
            home_score=0,
            home_formation="4-3-3",
            away_team_id=2,
            away_team_name="Away Team",
            away_players=away_players.copy(),
            away_score=0,
            away_formation="4-4-2",
            country="Test Country",
        )

        mock_load_tracking.return_value = (tracking_data, metadata)
        mock_quality_check.return_value = False

        # Call get_game with tracking data only
        game = get_game(
            tracking_data_loc="test_tracking.dat",
            tracking_metadata_loc="test_metadata.xml",
            tracking_data_provider="tracab",
            check_quality=False,
            _check_game_class_=False,
            verbose=False,
        )

        # Verify starter information was added
        self.assertIn("starter", game.home_players.columns)
        self.assertIn("starter", game.away_players.columns)

        # Players 1 and 2 should be starters (they have data in frame 0)
        self.assertTrue(
            game.home_players.loc[game.home_players["shirt_num"] == 1, "starter"].iloc[
                0
            ]
        )
        self.assertTrue(
            game.home_players.loc[game.home_players["shirt_num"] == 2, "starter"].iloc[
                0
            ]
        )
        self.assertFalse(
            game.home_players.loc[game.home_players["shirt_num"] == 3, "starter"].iloc[
                0
            ]
        )

        self.assertTrue(
            game.away_players.loc[game.away_players["shirt_num"] == 1, "starter"].iloc[
                0
            ]
        )
        self.assertTrue(
            game.away_players.loc[game.away_players["shirt_num"] == 2, "starter"].iloc[
                0
            ]
        )
        self.assertFalse(
            game.away_players.loc[game.away_players["shirt_num"] == 3, "starter"].iloc[
                0
            ]
        )


if __name__ == "__main__":
    unittest.main()
