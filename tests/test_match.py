import os
import unittest
from dataclasses import fields

from databallpy import get_game
from databallpy.match import Match
from databallpy.utils.utils import _values_are_equal_


class TestMatch(unittest.TestCase):
    def setUp(self):
        base_path = os.path.join("tests", "test_data")

        td_tracab_loc = os.path.join(base_path, "tracab_td_test.dat")
        md_tracab_loc = os.path.join(base_path, "tracab_metadata_test.xml")
        ed_opta_loc = os.path.join(base_path, "f24_test.xml")
        md_opta_loc = os.path.join(base_path, "f7_test.xml")

        self.game = get_game(
            tracking_data_loc=td_tracab_loc,
            tracking_metadata_loc=md_tracab_loc,
            tracking_data_provider="tracab",
            event_data_loc=ed_opta_loc,
            event_metadata_loc=md_opta_loc,
            event_data_provider="opta",
        )

    def test_match(self):
        with self.assertWarns(DeprecationWarning):
            match = Match(
                tracking_data=self.game.tracking_data,
                tracking_data_provider="tracab",
                event_data=self.game.event_data,
                pitch_dimensions=self.game.pitch_dimensions,
                periods=self.game.periods,
                frame_rate=self.game.frame_rate,
                home_team_id=self.game.home_team_id,
                home_formation=self.game.home_formation,
                home_score=self.game.home_score,
                home_team_name=self.game.home_team_name,
                home_players=self.game.home_players,
                away_team_id=self.game.away_team_id,
                away_formation=self.game.away_formation,
                away_score=self.game.away_score,
                away_team_name=self.game.away_team_name,
                away_players=self.game.away_players,
                country=self.game.country,
                shot_events=self.game.shot_events,
                pass_events=self.game.pass_events,
                dribble_events=self.game.dribble_events,
                other_events=self.game.other_events,
                allow_synchronise_tracking_and_event_data=self.game.allow_synchronise_tracking_and_event_data,
                _tracking_timestamp_is_precise=self.game._tracking_timestamp_is_precise,
                _event_timestamp_is_precise=self.game._event_timestamp_is_precise,
                _periods_changed_playing_direction=self.game._periods_changed_playing_direction,
            )

        for current_field in fields(match):
            if not _values_are_equal_(
                getattr(match, current_field.name),
                getattr(self.game, current_field.name),
            ):
                import pdb

                pdb.set_trace()
