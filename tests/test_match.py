import os
import unittest

import pandas as pd

from databallpy.errors import DataBallPyError
from databallpy.get_match import get_match
from databallpy.match import Match


class TestMatch(unittest.TestCase):
    def setUp(self):
        td_tracab_loc = "tests/test_data/tracab_td_test.dat"
        md_tracab_loc = "tests/test_data/tracab_metadata_test.xml"
        ed_opta_loc = "tests/test_data/f24_test.xml"
        md_opta_loc = "tests/test_data/f7_test.xml"
        self.td_provider = "tracab"
        self.ed_provider = "opta"

        self.expected_match_tracab_opta = get_match(
            tracking_data_loc=td_tracab_loc,
            tracking_metadata_loc=md_tracab_loc,
            tracking_data_provider="tracab",
            event_data_loc=ed_opta_loc,
            event_metadata_loc=md_opta_loc,
            event_data_provider="opta",
            check_quality=False,
        )

        self.expected_match_tracab = get_match(
            tracking_data_loc=td_tracab_loc,
            tracking_metadata_loc=md_tracab_loc,
            tracking_data_provider="tracab",
            check_quality=False,
        )

        td_metrica_loc = "tests/test_data/metrica_tracking_data_test.txt"
        md_metrica_loc = "tests/test_data/metrica_metadata_test.xml"
        ed_metrica_loc = "tests/test_data/metrica_event_data_test.json"

        self.expected_match_metrica = get_match(
            tracking_data_loc=td_metrica_loc,
            tracking_metadata_loc=md_metrica_loc,
            tracking_data_provider="metrica",
            event_data_loc=ed_metrica_loc,
            event_metadata_loc=md_metrica_loc,
            event_data_provider="metrica",
            check_quality=False,
        )

        self.match_to_sync = get_match(
            tracking_data_loc="tests/test_data/sync/tracab_td_sync_test.dat",
            tracking_metadata_loc="tests/test_data/sync/tracab_metadata_sync_test.xml",
            tracking_data_provider="tracab",
            event_data_loc="tests/test_data/sync/opta_events_sync_test.xml",
            event_metadata_loc="tests/test_data/sync/opta_metadata_sync_test.xml",
            event_data_provider="opta",
            check_quality=False,
        )

        self.expected_match_opta = get_match(
            event_data_loc=ed_opta_loc,
            event_metadata_loc=md_opta_loc,
            event_data_provider="opta",
        )

    def test_match_eq(self):
        assert self.expected_match_metrica == self.expected_match_metrica
        assert self.expected_match_metrica != self.expected_match_tracab_opta

    def test_match_copy(self):
        copied = self.expected_match_tracab_opta.copy()
        assert self.expected_match_tracab_opta == copied

        copied.pitch_dimensions[0] = 22.0
        assert self.expected_match_tracab_opta != copied

        copied.pitch_dimensions[0] = self.expected_match_tracab_opta.pitch_dimensions[0]
        assert self.expected_match_tracab_opta == copied

        copied.tracking_data.iloc[0, 0] = "wrong input"
        assert self.expected_match_tracab_opta != copied

    def test_match_post_init(self):

        # tracking data
        with self.assertRaises(TypeError):
            Match(
                tracking_data="tracking_data",
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=pd.DataFrame(
                    {"frame": [1], "home_1_x": [12], "ball_z": [13]}
                ),
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        # tracking data provider
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=14.3,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        # event data
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data="event_data",
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=pd.DataFrame(
                    {
                        "event_id": [1],
                        "player": ["player_1"],
                        "event": ["pass"],
                    }
                ),
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        # event data provider
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=["opta"],
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        # pitch dimensions
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions={1: 22, 2: 11},
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=[10.0, 11.0, 12.0],
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=[10, 11.0],
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        # periods
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=[1, 2, 3, 4, 5],
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=pd.DataFrame({"times": [1, 2, 3, 4, 5]}),
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=pd.DataFrame({"period": [0, 1, 2, 3, 4]}),
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=pd.DataFrame({"period": [1, 1, 2, 3, 4, 5]}),
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            periods = self.expected_match_tracab_opta.periods.copy()
            periods["start_datetime_ed"] = periods["start_datetime_ed"].dt.tz_localize(
                None
            )
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        # frame rate
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=25.0,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=-25,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        # team id
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=123.0,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        # team name
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=["teamone"],
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        # team score
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=11.5,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=-3,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        # team formation
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=[1, 4, 2, 2],
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation="one-four-three-three",
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        # team players
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players="one-four-three-three",
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players.drop(
                    "shirt_num", axis=1
                ),
                country=self.expected_match_tracab_opta.country,
            )

        # pitch axis
        with self.assertRaises(DataBallPyError):
            td_changed = self.expected_match_tracab_opta.tracking_data.copy()
            td_changed["ball_x"] += 10.0
            Match(
                tracking_data=td_changed,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        with self.assertRaises(DataBallPyError):
            td_changed = self.expected_match_tracab_opta.tracking_data.copy()
            td_changed["ball_y"] += 10.0

            Match(
                tracking_data=td_changed,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        # playing direction
        with self.assertRaises(DataBallPyError):
            td_changed = self.expected_match_tracab_opta.tracking_data.copy()
            td_changed.loc[0, "home_34_x"] = 3.0

            Match(
                tracking_data=td_changed,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        with self.assertRaises(DataBallPyError):
            td_changed = self.expected_match_tracab_opta.tracking_data.copy()
            td_changed.loc[0, "away_17_x"] = -3.0
            Match(
                tracking_data=td_changed,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=self.expected_match_tracab_opta.country,
            )

        # country
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.expected_match_tracab_opta.pitch_dimensions,
                periods=self.expected_match_tracab_opta.periods,
                frame_rate=self.expected_match_tracab_opta.frame_rate,
                home_team_id=self.expected_match_tracab_opta.home_team_id,
                home_formation=self.expected_match_tracab_opta.home_formation,
                home_score=self.expected_match_tracab_opta.home_score,
                home_team_name=self.expected_match_tracab_opta.home_team_name,
                home_players=self.expected_match_tracab_opta.home_players,
                away_team_id=self.expected_match_tracab_opta.away_team_id,
                away_formation=self.expected_match_tracab_opta.away_formation,
                away_score=self.expected_match_tracab_opta.away_score,
                away_team_name=self.expected_match_tracab_opta.away_team_name,
                away_players=self.expected_match_tracab_opta.away_players,
                country=["Netherlands", "Germany"],
            )

    def test_preprosessing_status(self):
        match = self.match_to_sync.copy()
        assert match.is_synchronised is False
        assert (
            match.preprocessing_status
            == "Preprocessing status:\n\tis_synchronised = False"
        )
        match.synchronise_tracking_and_event_data(n_batches_per_half=1)
        assert match.is_synchronised is True
        assert (
            match.preprocessing_status
            == "Preprocessing status:\n\tis_synchronised = True"
        )

    def test__repr__(self):
        assert (
            self.expected_match_metrica.__repr__()
            == "databallpy.match.Match object: Team A 0 - 2 Team B 2019-02-21 03:30:07"
        )
        assert (
            self.expected_match_metrica.name
            == "Team A 0 - 2 Team B 2019-02-21 03:30:07"
        )

    def test_match__eq__(self):
        assert not self.expected_match_tracab_opta == pd.DataFrame()

    def test_match_name(self):
        assert (
            self.expected_match_tracab_opta.name
            == "TeamOne 3 - 1 TeamTwo 2023-01-14 16:46:39"
        )

    def test_match_home_players_column_ids(self):
        assert self.expected_match_tracab_opta.home_players_column_ids() == [
            "home_34_x",
            "home_34_y",
        ]

    def test_match_away_players_column_ids(self):
        assert self.expected_match_tracab_opta.away_players_column_ids() == [
            "away_17_x",
            "away_17_y",
        ]

    def test_match_player_column_id_to_full_name(self):
        res_name_home = self.expected_match_tracab_opta.player_column_id_to_full_name(
            "home_1"
        )
        assert res_name_home == "Piet Schrijvers"

        res_name_away = self.expected_match_tracab_opta.player_column_id_to_full_name(
            "away_2"
        )
        assert res_name_away == "TestSpeler"

    def test_match_player_id_to_column_id(self):
        res_column_id_home = self.expected_match_tracab_opta.player_id_to_column_id(
            19367
        )
        assert res_column_id_home == "home_1"

        res_column_id_away = self.expected_match_tracab_opta.player_id_to_column_id(
            450445
        )
        assert res_column_id_away == "away_2"

        with self.assertRaises(ValueError):
            self.expected_match_tracab_opta.player_id_to_column_id(4)

    def test_match_requires_event_data_wrapper(self):
        match = self.expected_match_opta.copy()
        with self.assertRaises(DataBallPyError):
            match.synchronise_tracking_and_event_data()

    def test_match_requires_tracking_data_wrapper(self):
        match = self.expected_match_tracab.copy()
        with self.assertRaises(DataBallPyError):
            match.synchronise_tracking_and_event_data()

    def test_save_match(self):
        assert not os.path.exists(
            "tests/test_data/TeamOne 3 - 1 TeamTwo 2023-01-22 16:46:39.pickle"
        )
        match = self.match_to_sync.copy()
        match.save_match(path="tests/test_data")
        assert os.path.exists(
            "tests/test_data/TeamOne 3 - 1 TeamTwo 2023-01-22 16:46:39.pickle"
        )
        os.remove("tests/test_data/TeamOne 3 - 1 TeamTwo 2023-01-22 16:46:39.pickle")
