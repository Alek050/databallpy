import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from databallpy.get_match import get_match
from databallpy.match import Match
from databallpy.utils.errors import DataBallPyError
from databallpy.utils.warnings import DataBallPyWarning
from tests.expected_outcomes import (
    DRIBBLE_EVENTS_OPTA_TRACAB,
    PASS_EVENTS_OPTA_TRACAB,
    SHOT_EVENTS_OPTA_TRACAB,
    TACKLE_EVENTS_OPTA_TRACAB,
)


class TestMatch(unittest.TestCase):
    def setUp(self):
        base_path = os.path.join("tests", "test_data")

        td_tracab_loc = os.path.join(base_path, "tracab_td_test.dat")
        md_tracab_loc = os.path.join(base_path, "tracab_metadata_test.xml")
        ed_opta_loc = os.path.join(base_path, "f24_test.xml")
        md_opta_loc = os.path.join(base_path, "f7_test.xml")
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

        td_metrica_loc = os.path.join(base_path, "metrica_tracking_data_test.txt")
        md_metrica_loc = os.path.join(base_path, "metrica_metadata_test.xml")
        ed_metrica_loc = os.path.join(base_path, "metrica_event_data_test.json")

        self.expected_match_metrica = get_match(
            tracking_data_loc=td_metrica_loc,
            tracking_metadata_loc=md_metrica_loc,
            tracking_data_provider="metrica",
            event_data_loc=ed_metrica_loc,
            event_metadata_loc=md_metrica_loc,
            event_data_provider="metrica",
            check_quality=False,
        )

        sync_base_path = os.path.join(base_path, "sync")
        self.match_to_sync = get_match(
            tracking_data_loc=os.path.join(sync_base_path, "tracab_td_sync_test.dat"),
            tracking_metadata_loc=os.path.join(
                sync_base_path, "tracab_metadata_sync_test.xml"
            ),
            tracking_data_provider="tracab",
            event_data_loc=os.path.join(sync_base_path, "opta_events_sync_test.xml"),
            event_metadata_loc=os.path.join(
                sync_base_path, "opta_metadata_sync_test.xml"
            ),
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

        copied.tracking_data.iloc[0, 0] = -100000.0
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
                tracking_data=self.expected_match_tracab_opta.tracking_data.drop(
                    columns=["datetime"]
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

        with self.assertRaises(TypeError):
            td = self.expected_match_tracab_opta.tracking_data.copy()
            td["datetime"] = td["datetime"].astype(str)
            Match(
                tracking_data=td,
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
            td = self.expected_match_tracab_opta.tracking_data.copy()
            td["datetime"] = td["datetime"].dt.tz_localize(None)
            Match(
                tracking_data=td,
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
                        "databallpy_event": ["pass"],
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

        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=pd.DataFrame(
                    {
                        "event_id": [1],
                        "databallpy_event": ["pass"],
                        "period_id": [1],
                        "team_id": [1],
                        "player_id": [1],
                        "start_x": [1],
                        "start_y": [1],
                        "player": ["player_1"],
                        "datetime": ["2020-01-01 00:00:00"],  # not datetime object
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

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.expected_match_tracab_opta.tracking_data,
                tracking_data_provider=self.td_provider,
                event_data=pd.DataFrame(
                    {
                        "event_id": [1],
                        "databallpy_event": ["pass"],
                        "period_id": [1],
                        "team_id": [1],
                        "player_id": [1],
                        "start_x": [1],
                        "start_y": [1],
                        "player": ["player_1"],
                        "datetime": pd.to_datetime(
                            ["2020-01-01 00:00:00"]
                        ),  # no timezone assigned
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
                event_data=self.expected_match_tracab_opta.event_data,
                event_data_provider=["opta"],
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
                periods=pd.DataFrame({"period_id": [0, 1, 2, 3, 4]}),
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
                periods=pd.DataFrame({"period_idw": [1, 1, 2, 3, 4, 5]}),
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
        with self.assertWarns(DataBallPyWarning):
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

        with self.assertWarns(DataBallPyWarning):
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
        # shot_events
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
                country=self.expected_match_tracab_opta.country,
                shot_events=["shot", "goal"],
            )
        # shot_events
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
                country=self.expected_match_tracab_opta.country,
                shot_events={"shot": "goal"},
            )

    def test_preprosessing_status(self):
        match = self.match_to_sync.copy()
        match.allow_synchronise_tracking_and_event_data = True
        assert match.is_synchronised is False
        assert (
            match.preprocessing_status
            == "Preprocessing status:\n\tis_synchronised = False"
        )
        match.synchronise_tracking_and_event_data(n_batches=2)
        assert match.is_synchronised is True
        assert (
            match.preprocessing_status
            == "Preprocessing status:\n\tis_synchronised = True"
        )

    def test_synchronise_tracking_and_event_data_not_allowed(self):
        match = self.match_to_sync.copy()
        match.allow_synchronise_tracking_and_event_data = False
        with self.assertRaises(DataBallPyError):
            match.synchronise_tracking_and_event_data(n_batches=2)

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

    def test_match_date(self):
        assert self.expected_match_tracab_opta.date == pd.Timestamp(
            "2023-01-14 16:46:39.720000+0100", tz="Europe/Amsterdam"
        )

    def test_match_name(self):
        assert (
            self.expected_match_tracab_opta.name
            == "TeamOne 3 - 1 TeamTwo 2023-01-14 16:46:39"
        )
        assert (
            self.expected_match_opta.name == "TeamOne 3 - 1 TeamTwo 2023-01-22 12:18:32"
        )

    def test_match_name_no_date(self):
        match = self.expected_match_tracab_opta.copy()
        match.periods = match.periods.drop(
            columns=["start_datetime_td", "start_datetime_ed"], errors="ignore"
        )
        assert match.name == "TeamOne 3 - 1 TeamTwo"

    def test_match_home_players_column_ids(self):
        with self.assertWarns(DeprecationWarning):
            assert self.expected_match_tracab_opta.home_players_column_ids() == [
                "home_34",
            ]

    def test_match_away_players_column_ids(self):
        with self.assertWarns(DeprecationWarning):
            assert self.expected_match_tracab_opta.away_players_column_ids() == [
                "away_17",
            ]

    def test_match_get_column_ids(self):
        match = self.expected_match_tracab_opta.copy()
        match.frame_rate = 1
        match.home_players = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "shirt_num": [11, 22, 33, 44],
                "start_frame": [1, 1, 100, 20],
                "end_frame": [200, 100, 200, 200],
                "position": ["goalkeeper", "defender", "midfielder", "forward"],
            }
        )
        match.away_players = pd.DataFrame(
            {
                "id": [5, 6, 7, 8],
                "shirt_num": [55, 66, 77, 88],
                "start_frame": [1, 1, 80, -999],
                "end_frame": [200, 80, 200, -999],
                "position": ["defender", "defender", "midfielder", "goalkeeper"],
            }
        )

        res_all = match.get_column_ids()
        expected_all = {
            "home_11",
            "home_22",
            "home_33",
            "home_44",
            "away_55",
            "away_66",
            "away_77",
        }
        self.assertSetEqual(set(res_all), expected_all)

        home = match.get_column_ids(team="home", min_minutes_played=2)
        self.assertSetEqual(set(home), {"home_11", "home_44"})

        away = match.get_column_ids(team="away", positions=["defender"])
        self.assertSetEqual(set(away), {"away_55", "away_66"})

        with self.assertRaises(ValueError):
            match.get_column_ids(team="wrong")
        with self.assertRaises(ValueError):
            match.get_column_ids(positions=["striker"])
        with self.assertRaises(TypeError):
            match.get_column_ids(min_minutes_played="fifteen")

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

    def test_match_shots_df_without_td_features(self):
        shot_attribute_names = [
            "event_id",
            "period_id",
            "minutes",
            "seconds",
            "datetime",
            "start_x",
            "start_y",
            "team_id",
            "team_side",
            "player_id",
            "jersey",
            "outcome",
            "related_event_id",
            "xT",
            "body_part",
            "possession_type",
            "set_piece",
            "outcome_str",
            "y_target",
            "z_target",
            "first_touch",
            "ball_goal_distance",
            "ball_gk_distance",
            "shot_angle",
            "gk_optimal_loc_distance",
            "pressure_on_ball",
            "n_obstructive_players",
            "n_obstructive_defenders",
            "goal_gk_distance",
            "xG",
        ]
        expected_df = pd.DataFrame(
            {
                attr: [getattr(shot, attr) for shot in SHOT_EVENTS_OPTA_TRACAB.values()]
                for attr in shot_attribute_names
            }
        )

        match = self.expected_match_tracab_opta.copy()
        # make sure it will not try to add tracking data features
        match.allow_synchronise_tracking_and_event_data = False
        shots_df = self.expected_match_tracab_opta.shots_df
        pd.testing.assert_frame_equal(shots_df, expected_df)

    @patch("databallpy.events.shot_event." "ShotEvent.add_tracking_data_features")
    def test_match_shots_df_tracking_data_features(
        self, mock_add_tracking_data_features
    ):
        mock_add_tracking_data_features.return_value = "Return value"

        # for away team shot
        match = self.expected_match_tracab_opta.copy()
        match.tracking_data["event_id"] = [np.nan, np.nan, np.nan, np.nan, 2512690516]
        match.tracking_data["databallpy_event"] = [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            "shot",
        ]
        match._is_synchronised = True
        match.shot_events = {2512690516: match.shot_events[2512690516]}
        match.add_tracking_data_features_to_shots()

        expected_td_frame = match.tracking_data.iloc[-1]
        expected_column_id = "away_1"
        expected_gk_column_id = "home_1"

        called_args, _ = mock_add_tracking_data_features.call_args_list[-1]
        called_td_frame = called_args[0]
        called_column_id = called_args[1]
        called_gk_column_id = called_args[2]

        pd.testing.assert_series_equal(called_td_frame, expected_td_frame)
        assert called_column_id == expected_column_id
        assert called_gk_column_id == expected_gk_column_id

        # for home team shot
        match = self.expected_match_tracab_opta.copy()
        match.tracking_data["event_id"] = [np.nan, np.nan, 2512690515, np.nan, np.nan]
        match.tracking_data["databallpy_event"] = [
            np.nan,
            np.nan,
            "own_goal",
            np.nan,
            np.nan,
        ]
        match._is_synchronised = True
        match.shot_events = {2512690515: match.shot_events[2512690515]}
        match.add_tracking_data_features_to_shots()

        expected_td_frame = match.tracking_data.iloc[-3]
        expected_column_id = "home_2"
        expected_gk_column_id = "away_2"

        called_args, _ = mock_add_tracking_data_features.call_args_list[-1]
        called_td_frame = called_args[0]
        called_column_id = called_args[1]
        called_gk_column_id = called_args[2]

        pd.testing.assert_series_equal(called_td_frame, expected_td_frame)
        assert called_column_id == expected_column_id
        assert called_gk_column_id == expected_gk_column_id

        with self.assertRaises(DataBallPyError):
            match._is_synchronised = False
            match.add_tracking_data_features_to_shots()

    @patch("databallpy.events.pass_event." "PassEvent.add_tracking_data_features")
    def test_match_add_tracking_data_features_to_passes(
        self, mock_add_tracking_data_features
    ):
        mock_add_tracking_data_features.return_value = "Return value"

        match = self.expected_match_tracab_opta.copy()
        match._is_synchronised = True

        # no player_possession column
        with self.assertRaises(DataBallPyError):
            match.add_tracking_data_features_to_passes()

        match.tracking_data["player_possession"] = None
        match._is_synchronised = False
        # tracking data not synchronised
        with self.assertRaises(DataBallPyError):
            match.add_tracking_data_features_to_passes()

        match._is_synchronised = True

        match.tracking_data["event_id"] = [np.nan, np.nan, 2499594225, np.nan, np.nan]
        match.tracking_data["databallpy_event"] = [None, None, "pass", None, None]
        match.tracking_data["home_1_x"] = [-50.0, -50.0, -50.0, -50.0, -50.0]
        match.tracking_data["home_1_y"] = [0.0, 0.0, 0.0, 0.0, 0.0]

        # case 1
        match1 = match.copy()
        match1.pass_events = {2499594225: match.pass_events[2499594225]}

        match1.add_tracking_data_features_to_passes()

        called_args, _ = mock_add_tracking_data_features.call_args_list[-1]
        called_td_frame = called_args[0]
        called_passer_column_id = called_args[1]
        called_receiver_column_id = called_args[2]
        called_end_loc_td = called_args[3]
        called_pitch_dimensions = called_args[4]
        called_opponent_column_ids = called_args[5]

        pd.testing.assert_series_equal(called_td_frame, match1.tracking_data.iloc[2])
        assert called_passer_column_id == "home_1"
        assert called_receiver_column_id == "home_34"
        np.testing.assert_array_equal(called_end_loc_td, np.array([2.76, -0.70]))
        assert called_pitch_dimensions == match1.pitch_dimensions
        assert called_opponent_column_ids == ["away_17"]

        # # case 2
        match2 = match.copy()
        match2.pass_events = {2499594243: match.pass_events[2499594243]}
        match2.pass_events[2499594243].end_x = 3.0
        match2.pass_events[2499594243].end_y = 0.0
        match2.tracking_data["player_possession"] = [
            np.nan,
            np.nan,
            "away_1",
            np.nan,
            "away_2",
        ]
        match2.tracking_data["event_id"] = [np.nan, np.nan, 2499594243, np.nan, np.nan]
        match2.tracking_data = match2.tracking_data.rename(
            columns={"home_1_x": "away_1_x", "home_1_y": "away_1_y"}
        )
        match2.tracking_data["away_1_x"] = match2.tracking_data["away_1_x"] * -1

        match2.add_tracking_data_features_to_passes()
        called_args, _ = mock_add_tracking_data_features.call_args_list[-1]
        called_td_frame = called_args[0]
        called_passer_column_id = called_args[1]
        called_receiver_column_id = called_args[2]
        called_end_loc_td = called_args[3]
        called_pitch_dimensions = called_args[4]
        called_opponent_column_ids = called_args[5]

        pd.testing.assert_series_equal(called_td_frame, match2.tracking_data.iloc[2])
        assert called_passer_column_id == "away_1"
        assert called_receiver_column_id == "away_2"
        np.testing.assert_array_equal(called_end_loc_td, np.array([2.76, -0.70]))
        assert called_pitch_dimensions == match2.pitch_dimensions
        assert called_opponent_column_ids == ["home_34"]

        # # case 3 end loc diff too big
        match3 = match.copy()
        match3.pass_events = {2499594243: match.pass_events[2499594243]}
        match3.pass_events[2499594243].end_x = 30.0
        match3.pass_events[2499594243].end_y = 50.0
        match3.tracking_data["player_possession"] = [
            np.nan,
            np.nan,
            "away_1",
            np.nan,
            "away_2",
        ]
        match3.tracking_data["event_id"] = [np.nan, np.nan, 2499594243, np.nan, np.nan]

        mock_add_tracking_data_features.reset_mock()
        match3.add_tracking_data_features_to_passes()
        mock_add_tracking_data_features.assert_not_called()

        # case 4
        match4 = match.copy()
        match4.tracking_data["player_possession"] = [
            np.nan,
            np.nan,
            "away_1",
            np.nan,
            "away_1",
        ]
        match4.pass_events = {2499594243: match.pass_events[2499594243]}
        match4.pass_events[2499594243].end_x = 3.0
        match4.pass_events[2499594243].end_y = 0.0
        match4.tracking_data["event_id"] = [np.nan, np.nan, 2499594243, np.nan, np.nan]
        match4.tracking_data = match4.tracking_data.rename(
            columns={"home_1_x": "away_1_x", "home_1_y": "away_1_y"}
        )
        match4.tracking_data["away_1_x"] = match4.tracking_data["away_1_x"] * -1

        match4.add_tracking_data_features_to_passes()
        called_args, _ = mock_add_tracking_data_features.call_args_list[-1]
        called_td_frame = called_args[0]
        called_passer_column_id = called_args[1]
        called_receiver_column_id = called_args[2]
        called_end_loc_td = called_args[3]
        called_pitch_dimensions = called_args[4]
        called_opponent_column_ids = called_args[5]

        pd.testing.assert_series_equal(called_td_frame, match4.tracking_data.iloc[2])
        assert called_passer_column_id == "away_1"
        assert called_receiver_column_id == "away_17"
        np.testing.assert_array_equal(called_end_loc_td, np.array([2.76, -0.70]))
        assert called_pitch_dimensions == match4.pitch_dimensions
        assert called_opponent_column_ids == ["home_34"]

        # case 5
        match5 = match.copy()
        match5.tracking_data["player_possession"] = [
            np.nan,
            np.nan,
            "away_1",
            np.nan,
            "away_1",
        ]
        match5.pass_events = {2499594243: match.pass_events[2499594243]}
        match5.pass_events[2499594243].end_x = 3.0
        match5.pass_events[2499594243].end_y = 0.0
        match5.tracking_data["event_id"] = [np.nan, np.nan, 2499594243, np.nan, np.nan]
        match5.tracking_data["ball_x"] = np.nan
        match5.tracking_data = match5.tracking_data.rename(
            columns={"home_1_x": "away_1_x", "home_1_y": "away_1_y"}
        )
        match5.tracking_data["away_1_x"] = match5.tracking_data["away_1_x"] * -1

        mock_add_tracking_data_features.reset_mock()
        match5.add_tracking_data_features_to_passes()
        mock_add_tracking_data_features.assert_not_called()

        # case 6, end loc to close too start loc of pass
        match6 = match.copy()
        match6.tracking_data["player_possession"] = [
            np.nan,
            np.nan,
            "away_1",
            np.nan,
            "away_1",
        ]
        match6.pass_events = {2499594243: match.pass_events[2499594243]}
        match6.pass_events[2499594243].end_x = 3.0
        match6.pass_events[2499594243].end_y = 0.0
        match6.tracking_data["event_id"] = [np.nan, np.nan, 2499594243, np.nan, np.nan]
        match6.tracking_data = match6.tracking_data.rename(
            columns={"home_1_x": "away_1_x", "home_1_y": "away_1_y"}
        )
        match6.tracking_data["away_1_x"] = [0, 0, 2.76, 0, 0]  # too close to the ball

        mock_add_tracking_data_features.reset_mock()
        match6.add_tracking_data_features_to_passes()
        mock_add_tracking_data_features.assert_not_called()

    def test_match_dribbles_df_without_td_features(self):
        dribble_attribute_names = [
            "event_id",
            "period_id",
            "minutes",
            "seconds",
            "datetime",
            "start_x",
            "start_y",
            "team_id",
            "team_side",
            "player_id",
            "jersey",
            "outcome",
            "related_event_id",
            "xT",
            "body_part",
            "possession_type",
            "set_piece",
            "duel_type",
            "with_opponent",
        ]
        expected_df = pd.DataFrame(
            {
                attr: [
                    getattr(dribble, attr)
                    for dribble in DRIBBLE_EVENTS_OPTA_TRACAB.values()
                ]
                for attr in dribble_attribute_names
            }
        )
        dribbles_df = self.expected_match_tracab_opta.dribbles_df
        pd.testing.assert_frame_equal(dribbles_df, expected_df)

    def test_match_passes_df_without_td_features(self):
        pass_attribute_names = [
            "event_id",
            "period_id",
            "minutes",
            "seconds",
            "datetime",
            "start_x",
            "start_y",
            "team_id",
            "team_side",
            "player_id",
            "jersey",
            "outcome",
            "related_event_id",
            "xT",
            "body_part",
            "possession_type",
            "set_piece",
            "outcome_str",
            "end_x",
            "end_y",
            "pass_type",
            "receiver_player_id",
            "pass_length",
            "forward_distance",
            "passer_goal_distance",
            "pass_end_loc_goal_distance",
            "opponents_in_passing_lane",
            "pressure_on_passer",
            "pressure_on_receiver",
            "pass_goal_angle",
        ]
        expected_df = pd.DataFrame(
            {
                attr: [
                    getattr(pass_, attr) for pass_ in PASS_EVENTS_OPTA_TRACAB.values()
                ]
                for attr in pass_attribute_names
            }
        )
        passes_df = self.expected_match_tracab_opta.passes_df
        pd.testing.assert_frame_equal(passes_df, expected_df)

    def test_match_other_events_df(self):
        other_event_attribute_names = [
            "event_id",
            "period_id",
            "minutes",
            "seconds",
            "datetime",
            "start_x",
            "start_y",
            "team_id",
            "team_side",
            "player_id",
            "jersey",
            "outcome",
            "related_event_id",
        ]
        expected_df = pd.DataFrame(
            {
                attr: [
                    getattr(other_event, attr)
                    for other_event in TACKLE_EVENTS_OPTA_TRACAB.values()
                ]
                for attr in other_event_attribute_names
            }
        )
        expected_df["name"] = "TackleEvent"
        other_events_df = self.expected_match_tracab_opta.other_events_df
        pd.testing.assert_frame_equal(other_events_df, expected_df)

    def test_match_get_event(self):
        match = self.expected_match_tracab_opta.copy()
        event = match.get_event(2512690515)
        assert event == match.shot_events[2512690515]

        event = match.get_event(2499594225)
        assert event == match.pass_events[2499594225]

        event = match.get_event(2499594285)
        assert event == match.dribble_events[2499594285]

        with self.assertRaises(ValueError):
            match.get_event(2499594286)

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
            os.path.join(
                "tests", "test_data", "TeamOne 3 - 1 TeamTwo 2023-01-22 16_46_39.pickle"
            )
        )
        match = self.match_to_sync.copy()
        match.allow_synchronise_tracking_and_event_data = True
        match.save_match(path=os.path.join("tests", "test_data"))
        assert os.path.exists(
            os.path.join(
                "tests", "test_data", "TeamOne 3 - 1 TeamTwo 2023-01-22 16_46_39.pickle"
            )
        )
        os.remove(
            os.path.join(
                "tests", "test_data", "TeamOne 3 - 1 TeamTwo 2023-01-22 16_46_39.pickle"
            )
        )

    def test_all_events(self):
        match = self.expected_match_opta.copy()
        expected_events = {
            **match.shot_events,
            **match.pass_events,
            **match.dribble_events,
            **match.other_events,
        }
        assert match.all_events == expected_events

    def test_get_frames(self):
        match = self.expected_match_tracab_opta.copy()
        res = match.get_frames(1509993)
        pd.testing.assert_frame_equal(
            match.get_frames(1509993), match.get_frames([1509993])
        )
        pd.testing.assert_frame_equal(res, match.tracking_data.iloc[0:1])

        res2 = match.get_frames(1509993, playing_direction="possession_oriented")
        pd.testing.assert_frame_equal(
            match.get_frames(1509993, playing_direction="possession_oriented"),
            match.get_frames([1509993], playing_direction="possession_oriented"),
        )
        cols = ["ball_x", "ball_y", "home_34_x", "home_34_y", "away_17_x", "away_17_y"]
        match.tracking_data[cols] = match.tracking_data[cols] * -1
        pd.testing.assert_frame_equal(res2, match.tracking_data.iloc[0:1])

        with self.assertRaises(ValueError):
            match.get_frames(1509993, playing_direction="wrong")

        with self.assertRaises(ValueError):
            match.get_frames(999)

        match.tracking_data.drop(columns=["ball_possession"], inplace=True)
        with self.assertRaises(ValueError):
            match.get_frames(1509993, playing_direction="possession_oriented")

    def test_get_event_frame(self):
        match = self.expected_match_tracab_opta.copy()
        pass_event = match.get_event(list(match.pass_events.keys())[0])
        match.tracking_data.loc[0, "event_id"] = pass_event.event_id

        with self.assertRaises(DataBallPyError):
            match.get_event_frame(pass_event.event_id)
        match._is_synchronised = True

        with self.assertRaises(ValueError):
            match.get_event_frame(999)

        res_team = match.get_event_frame(
            pass_event.event_id, playing_direction="team_oriented"
        )
        pd.testing.assert_frame_equal(res_team, match.tracking_data.iloc[0:1])

        res_possession = match.get_event_frame(
            pass_event.event_id, playing_direction="possession_oriented"
        )
        pd.testing.assert_frame_equal(
            res_possession, match.tracking_data.iloc[0:1]
        )  # event team side is home

        pass_event.team_side = "away"
        res_possession = match.get_event_frame(
            pass_event.event_id, playing_direction="possession_oriented"
        )
        cols = ["ball_x", "ball_y", "home_34_x", "home_34_y", "away_17_x", "away_17_y"]
        match.tracking_data[cols] = match.tracking_data[cols] * -1
        pd.testing.assert_frame_equal(res_possession, match.tracking_data.iloc[0:1])

        with self.assertRaises(ValueError):
            match.get_event_frame(pass_event.event_id, playing_direction="wrong")
