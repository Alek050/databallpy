import os
import unittest

import pandas as pd

from databallpy.game import Game
from databallpy.schemas.event_data import EventData
from databallpy.schemas.tracking_data import TrackingData
from databallpy.utils.errors import DataBallPyError
from databallpy.utils.get_game import get_game
from databallpy.utils.warnings import DataBallPyWarning
from databallpy.schemas import TrackingDataSchema
from tests.expected_outcomes import (
    DRIBBLE_EVENTS_OPTA_TRACAB,
    PASS_EVENTS_OPTA_TRACAB,
    SHOT_EVENTS_OPTA_TRACAB,
    TACKLE_EVENTS_OPTA_TRACAB,
)

from pandera.errors import SchemaError


class TestGame(unittest.TestCase):
    def setUp(self):
        base_path = os.path.join("tests", "test_data")

        td_tracab_loc = os.path.join(base_path, "tracab_td_test.dat")
        md_tracab_loc = os.path.join(base_path, "tracab_metadata_test.xml")
        ed_opta_loc = os.path.join(base_path, "f24_test.xml")
        md_opta_loc = os.path.join(base_path, "f7_test.xml")
        self.td_provider = "tracab"

        self.expected_game_tracab_opta = get_game(
            tracking_data_loc=td_tracab_loc,
            tracking_metadata_loc=md_tracab_loc,
            tracking_data_provider="tracab",
            event_data_loc=ed_opta_loc,
            event_metadata_loc=md_opta_loc,
            event_data_provider="opta",
            check_quality=False,
        )

        self.expected_game_tracab = get_game(
            tracking_data_loc=td_tracab_loc,
            tracking_metadata_loc=md_tracab_loc,
            tracking_data_provider="tracab",
            check_quality=False,
        )

        td_metrica_loc = os.path.join(base_path, "metrica_tracking_data_test.txt")
        md_metrica_loc = os.path.join(base_path, "metrica_metadata_test.xml")
        ed_metrica_loc = os.path.join(base_path, "metrica_event_data_test.json")

        self.expected_game_metrica = get_game(
            tracking_data_loc=td_metrica_loc,
            tracking_metadata_loc=md_metrica_loc,
            tracking_data_provider="metrica",
            event_data_loc=ed_metrica_loc,
            event_metadata_loc=md_metrica_loc,
            event_data_provider="metrica",
            check_quality=False,
        )

        sync_base_path = os.path.join(base_path, "sync")
        self.game_to_sync = get_game(
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

        self.expected_game_opta = get_game(
            event_data_loc=ed_opta_loc,
            event_metadata_loc=md_opta_loc,
            event_data_provider="opta",
        )

    def test_game_eq(self):
        assert self.expected_game_metrica == self.expected_game_metrica
        assert self.expected_game_metrica != self.expected_game_tracab_opta

    def test_game_copy(self):
        copied = self.expected_game_tracab_opta.copy()
        assert self.expected_game_tracab_opta == copied

        copied.pitch_dimensions[0] = 22.0
        assert self.expected_game_tracab_opta != copied

        copied.pitch_dimensions[0] = self.expected_game_tracab_opta.pitch_dimensions[0]
        assert self.expected_game_tracab_opta == copied

        copied.tracking_data.iloc[0, 0] = -100000.0
        assert self.expected_game_tracab_opta != copied

    def test_game_post_init(self):
        # tracking data
        with self.assertRaises(TypeError):
            Game(
                tracking_data="tracking_data",
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        with self.assertRaises(TypeError):
            tracking_data = self.expected_game_tracab_opta.tracking_data.copy()
            Game(
                tracking_data=TrackingData(
                    tracking_data, provider=6, frame_rate=tracking_data.frame_rate
                ),
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        with self.assertRaises(TypeError):
            tracking_data = self.expected_game_tracab_opta.tracking_data.copy()
            Game(
                tracking_data=TrackingData(
                    tracking_data, provider=tracking_data.provider, frame_rate=6.4
                ),
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            tracking_data = self.expected_game_tracab_opta.tracking_data.copy()
            Game(
                tracking_data=TrackingData(
                    tracking_data, provider=tracking_data.provider, frame_rate=-5
                ),
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        # event data
        with self.assertRaises(TypeError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data="event_data",
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=EventData(
                    {
                        "event_id": [1],
                        "player": ["player_1"],
                        "databallpy_event": ["pass"],
                    },
                    provider="opta",
                ),
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        with self.assertRaises(TypeError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=EventData(
                    {
                        "event_id": [1],
                        "databallpy_event": ["pass"],
                        "period_id": [1],
                        "team_id": [1],
                        "player_id": [1],
                        "start_x": [1],
                        "start_y": [1],
                        "player_name": ["player_1"],
                        "datetime": ["2020-01-01 00:00:00"],  # not datetime object
                    },
                    provider="opta",
                ),
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=EventData(
                    {
                        "event_id": [1],
                        "databallpy_event": ["pass"],
                        "period_id": [1],
                        "team_id": [1],
                        "player_id": [1],
                        "start_x": [1],
                        "start_y": [1],
                        "player_name": ["player_1"],
                        "datetime": pd.to_datetime(
                            ["2020-01-01 00:00:00"]
                        ),  # no timezone assigned
                    },
                    provider="opta",
                ),
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        # event data provider
        with self.assertRaises(TypeError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=EventData(
                    {
                        "event_id": [1],
                        "databallpy_event": ["pass"],
                        "period_id": [1],
                        "team_id": [1],
                        "player_id": [1],
                        "start_x": [1],
                        "start_y": [1],
                        "player_name": ["player_1"],
                        "datetime": pd.to_datetime(
                            ["2020-01-01 00:00:00"],
                        ).tz_localize("UTC"),
                    },
                    provider=["opta"],
                ),
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        # pitch dimensions
        with self.assertRaises(TypeError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions={1: 22, 2: 11},
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=[10.0, 11.0, 12.0],
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        with self.assertRaises(TypeError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=[10, 11.0],
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=[10.0, 68.0],
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=[105.0, 101.0],
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        # periods
        with self.assertRaises(TypeError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=[1, 2, 3, 4, 5],
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=pd.DataFrame({"times": [1, 2, 3, 4, 5]}),
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=pd.DataFrame({"period_id": [0, 1, 2, 3, 4]}),
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=pd.DataFrame({"period_idw": [1, 1, 2, 3, 4, 5]}),
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            periods = self.expected_game_tracab_opta.periods.copy()
            periods["start_datetime_ed"] = periods["start_datetime_ed"].dt.tz_localize(
                None
            )
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        # team id
        with self.assertRaises(TypeError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=123.0,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        # team name
        with self.assertRaises(TypeError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=["teamone"],
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        # team score
        with self.assertRaises(TypeError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=11.5,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=-3,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        # team formation
        with self.assertRaises(TypeError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=[1, 4, 2, 2],
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation="one-four-three-three",
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        # team players
        with self.assertRaises(TypeError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players="one-four-three-three",
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        with self.assertRaises(ValueError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players.drop(
                    "shirt_num", axis=1
                ),
                country=self.expected_game_tracab_opta.country,
            )

        wrong_pos = self.expected_game_tracab_opta.away_players.copy()
        wrong_pos["position"] = "unknown_position"
        with self.assertRaises(ValueError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=wrong_pos,
                country=self.expected_game_tracab_opta.country,
            )

        # playing direction
        with self.assertRaises(DataBallPyError):
            td_changed = self.expected_game_tracab_opta.tracking_data.copy()
            td_changed.loc[0, "home_34_x"] = 3.0
            Game(
                tracking_data=td_changed,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        with self.assertRaises(DataBallPyError):
            td_changed = self.expected_game_tracab_opta.tracking_data.copy()
            td_changed.loc[0, "away_17_x"] = -3.0
            Game(
                tracking_data=td_changed,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
            )

        # country
        with self.assertRaises(TypeError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=["Netherlands", "Germany"],
            )
        # shot_events
        with self.assertRaises(TypeError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
                shot_events=["shot", "goal"],
            )
        # shot_events
        with self.assertRaises(TypeError):
            Game(
                tracking_data=self.expected_game_tracab_opta.tracking_data,
                event_data=self.expected_game_tracab_opta.event_data,
                pitch_dimensions=self.expected_game_tracab_opta.pitch_dimensions,
                periods=self.expected_game_tracab_opta.periods,
                home_team_id=self.expected_game_tracab_opta.home_team_id,
                home_formation=self.expected_game_tracab_opta.home_formation,
                home_score=self.expected_game_tracab_opta.home_score,
                home_team_name=self.expected_game_tracab_opta.home_team_name,
                home_players=self.expected_game_tracab_opta.home_players,
                away_team_id=self.expected_game_tracab_opta.away_team_id,
                away_formation=self.expected_game_tracab_opta.away_formation,
                away_score=self.expected_game_tracab_opta.away_score,
                away_team_name=self.expected_game_tracab_opta.away_team_name,
                away_players=self.expected_game_tracab_opta.away_players,
                country=self.expected_game_tracab_opta.country,
                shot_events={"shot": "goal"},
            )

    def test_preprosessing_status(self):
        game = self.game_to_sync.copy()
        game.allow_synchronise_tracking_and_event_data = True
        assert game.is_synchronised is False
        assert (
            game.preprocessing_status
            == "Preprocessing status:\n\tis_synchronised = False"
        )
        game.synchronise_tracking_and_event_data(n_batches=2)
        assert game.is_synchronised is True
        assert (
            game.preprocessing_status
            == "Preprocessing status:\n\tis_synchronised = True"
        )

    def test_precise_timestamp(self):
        assert self.expected_game_tracab_opta.tracking_timestamp_is_precise is True
        assert self.expected_game_tracab_opta.event_timestamp_is_precise is True
        with self.assertRaises(AttributeError):
            self.expected_game_tracab_opta.tracking_timestamp_is_precise = False

    def test_synchronise_tracking_and_event_data_not_allowed(self):
        game = self.game_to_sync.copy()
        game.allow_synchronise_tracking_and_event_data = False
        with self.assertRaises(DataBallPyError):
            game.synchronise_tracking_and_event_data(n_batches=2)

    def test__repr__(self):
        assert (
            self.expected_game_metrica.__repr__()
            == "databallpy.game.Game object: Team A 0 - 2 Team B 2019-02-21 03:30:07"
        )
        assert (
            self.expected_game_metrica.name == "Team A 0 - 2 Team B 2019-02-21 03:30:07"
        )

    def test_game__eq__(self):
        assert not self.expected_game_tracab_opta == pd.DataFrame()

    def test_game_date(self):
        assert self.expected_game_tracab_opta.date == pd.Timestamp(
            "2023-01-14 16:46:39.720000+0100", tz="Europe/Amsterdam"
        )

    def test_game_name(self):
        assert (
            self.expected_game_tracab_opta.name
            == "TeamOne 3 - 1 TeamTwo 2023-01-14 16:46:39"
        )
        assert (
            self.expected_game_opta.name == "TeamOne 3 - 1 TeamTwo 2023-01-22 12:18:32"
        )

    def test_game_name_no_date(self):
        game = self.expected_game_tracab_opta.copy()
        game.periods = game.periods.drop(
            columns=["start_datetime_td", "start_datetime_ed"], errors="ignore"
        )
        assert game.name == "TeamOne 3 - 1 TeamTwo"

    def test_game_home_players_column_ids(self):
        with self.assertWarns(DeprecationWarning):
            assert self.expected_game_tracab_opta.home_players_column_ids() == [
                "home_34",
            ]

    def test_game_away_players_column_ids(self):
        with self.assertWarns(DeprecationWarning):
            assert self.expected_game_tracab_opta.away_players_column_ids() == [
                "away_17",
            ]

    def test_game_tracking_data_provider_depricated(self):
        game = self.expected_game_tracab_opta.copy()
        with self.assertWarns(DeprecationWarning):
            assert game.tracking_data_provider == game.tracking_data.provider

    def test_game_frame_rate_depricated(self):
        game = self.expected_game_tracab_opta.copy()
        with self.assertWarns(DeprecationWarning):
            assert game.frame_rate == game.tracking_data.frame_rate

    def test_game_event_data_provider_depricated(self):
        game = self.expected_game_tracab_opta.copy()
        with self.assertWarns(DeprecationWarning):
            assert game.event_data_provider == game.event_data.provider

    def test_game_get_column_ids(self):
        game = self.expected_game_tracab_opta.copy()
        game.tracking_data = TrackingData(
            game.tracking_data, provider=game.tracking_data.provider, frame_rate=1
        )
        game.home_players = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "shirt_num": [11, 22, 33, 44],
                "start_frame": [1, 1, 100, 20],
                "end_frame": [200, 100, 200, 200],
                "position": ["goalkeeper", "defender", "midfielder", "forward"],
            }
        )
        game.away_players = pd.DataFrame(
            {
                "id": [5, 6, 7, 8],
                "shirt_num": [55, 66, 77, 88],
                "start_frame": [1, 1, 80, -999],
                "end_frame": [200, 80, 200, -999],
                "position": ["defender", "defender", "midfielder", "goalkeeper"],
            }
        )

        res_all = game.get_column_ids()
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

        home = game.get_column_ids(team="home", min_minutes_played=2)
        self.assertSetEqual(set(home), {"home_11", "home_44"})

        away = game.get_column_ids(team="away", positions=["defender"])
        self.assertSetEqual(set(away), {"away_55", "away_66"})

        with self.assertRaises(ValueError):
            game.get_column_ids(team="wrong")
        with self.assertRaises(ValueError):
            game.get_column_ids(positions=["striker"])
        with self.assertRaises(TypeError):
            game.get_column_ids(min_minutes_played="fifteen")

        with self.assertWarns(DataBallPyWarning):
            game.away_players.drop(columns=["position"], inplace=True)
            away = game.get_column_ids(team="away", positions=["defender"])
            self.assertSetEqual(set(away), {"away_55", "away_66", "away_77"})

    def test_game_player_column_id_to_full_name(self):
        res_name_home = self.expected_game_tracab_opta.player_column_id_to_full_name(
            "home_1"
        )
        assert res_name_home == "Piet Schrijvers"

        res_name_away = self.expected_game_tracab_opta.player_column_id_to_full_name(
            "away_2"
        )
        assert res_name_away == "TestSpeler"

    def test_game_player_id_to_column_id(self):
        res_column_id_home = self.expected_game_tracab_opta.player_id_to_column_id(19367)
        assert res_column_id_home == "home_1"

        res_column_id_away = self.expected_game_tracab_opta.player_id_to_column_id(
            450445
        )
        assert res_column_id_away == "away_2"

        with self.assertRaises(ValueError):
            self.expected_game_tracab_opta.player_id_to_column_id(4)

    def test_game_shots_df_without_td_features(self):
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
            "xt",
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
            "xg",
        ]
        expected_df = pd.DataFrame(
            {
                attr: [getattr(shot, attr) for shot in SHOT_EVENTS_OPTA_TRACAB.values()]
                for attr in shot_attribute_names
            }
        )

        game = self.expected_game_tracab_opta.copy()
        # make sure it will not try to add tracking data features
        game.allow_synchronise_tracking_and_event_data = False
        shots_df = self.expected_game_tracab_opta.shots_df
        pd.testing.assert_frame_equal(shots_df, expected_df)

    def test_game_dribbles_df_without_td_features(self):
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
            "xt",
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
        dribbles_df = self.expected_game_tracab_opta.dribbles_df
        pd.testing.assert_frame_equal(dribbles_df, expected_df)

    def test_game_passes_df_without_td_features(self):
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
            "xt",
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
        passes_df = self.expected_game_tracab_opta.passes_df
        pd.testing.assert_frame_equal(passes_df, expected_df)

    def test_game_other_events_df(self):
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
        other_events_df = self.expected_game_tracab_opta.other_events_df
        pd.testing.assert_frame_equal(other_events_df, expected_df)

    def test_game_get_event(self):
        game = self.expected_game_tracab_opta.copy()
        event = game.get_event(9)
        assert event == game.shot_events[9]

        event = game.get_event(4)
        assert event == game.pass_events[4]

        event = game.get_event(7)
        assert event == game.dribble_events[7]

        with self.assertRaises(ValueError):
            game.get_event(1)

    def test_game_requires_event_data_wrapper(self):
        game = self.expected_game_opta.copy()
        with self.assertRaises(DataBallPyError):
            game.synchronise_tracking_and_event_data()

    def test_game_requires_tracking_data_wrapper(self):
        game = self.expected_game_tracab.copy()
        with self.assertRaises(DataBallPyError):
            game.synchronise_tracking_and_event_data()

    def test_save_game(self):
        assert not os.path.exists(
            os.path.join(
                "tests", "test_data", "TeamOne 3 - 1 TeamTwo 2023-01-22 16_46_39.pickle"
            )
        )
        game = self.game_to_sync.copy()
        game.allow_synchronise_tracking_and_event_data = True
        game.save_game(path=os.path.join("tests", "test_data"))
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
        game = self.expected_game_opta.copy()
        expected_events = {
            **game.shot_events,
            **game.pass_events,
            **game.dribble_events,
            **game.other_events,
        }
        assert game.all_events == expected_events

    def test_get_frames(self):
        game = self.expected_game_tracab_opta.copy()
        res = game.get_frames(1509993)
        pd.testing.assert_frame_equal(
            game.get_frames(1509993), game.get_frames([1509993])
        )
        pd.testing.assert_frame_equal(res, game.tracking_data.iloc[0:1])

        res2 = game.get_frames(1509993, playing_direction="possession_oriented")
        pd.testing.assert_frame_equal(
            game.get_frames(1509993, playing_direction="possession_oriented"),
            game.get_frames([1509993], playing_direction="possession_oriented"),
        )
        cols = ["ball_x", "ball_y", "home_34_x", "home_34_y", "away_17_x", "away_17_y"]
        game.tracking_data[cols] = game.tracking_data[cols] * -1
        pd.testing.assert_frame_equal(res2, game.tracking_data.iloc[0:1])

        with self.assertRaises(ValueError):
            game.get_frames(1509993, playing_direction="wrong")

        with self.assertRaises(ValueError):
            game.get_frames(999)

        game.tracking_data.drop(columns=["ball_possession"], inplace=True)
        with self.assertRaises(ValueError):
            game.get_frames(1509993, playing_direction="possession_oriented")

    def test_get_event_frame(self):
        game = self.expected_game_tracab_opta.copy()
        pass_event = game.get_event(list(game.pass_events.keys())[0])
        game.tracking_data.loc[0, "event_id"] = pass_event.event_id

        with self.assertRaises(DataBallPyError):
            game.get_event_frame(pass_event.event_id)
        game._is_synchronised = True

        with self.assertRaises(ValueError):
            game.get_event_frame(999)

        res_team = game.get_event_frame(
            pass_event.event_id, playing_direction="team_oriented"
        )
        pd.testing.assert_frame_equal(res_team, game.tracking_data.iloc[0:1])

        res_possession = game.get_event_frame(
            pass_event.event_id, playing_direction="possession_oriented"
        )
        pd.testing.assert_frame_equal(
            res_possession, game.tracking_data.iloc[0:1]
        )  # event team side is home

        pass_event.team_side = "away"
        res_possession = game.get_event_frame(
            pass_event.event_id, playing_direction="possession_oriented"
        )
        cols = ["ball_x", "ball_y", "home_34_x", "home_34_y", "away_17_x", "away_17_y"]
        game.tracking_data[cols] = game.tracking_data[cols] * -1
        pd.testing.assert_frame_equal(res_possession, game.tracking_data.iloc[0:1])

        with self.assertRaises(ValueError):
            game.get_event_frame(pass_event.event_id, playing_direction="wrong")
