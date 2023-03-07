import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from databallpy import DataBallPyError
from databallpy.load_data.event_data.metrica_event_data import load_metrica_event_data
from databallpy.load_data.event_data.opta import load_opta_event_data
from databallpy.load_data.tracking_data.metrica_tracking_data import (
    load_metrica_tracking_data,
)
from databallpy.load_data.tracking_data.tracab import load_tracab_tracking_data
from databallpy.match import Match, get_match, get_open_match
from tests.mocks import ED_METRICA_RAW, MD_METRICA_RAW, TD_METRICA_RAW


class TestMatch(unittest.TestCase):
    def setUp(self):
        self.td_tracab_loc = "tests/test_data/tracab_td_test.dat"
        self.md_tracab_loc = "tests/test_data/tracab_metadata_test.xml"
        self.td_provider = "tracab"
        self.ed_opta_loc = "tests/test_data/f24_test.xml"
        self.md_opta_loc = "tests/test_data/f7_test.xml"
        self.ed_provider = "opta"

        self.td_tracab, self.md_tracab = load_tracab_tracking_data(
            self.td_tracab_loc, self.md_tracab_loc
        )
        self.ed_opta, self.md_opta = load_opta_event_data(
            f7_loc=self.md_opta_loc, f24_loc=self.ed_opta_loc
        )

        self.corrected_ed = self.ed_opta.copy()
        self.corrected_ed["start_x"] *= (
            100.0 / 106.0
        )  # pitch dimensions of td and ed metadata
        self.corrected_ed["start_y"] *= 50.0 / 68.0

        self.expected_periods = pd.DataFrame(
            {
                "period": [1, 2, 3, 4, 5],
                "start_frame": [1509993, 1509996, 0, 0, 0],
                "end_frame": [1509994, 1509997, 0, 0, 0],
                "start_time_td": [
                    np.datetime64("2023-01-14")
                    + np.timedelta64(int(1509993 / 25), "s"),
                    np.datetime64("2023-01-14")
                    + np.timedelta64(int(1509996 / 25), "s"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                ],
                "end_time_td": [
                    np.datetime64("2023-01-14")
                    + np.timedelta64(int(1509994 / 25), "s"),
                    np.datetime64("2023-01-14")
                    + np.timedelta64(int(1509997 / 25), "s"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                ],
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

        self.expected_home_players = pd.DataFrame(
            {
                "id": [19367, 45849],
                "full_name": ["Piet Schrijvers", "Jan Boskamp"],
                "shirt_num": [1, 2],
                "start_frame": [1509993, 1509993],
                "end_frame": [1509997, 1509995],
                "formation_place": [4, 0],
                "position": ["midfielder", "midfielder"],
                "starter": [True, False],
            }
        )

        self.expected_away_players = pd.DataFrame(
            {
                "id": [184934, 450445],
                "full_name": ["Pepijn Blok", "TestSpeler"],
                "shirt_num": [1, 2],
                "start_frame": [1509993, 1509993],
                "end_frame": [1509997, 1509994],
                "formation_place": [8, 0],
                "position": ["midfielder", "midfielder"],
                "starter": [True, False],
            }
        )

        self.expected_match_tracab_opta = Match(
            tracking_data=self.td_tracab,
            tracking_data_provider=self.td_provider,
            event_data=self.corrected_ed,
            event_data_provider=self.ed_provider,
            pitch_dimensions=self.md_tracab.pitch_dimensions,
            periods=self.expected_periods,
            frame_rate=self.md_tracab.frame_rate,
            home_team_id=self.md_opta.home_team_id,
            home_formation=self.md_opta.home_formation,
            home_score=self.md_opta.home_score,
            home_team_name=self.md_opta.home_team_name,
            home_players=self.expected_home_players,
            away_team_id=self.md_opta.away_team_id,
            away_formation=self.md_opta.away_formation,
            away_score=self.md_opta.away_score,
            away_team_name=self.md_opta.away_team_name,
            away_players=self.expected_away_players,
        )
        self.td_metrica_loc = "tests/test_data/metrica_tracking_data_test.txt"
        self.md_metrica_loc = "tests/test_data/metrica_metadata_test.xml"
        self.ed_metrica_loc = "tests/test_data/metrica_event_data_test.json"
        self.td_metrica, self.md_metrica = load_metrica_tracking_data(
            self.td_metrica_loc, self.md_metrica_loc
        )
        self.ed_metrica, _ = load_metrica_event_data(
            self.ed_metrica_loc, self.md_metrica_loc
        )

        self.expected_match_metrica = Match(
            tracking_data=self.td_metrica,
            tracking_data_provider="metrica",
            event_data=self.ed_metrica,
            event_data_provider="metrica",
            pitch_dimensions=self.md_metrica.pitch_dimensions,
            periods=self.md_metrica.periods_frames,
            frame_rate=self.md_metrica.frame_rate,
            home_team_id=self.md_metrica.home_team_id,
            home_formation=self.md_metrica.home_formation,
            home_score=self.md_metrica.home_score,
            home_team_name=self.md_metrica.home_team_name,
            home_players=self.md_metrica.home_players,
            away_team_id=self.md_metrica.away_team_id,
            away_formation=self.md_metrica.away_formation,
            away_score=self.md_metrica.away_score,
            away_team_name=self.md_metrica.away_team_name,
            away_players=self.md_metrica.away_players,
        )

    def test_match_eq(self):
        assert self.expected_match_metrica == self.expected_match_metrica
        assert self.expected_match_metrica != self.expected_match_tracab_opta

    def test_match_post_init(self):

        # tracking data
        with self.assertRaises(TypeError):
            Match(
                tracking_data="tracking_data",
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=pd.DataFrame(
                    {"timestamp": [], "ball_x": [], "ball_z": []}
                ),
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        # tracking data provider
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=14.3,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        # event data
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data="event_data",
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=pd.DataFrame(
                    {
                        "event_id": [],
                        "player": [],
                        "event": [],
                    }
                ),
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        # event data provider
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=["opta"],
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        # pitch dimensions
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions={1: 22, 2: 11},
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=[10.0, 11.0, 12.0],
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=[10, 11.0],
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        # periods
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=[1, 2, 3, 4, 5],
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=pd.DataFrame({"times": [1, 2, 3, 4, 5]}),
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=pd.DataFrame({"period": [0, 1, 2, 3, 4]}),
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=pd.DataFrame({"period": [1, 1, 2, 3, 4, 5]}),
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        # frame rate
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods,
                frame_rate=25.0,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods,
                frame_rate=-25,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        # team id
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=123.0,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        # team name
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=["teamone"],
                away_players=self.expected_away_players,
            )

        # team score
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=11.5,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=-3,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        # team formation
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=[1, 4, 2, 2],
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation="one-four-three-three",
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        # team players
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=["player1", "player2"],
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players.drop("shirt_num", axis=1),
            )

        # pitch axis
        with self.assertRaises(DataBallPyError):
            td_changed = self.td_tracab.copy()
            td_changed["ball_x"] += 10.0
            Match(
                tracking_data=td_changed,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )
        with self.assertRaises(DataBallPyError):
            td_changed = self.td_tracab.copy()
            td_changed["ball_y"] += 10.0
            Match(
                tracking_data=td_changed,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        # playing direction
        with self.assertRaises(DataBallPyError):
            td_changed = self.td_tracab.copy()
            td_changed.loc[0, "home_34_x"] = 3.0
            Match(
                tracking_data=td_changed,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

        with self.assertRaises(DataBallPyError):
            td_changed = self.td_tracab.copy()
            td_changed.loc[0, "away_17_x"] = -3.0
            Match(
                tracking_data=td_changed,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players,
            )

    def test_get_match_wrong_provider(self):
        self.assertRaises(
            AssertionError,
            get_match,
            tracking_data_loc=self.td_tracab_loc,
            tracking_metadata_loc=self.md_tracab_loc,
            event_data_loc=self.ed_opta_loc,
            event_metadata_loc=self.md_opta_loc,
            tracking_data_provider=self.td_provider,
            event_data_provider="wrong",
        )

        self.assertRaises(
            AssertionError,
            get_match,
            tracking_data_loc=self.td_tracab_loc,
            tracking_metadata_loc=self.md_tracab_loc,
            event_data_loc=self.ed_opta_loc,
            event_metadata_loc=self.md_opta_loc,
            tracking_data_provider="also wrong",
            event_data_provider=self.ed_provider,
        )

    def test_get_match(self):
        match = get_match(
            tracking_data_loc=self.td_tracab_loc,
            tracking_metadata_loc=self.md_tracab_loc,
            event_data_loc=self.ed_opta_loc,
            event_metadata_loc=self.md_opta_loc,
            tracking_data_provider=self.td_provider,
            event_data_provider=self.ed_provider,
        )

        assert match == self.expected_match_tracab_opta

    def test_match__eq__(self):
        assert not self.expected_match_tracab_opta == pd.DataFrame()

    def test_match_name(self):
        assert self.expected_match_tracab_opta.name == "TeamOne 3 - 1 TeamTwo"

    def test_match_home_players_column_ids(self):
        assert self.expected_match_tracab_opta.home_players_column_ids == [
            "home_34_x",
            "home_34_y",
        ]

    def test_match_away_players_column_ids(self):
        assert self.expected_match_tracab_opta.away_players_column_ids == [
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

    def test_match_metrica_data(self):

        res = get_match(
            tracking_data_loc=self.td_metrica_loc,
            tracking_metadata_loc=self.md_metrica_loc,
            tracking_data_provider="metrica",
            event_data_loc=self.ed_metrica_loc,
            event_metadata_loc=self.md_metrica_loc,
            event_data_provider="metrica",
        )
        assert res == self.expected_match_metrica

    def test_match_wrong_input(self):
        assert not self.expected_match_tracab_opta == 3

    @patch(
        "requests.get",
        side_effect=[
            Mock(text=TD_METRICA_RAW),
            Mock(text=MD_METRICA_RAW),
            Mock(text=ED_METRICA_RAW),
            Mock(text=MD_METRICA_RAW),
        ],
    )
    def test_get_open_match(self, _):
        match = get_open_match()
        assert match == self.expected_match_metrica
