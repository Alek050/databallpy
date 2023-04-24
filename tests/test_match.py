import datetime as dt
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
from databallpy.load_data.tracking_data.fifa import load_fifa_tracking_data
from databallpy.load_data.event_data.instat import load_instat_event_data
from databallpy.match import (
    Match,
    _create_sim_mat,
    _needleman_wunsch,
    get_match,
    get_open_match,
    get_matching_full_name,
)
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
                "start_frame": [1509993, 1509996, -999, -999, -999],
                "end_frame": [1509994, 1509997, -999, -999, -999],
                "start_datetime_td": [
                    pd.to_datetime("2023-01-14")
                    + dt.timedelta(milliseconds=int((1509993 / 25) * 1000)),
                    pd.to_datetime("2023-01-14")
                    + dt.timedelta(milliseconds=int((1509996 / 25) * 1000)),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                ],
                "end_datetime_td": [
                    pd.to_datetime("2023-01-14")
                    + dt.timedelta(milliseconds=int((1509994 / 25) * 1000)),
                    pd.to_datetime("2023-01-14")
                    + dt.timedelta(milliseconds=int((1509997 / 25) * 1000)),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                ],
                "start_datetime_ed": [
                    pd.to_datetime("2023-01-22T12:18:32.000"),
                    pd.to_datetime("2023-01-22T13:21:13.000"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                ],
                "end_datetime_ed": [
                    pd.to_datetime("2023-01-22T13:04:32.000"),
                    pd.to_datetime("2023-01-22T14:09:58.000"),
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
        self.td_tracab["period"] = [1, 1, -999, 2, 2]
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
        self.td_metrica["period"] = [1, 1, 1, 2, 2, 2]
        self.ed_metrica, md_metrica_ed = load_metrica_event_data(
            self.ed_metrica_loc, self.md_metrica_loc
        )

        self.md_metrica.periods_frames[
            "start_datetime_ed"
        ] = md_metrica_ed.periods_frames["start_datetime_ed"].values
        self.md_metrica.periods_frames[
            "end_datetime_ed"
        ] = md_metrica_ed.periods_frames["end_datetime_ed"].values

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

        self.td_fifa_loc = "tests/test_data/fifa_td_test.txt"
        self.md_fifa_loc = "tests/test_data/fifa_metadata_test.xml"
        self.td_fifa, self.md_fifa = load_fifa_tracking_data(
            self.td_fifa_loc, self.md_fifa_loc, verbose=False
        )

        self.ed_instat_loc = "tests/test_data/instat_ed_test.json"
        self.md_instat_loc = "tests/test_data/instat_md_test.json"
        self.ed_instat, self.md_instat = load_instat_event_data(
            self.ed_instat_loc, self.md_instat_loc
        )

        player_cols_fifa_instat = self.md_instat.home_players.columns.difference(
            self.md_fifa.home_players.columns
        ).to_list()
        player_cols_fifa_instat.append("id")
        home_players = self.md_fifa.home_players.merge(
            self.md_instat.home_players[player_cols_fifa_instat], on="id"
        )
        away_players = self.md_fifa.away_players.merge(
            self.md_instat.away_players[player_cols_fifa_instat], on="id"
        )

        if not self.md_fifa.pitch_dimensions == self.md_instat.pitch_dimensions:
            x_correction = (
                self.md_fifa.pitch_dimensions[0] / self.md_instat.pitch_dimensions[0]
            )
            y_correction = (
                self.md_fifa.pitch_dimensions[1] / self.md_instat.pitch_dimensions[1]
            )
            self.ed_instat["start_x"] *= x_correction
            self.ed_instat["start_y"] *= y_correction
        
        periods_cols = self.md_instat.periods_frames.columns.difference(
                self.md_fifa.periods_frames.columns
            ).to_list()
        periods_cols.sort(reverse=True)
        merged_periods = pd.concat(
                (
                    self.md_fifa.periods_frames,
                    self.md_instat.periods_frames[periods_cols],
                ),
                axis=1,
            )

        self.expected_match_fifa_instat = Match(
            tracking_data=self.td_fifa,
            tracking_data_provider="fifa",
            event_data=self.ed_instat,
            event_data_provider="instat",
            pitch_dimensions=self.md_fifa.pitch_dimensions,
            periods=merged_periods,
            frame_rate=self.md_fifa.frame_rate,
            home_team_id=self.md_instat.home_team_id,
            home_formation=self.md_instat.home_formation,
            home_score=self.md_instat.home_score,
            home_team_name=self.md_instat.home_team_name,
            home_players=home_players,
            away_team_id=self.md_instat.away_team_id,
            away_formation=self.md_instat.away_formation,
            away_score=self.md_instat.away_score,
            away_team_name=self.md_instat.away_team_name,
            away_players=away_players,
        )

        self.match_to_sync = get_match(
            tracking_data_loc="tests/test_data/sync/tracab_td_sync_test.dat",
            tracking_metadata_loc="tests/test_data/sync/tracab_metadata_sync_test.xml",
            tracking_data_provider="tracab",
            event_data_loc="tests/test_data/sync/opta_events_sync_test.xml",
            event_metadata_loc="tests/test_data/sync/opta_metadata_sync_test.xml",
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
                tracking_data=pd.DataFrame({"frame": [], "ball_x": [], "ball_z": []}),
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
        assert (
            self.expected_match_tracab_opta.name
            == "TeamOne 3 - 1 TeamTwo 2023-01-14 16:46:39"
        )

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

    def test_synchronise_tracking_and_event_data(self):
        expected_event_data = self.match_to_sync.event_data.copy()
        expected_tracking_data = self.match_to_sync.tracking_data.copy()
        expected_tracking_data["period"] = [1] * 13
        expected_tracking_data["event"] = [
            "pass",
            "pass",
            np.nan,
            np.nan,
            np.nan,
            "take on",
            "tackle",
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]
        expected_tracking_data["event_id"] = [
            2499594225,
            2499594243,
            np.nan,
            np.nan,
            np.nan,
            2499594285,
            2499594291,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]

        expected_event_data.loc[:, "tracking_frame"] = [
            np.nan,
            np.nan,
            np.nan,
            0.0,
            1.0,
            np.nan,
            np.nan,
            5.0,
            6.0,
        ]
        expected_event_data = expected_event_data[
            expected_event_data["type_id"].isin([1, 3, 7])
        ]

        self.match_to_sync.synchronise_tracking_and_event_data(n_batches_per_half=1)

        pd.testing.assert_frame_equal(
            self.match_to_sync.tracking_data, expected_tracking_data
        )
        pd.testing.assert_frame_equal(
            self.match_to_sync.event_data, expected_event_data
        )

    def test_needleman_wunsch(self):
        sim_list = [
            0,
            0,
            0,
            0.9,
            0,
            0,
            0,
            0,
            0,
            0,
            0.9,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.9,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        sim_mat = np.array(sim_list).reshape(10, 3)

        res = _needleman_wunsch(sim_mat)
        expected_res = {0: 1, 1: 3, 2: 7}

        assert res == expected_res

    def test_create_sim_mat(self):
        expected_res = np.array(
            [
                0.40006852,
                0.40006852,
                0.39676846,
                0.42604787,
                0.39410369,
                0.39410369,
                0.3922753,
                0.42664872,
                0.38802166,
                0.38802166,
                0.38767596,
                0.42703609,
                0.36787944,
                0.36787944,
                0.36787944,
                0.36787944,
                0.37342119,
                0.37342119,
                0.37878323,
                0.42755615,
                0.39987463,
                0.39987463,
                0.40133888,
                0.42945239,
                0.39263378,
                0.39263378,
                0.39708269,
                0.42795639,
                0.38521479,
                0.38521479,
                0.39260596,
                0.41395895,
                0.37126589,
                0.37126589,
                0.38263939,
                0.40387744,
                0.39703003,
                0.39703003,
                0.40487744,
                0.4017941,
                0.38921197,
                0.38921197,
                0.39914273,
                0.41462662,
                0.38150169,
                0.38150169,
                0.39270907,
                0.4194602,
                0.36787944,
                0.36787944,
                0.38141911,
                0.41999494,
            ]
        )
        expected_res = expected_res.reshape(13, 4)

        tracking_data = self.match_to_sync.tracking_data
        date = pd.to_datetime(str(self.match_to_sync.periods.iloc[0, 3])[:10])
        tracking_data["datetime"] = [
            date
            + dt.timedelta(milliseconds=int(x / self.match_to_sync.frame_rate * 1000))
            for x in tracking_data["frame"]
        ]
        tracking_data.reset_index(inplace=True)
        event_data = self.match_to_sync.event_data
        event_data = event_data[event_data["type_id"].isin([1, 3, 7])].reset_index()
        res = _create_sim_mat(
            tracking_batch=tracking_data,
            event_batch=event_data,
            match=self.match_to_sync,
        )

        np.testing.assert_allclose(expected_res, res)

    def test_create_sim_mat_without_player(self):
        expected_res = np.array(
            [
                0.40006852,
                0.40006852,
                0.86247631,
                0.42604787,
                0.39410369,
                0.39410369,
                0.8646367,
                0.42664872,
                0.38802166,
                0.38802166,
                0.86681802,
                0.42703609,
                0.36787944,
                0.36787944,
                0.36787944,
                0.36787944,
                0.37342119,
                0.37342119,
                0.87099448,
                0.42755615,
                0.39987463,
                0.39987463,
                0.87109937,
                0.42945239,
                0.39263378,
                0.39263378,
                0.87328136,
                0.42795639,
                0.38521479,
                0.38521479,
                0.87548448,
                0.41395895,
                0.37126589,
                0.37126589,
                0.87795412,
                0.40387744,
                0.39703003,
                0.39703003,
                0.87805984,
                0.4017941,
                0.38921197,
                0.38921197,
                0.87850958,
                0.41462662,
                0.38150169,
                0.38150169,
                0.87722815,
                0.4194602,
                0.36787944,
                0.36787944,
                0.87620901,
                0.41999494,
            ]
        )
        expected_res = expected_res.reshape(13, 4)

        tracking_data = self.match_to_sync.tracking_data
        date = pd.to_datetime(str(self.match_to_sync.periods.iloc[0, 3])[:10])
        tracking_data["datetime"] = [
            date
            + dt.timedelta(milliseconds=int(x / self.match_to_sync.frame_rate * 1000))
            for x in tracking_data["frame"]
        ]
        tracking_data.reset_index(inplace=True)
        event_data = self.match_to_sync.event_data
        event_data = event_data[event_data["type_id"].isin([1, 3, 7])].reset_index()
        event_data.loc[2, "player_id"] = np.nan

        res = _create_sim_mat(
            tracking_batch=tracking_data,
            event_batch=event_data,
            match=self.match_to_sync,
        )

        np.testing.assert_allclose(expected_res, res, rtol=1e-05)

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
        pd.testing.assert_frame_equal(
            match.periods, self.expected_match_metrica.periods
        )
        assert match == self.expected_match_metrica
    
    def test_get_matching_full_name(self):
        input = "Bart Christaan Albert van den Boom"
        options = ["Bart Chris", "Bart van den Boom", "Piet Pieters"]
        output = get_matching_full_name(input, options)
        assert output == "Bart van den Boom"

    def test_match_fifa_instat(self):
        match_instat_fifa = get_match(
            tracking_data_loc=self.td_fifa_loc,
            tracking_metadata_loc=self.md_fifa_loc,
            tracking_data_provider="fifa",
            event_data_loc=self.ed_instat_loc,
            event_metadata_loc=self.md_instat_loc,
            event_data_provider="instat"
        )

        assert match_instat_fifa == self.expected_match_fifa_instat
