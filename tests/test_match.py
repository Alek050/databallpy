import datetime as dt
import os
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from databallpy import DataBallPyError
from databallpy.load_data.event_data.instat import load_instat_event_data
from databallpy.load_data.event_data.metrica_event_data import load_metrica_event_data
from databallpy.load_data.event_data.opta import load_opta_event_data
from databallpy.load_data.tracking_data.inmotio import load_inmotio_tracking_data
from databallpy.load_data.tracking_data.metrica_tracking_data import (
    load_metrica_tracking_data,
)
from databallpy.load_data.tracking_data.tracab import load_tracab_tracking_data
from databallpy.match import (
    Match,
    _create_sim_mat,
    _needleman_wunsch,
    align_player_ids,
    get_match,
    get_matching_full_name,
    get_open_match,
    get_saved_match,
)
from expected_outcomes import (
    ED_OPTA,
    MD_INMOTIO,
    MD_INSTAT,
    MD_OPTA,
    MD_TRACAB,
    RES_SIM_MAT,
    RES_SIM_MAT_NO_PLAYER,
    TD_TRACAB,
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

        self.expected_periods_tracab_opta = pd.DataFrame(
            {
                "period": [1, 2, 3, 4, 5],
                "start_frame": [1509993, 1509996, -999, -999, -999],
                "end_frame": [1509994, 1509997, -999, -999, -999],
                "start_datetime_td": [
                    pd.to_datetime("2023-01-14").tz_localize("Europe/Amsterdam")
                    + dt.timedelta(milliseconds=int((1509993 / 25) * 1000)),
                    pd.to_datetime("2023-01-14").tz_localize("Europe/Amsterdam")
                    + dt.timedelta(milliseconds=int((1509996 / 25) * 1000)),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                ],
                "end_datetime_td": [
                    pd.to_datetime("2023-01-14").tz_localize("Europe/Amsterdam")
                    + dt.timedelta(milliseconds=int((1509994 / 25) * 1000)),
                    pd.to_datetime("2023-01-14").tz_localize("Europe/Amsterdam")
                    + dt.timedelta(milliseconds=int((1509997 / 25) * 1000)),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                ],
                "start_datetime_ed": [
                    pd.to_datetime("2023-01-22T12:18:32.000").tz_localize(
                        "Europe/Amsterdam"
                    ),
                    pd.to_datetime("2023-01-22T13:21:13.000").tz_localize(
                        "Europe/Amsterdam"
                    ),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                ],
                "end_datetime_ed": [
                    pd.to_datetime("2023-01-22T13:04:32.000").tz_localize(
                        "Europe/Amsterdam"
                    ),
                    pd.to_datetime("2023-01-22T14:09:58.000").tz_localize(
                        "Europe/Amsterdam"
                    ),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                ],
            }
        )

        self.expected_home_players_tracab_opta = pd.DataFrame(
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

        self.expected_away_players_tracab_opta = pd.DataFrame(
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
            periods=self.expected_periods_tracab_opta,
            frame_rate=self.md_tracab.frame_rate,
            home_team_id=self.md_opta.home_team_id,
            home_formation=self.md_opta.home_formation,
            home_score=self.md_opta.home_score,
            home_team_name=self.md_opta.home_team_name,
            home_players=self.expected_home_players_tracab_opta,
            away_team_id=self.md_opta.away_team_id,
            away_formation=self.md_opta.away_formation,
            away_score=self.md_opta.away_score,
            away_team_name=self.md_opta.away_team_name,
            away_players=self.expected_away_players_tracab_opta,
            country=self.md_opta.country,
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

        # add metrica even data timestamps
        self.md_metrica.periods_frames[
            "start_datetime_ed"
        ] = md_metrica_ed.periods_frames["start_datetime_ed"]
        self.md_metrica.periods_frames[
            "end_datetime_ed"
        ] = md_metrica_ed.periods_frames["end_datetime_ed"]

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
            country=self.md_metrica.country,
        )

        self.td_inmotio_loc = "tests/test_data/inmotio_td_test.txt"
        self.md_inmotio_loc = "tests/test_data/inmotio_metadata_test.xml"
        self.td_inmotio, self.md_inmotio = load_inmotio_tracking_data(
            self.td_inmotio_loc, self.md_inmotio_loc, verbose=False
        )

        self.ed_instat_loc = "tests/test_data/instat_ed_test.json"
        self.md_instat_loc = "tests/test_data/instat_md_test.json"
        self.ed_instat, self.md_instat = load_instat_event_data(
            self.ed_instat_loc, self.md_instat_loc
        )

        self.expected_home_players_inmotio_instat = pd.DataFrame(
            {
                "id": [1, 2],
                "full_name": ["Player 1", "Player 2"],
                "shirt_num": [1, 2],
                "player_type": ["Goalkeeper", "Field player"],
                "start_frame": [2, 2],
                "end_frame": [6, 6],
                "position": ["Goalkeeper", "Defender"],
                "starter": [True, True],
            }
        )

        self.expected_away_players_inmotio_instat = pd.DataFrame(
            {
                "id": [3, 4],
                "full_name": ["Player 11", "Player 12"],
                "shirt_num": [1, 2],
                "player_type": ["Goalkeeper", "Field player"],
                "start_frame": [2, 2],
                "end_frame": [6, 6],
                "position": ["Goalkeeper", "Substitute player"],
                "starter": [True, False],
            }
        )

        self.expected_periods_inmotio_instat = pd.DataFrame(
            {
                "period": [1, 2, 3, 4, 5],
                "start_frame": [2, 5, -999, -999, -999],
                "end_frame": [3, 6, -999, -999, -999],
                "start_datetime_td": [
                    pd.to_datetime("2023-01-01 20:00:00").tz_localize(
                        "Europe/Amsterdam"
                    ),
                    pd.to_datetime("2023-01-01 21:00:00").tz_localize(
                        "Europe/Amsterdam"
                    ),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                ],
                "end_datetime_td": [
                    pd.to_datetime("2023-01-01 20:45:00").tz_localize(
                        "Europe/Amsterdam"
                    ),
                    pd.to_datetime("2023-01-01 21:45:00").tz_localize(
                        "Europe/Amsterdam"
                    ),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                ],
                "start_datetime_ed": [
                    pd.to_datetime("2023-01-01 20:00:00").tz_localize(
                        "Europe/Amsterdam"
                    ),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                ],
                "end_datetime_ed": [
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                ],
            }
        )

        self.expected_match_inmotio_instat = Match(
            tracking_data=self.td_inmotio,
            tracking_data_provider="inmotio",
            event_data=self.ed_instat,
            event_data_provider="instat",
            pitch_dimensions=self.md_inmotio.pitch_dimensions,
            periods=self.expected_periods_inmotio_instat,
            frame_rate=self.md_inmotio.frame_rate,
            home_team_id=self.md_instat.home_team_id,
            home_formation=self.md_instat.home_formation,
            home_score=self.md_instat.home_score,
            home_team_name=self.md_instat.home_team_name,
            home_players=self.expected_home_players_inmotio_instat,
            away_team_id=self.md_instat.away_team_id,
            away_formation=self.md_instat.away_formation,
            away_score=self.md_instat.away_score,
            away_team_name=self.md_instat.away_team_name,
            away_players=self.expected_away_players_inmotio_instat,
            country=self.md_instat.country,
        )

        self.match_to_sync = get_match(
            tracking_data_loc="tests/test_data/sync/tracab_td_sync_test.dat",
            tracking_metadata_loc="tests/test_data/sync/tracab_metadata_sync_test.xml",
            tracking_data_provider="tracab",
            event_data_loc="tests/test_data/sync/opta_events_sync_test.xml",
            event_metadata_loc="tests/test_data/sync/opta_metadata_sync_test.xml",
            event_data_provider="opta",
        )

        self.expected_match_tracab = Match(
            tracking_data=TD_TRACAB,
            tracking_data_provider="tracab",
            event_data=pd.DataFrame(),
            event_data_provider=None,
            pitch_dimensions=MD_TRACAB.pitch_dimensions,
            periods=MD_TRACAB.periods_frames,
            frame_rate=MD_TRACAB.frame_rate,
            home_team_id=MD_TRACAB.home_team_id,
            home_formation=MD_TRACAB.home_formation,
            home_score=MD_TRACAB.home_score,
            home_team_name=MD_TRACAB.home_team_name,
            home_players=MD_TRACAB.home_players,
            away_team_id=MD_TRACAB.away_team_id,
            away_formation=MD_TRACAB.away_formation,
            away_score=MD_TRACAB.away_score,
            away_team_name=MD_TRACAB.away_team_name,
            away_players=MD_TRACAB.away_players,
            country=MD_TRACAB.country,
        )

        self.expected_match_opta = Match(
            tracking_data=pd.DataFrame(),
            tracking_data_provider=None,
            event_data=ED_OPTA,
            event_data_provider="opta",
            pitch_dimensions=MD_OPTA.pitch_dimensions,
            periods=MD_OPTA.periods_frames,
            frame_rate=MD_OPTA.frame_rate,
            home_team_id=MD_OPTA.home_team_id,
            home_formation=MD_OPTA.home_formation,
            home_score=MD_OPTA.home_score,
            home_team_name=MD_OPTA.home_team_name,
            home_players=MD_OPTA.home_players,
            away_team_id=MD_OPTA.away_team_id,
            away_formation=MD_OPTA.away_formation,
            away_score=MD_OPTA.away_score,
            away_team_name=MD_OPTA.away_team_name,
            away_players=MD_OPTA.away_players,
            country=MD_OPTA.country,
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
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=pd.DataFrame(
                    {"frame": [1], "home_1_x": [12], "ball_z": [13]}
                ),
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
            )

        # tracking data provider
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=14.3,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
            )

        # event data
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data="event_data",
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=pd.DataFrame(
                    {
                        "event_id": [1],
                        "player": ["player_1"],
                        "event": ["pass"],
                    }
                ),
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
            )

        # event data provider
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=["opta"],
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
            )

        # pitch dimensions
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions={1: 22, 2: 11},
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=[10.0, 11.0, 12.0],
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
            )

        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=[10, 11.0],
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
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
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
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
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
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
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
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
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
            )

        with self.assertRaises(ValueError):
            periods = self.expected_periods_tracab_opta.copy()
            periods["start_datetime_ed"] = periods["start_datetime_ed"].dt.tz_localize(
                None
            )
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=periods,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
            )

        # frame rate
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods_tracab_opta,
                frame_rate=25.0,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods_tracab_opta,
                frame_rate=-25,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
            )

        # team id
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=123.0,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
            )

        # team name
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=["teamone"],
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
            )

        # team score
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=11.5,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=-3,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
            )

        # team formation
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=[1, 4, 2, 2],
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation="one-four-three-three",
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
            )

        # team players
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods_tracab_opta,
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
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
            )

        with self.assertRaises(ValueError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta.drop(
                    "shirt_num", axis=1
                ),
                country=self.md_opta.country,
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
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
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
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
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
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
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
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=self.md_opta.country,
            )

        # country
        with self.assertRaises(TypeError):
            Match(
                tracking_data=self.td_tracab,
                tracking_data_provider=self.td_provider,
                event_data=self.corrected_ed,
                event_data_provider=self.ed_provider,
                pitch_dimensions=self.md_tracab.pitch_dimensions,
                periods=self.expected_periods_tracab_opta,
                frame_rate=self.md_tracab.frame_rate,
                home_team_id=self.md_opta.home_team_id,
                home_formation=self.md_opta.home_formation,
                home_score=self.md_opta.home_score,
                home_team_name=self.md_opta.home_team_name,
                home_players=self.expected_home_players_tracab_opta,
                away_team_id=self.md_opta.away_team_id,
                away_formation=self.md_opta.away_formation,
                away_score=self.md_opta.away_score,
                away_team_name=self.md_opta.away_team_name,
                away_players=self.expected_away_players_tracab_opta,
                country=["Netherlands", "Germany"],
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

    def test__repr__and_name(self):
        assert (
            self.match_to_sync.__repr__()
            == "databallpy.match.Match object: TeamOne 3 - 1 \
TeamTwo 2023-01-22 16:46:39"
        )
        assert self.match_to_sync.name == "TeamOne 3 - 1 TeamTwo 2023-01-22 16:46:39"

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
        expected_res = RES_SIM_MAT
        expected_res = expected_res.reshape(13, 4)

        tracking_data = self.match_to_sync.tracking_data
        date = pd.to_datetime(
            str(self.match_to_sync.periods.iloc[0, 3])[:10]
        ).tz_localize("Europe/Amsterdam")
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
        expected_res = RES_SIM_MAT_NO_PLAYER
        expected_res = expected_res.reshape(13, 4)

        tracking_data = self.match_to_sync.tracking_data
        date = pd.to_datetime(
            str(self.match_to_sync.periods.iloc[0, 3])[:10]
        ).tz_localize("Europe/Amsterdam")
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

    def test_match_inmotio_instat(self):
        match_instat_inmotio = get_match(
            tracking_data_loc=self.td_inmotio_loc,
            tracking_metadata_loc=self.md_inmotio_loc,
            tracking_data_provider="inmotio",
            event_data_loc=self.ed_instat_loc,
            event_metadata_loc=self.md_instat_loc,
            event_data_provider="instat",
        )

        assert match_instat_inmotio == self.expected_match_inmotio_instat

    def test_match_inmotio_instat_unaligned_player_ids(self):
        match_instat_inmotio_unaligned_input = get_match(
            tracking_data_loc=self.td_inmotio_loc,
            tracking_metadata_loc=self.md_inmotio_loc,
            tracking_data_provider="inmotio",
            event_data_loc="tests/test_data/instat_ed_test_unaligned_player_ids.json",
            event_metadata_loc=self.md_instat_loc,
            event_data_provider="instat",
        )

        assert (
            match_instat_inmotio_unaligned_input == self.expected_match_inmotio_instat
        )

    def test_align_player_ids(self):
        unaligned_metadata = MD_INSTAT.copy()
        unaligned_metadata.away_players.loc[0, "id"] = 9
        aligned_metadata = align_player_ids(unaligned_metadata, MD_INMOTIO)
        assert aligned_metadata == MD_INSTAT

    def test_get_saved_match(self):
        expected_match = self.expected_match_metrica
        expected_match.save_match(name="test_match", path="tests/test_data")
        saved_match = get_saved_match(name="test_match", path="tests/test_data")
        assert expected_match == saved_match
        assert saved_match != self.expected_match_tracab_opta

    def test_match_only_tracking_data(self):
        match = get_match(
            tracking_data_loc=self.td_tracab_loc,
            tracking_metadata_loc=self.md_tracab_loc,
            tracking_data_provider="tracab",
        )
        assert match == self.expected_match_tracab

    def test_match_only_event_data(self):
        match = get_match(
            event_data_loc=self.ed_opta_loc,
            event_metadata_loc=self.md_opta_loc,
            event_data_provider="opta",
        )
        # match pitch dimensions = [10, 10],
        # while expected_match pitch dimensions = [106, 68]
        x_cols = [col for col in match.event_data.columns if "_x" in col]
        y_cols = [col for col in match.event_data.columns if "_y" in col]
        match.event_data[x_cols] = match.event_data[x_cols] / 106.0 * 10
        match.event_data[y_cols] = match.event_data[y_cols] / 68.0 * 10
        match.pitch_dimensions = [10.0, 10.0]
        assert match == self.expected_match_opta

    def test_match_requires_event_data_wrapper(self):
        match = self.expected_match_opta.copy()
        with self.assertRaises(DataBallPyError):
            match.synchronise_tracking_and_event_data()

    def test_match_requires_tracking_data_wrapper(self):
        match = self.expected_match_tracab.copy()
        with self.assertRaises(DataBallPyError):
            match.synchronise_tracking_and_event_data()

    def test_get_match_no_valid_input(self):
        with self.assertRaises(ValueError):
            get_match()
