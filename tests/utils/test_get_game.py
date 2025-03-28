import datetime as dt
import os
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from databallpy.data_parsers.event_data_parsers import (
    load_instat_event_data,
    load_metrica_event_data,
    load_opta_event_data,
)
from databallpy.data_parsers.tracking_data_parsers import (
    load_inmotio_tracking_data,
    load_metrica_tracking_data,
    load_tracab_tracking_data,
)
from databallpy.game import Game
from databallpy.schemas import EventData, TrackingData
from databallpy.utils.constants import MISSING_INT
from databallpy.utils.get_game import (
    get_game,
    get_match,
    get_open_game,
    get_open_match,
    get_saved_game,
    get_saved_match,
)
from tests.expected_outcomes import (
    DRIBBLE_EVENTS_METRICA,
    DRIBBLE_EVENTS_OPTA,
    DRIBBLE_EVENTS_OPTA_TRACAB,
    DRIBBLE_EVENTS_STATSBOMB,
    ED_OPTA,
    ED_SCISPORTS,
    ED_STATSBOMB,
    MD_OPTA,
    MD_SCISPORTS,
    MD_STATSBOMB,
    MD_TRACAB,
    PASS_EVENTS_METRICA,
    PASS_EVENTS_OPTA,
    PASS_EVENTS_OPTA_TRACAB,
    PASS_EVENTS_STATSBOMB,
    SHOT_EVENTS_METRICA,
    SHOT_EVENTS_OPTA,
    SHOT_EVENTS_OPTA_TRACAB,
    SHOT_EVENTS_STATSBOMB,
    SPORTEC_DATABALLPY_EVENTS,
    SPORTEC_DRIBBLE_EVENTS,
    SPORTEC_EVENT_DATA,
    SPORTEC_METADATA_ED,
    SPORTEC_METADATA_TD,
    SPORTEC_PASS_EVENTS,
    SPORTEC_SHOT_EVENTS,
    TD_TRACAB,
    TRACAB_SPORTEC_XML_TD,
)

from ..mocks import ED_METRICA_RAW, MD_METRICA_RAW, TD_METRICA_RAW


class TestGetGame(unittest.TestCase):
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
        self.td_tracab = TrackingData(
            self.td_tracab,
            provider=self.td_provider,
            frame_rate=1,
        )
        self.ed_opta, self.md_opta, self.dbp_events = load_opta_event_data(
            f7_loc=self.md_opta_loc, f24_loc=self.ed_opta_loc
        )

        self.corrected_ed = EventData(self.ed_opta.copy(), provider="opta")
        self.corrected_ed["start_x"] *= (
            100.0 / 106.0
        )  # pitch dimensions of td and ed metadata
        self.corrected_ed["start_y"] *= 50.0 / 68.0

        self.expected_periods_tracab_opta = pd.DataFrame(
            {
                "period_id": [1, 2, 3, 4, 5],
                "start_frame": [
                    1509993,
                    1509996,
                    MISSING_INT,
                    MISSING_INT,
                    MISSING_INT,
                ],
                "end_frame": [1509994, 1509997, MISSING_INT, MISSING_INT, MISSING_INT],
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
                "position": ["goalkeeper", "midfielder"],
                "starter": [True, True],
            }
        )

        self.expected_away_players_tracab_opta = pd.DataFrame(
            {
                "id": [184934, 450445],
                "full_name": ["Pepijn Blok", "TestSpeler"],
                "shirt_num": [1, 2],
                "start_frame": [1509993, 1509993],
                "end_frame": [1509997, 1509995],
                "formation_place": [8, 0],
                "position": ["midfielder", "goalkeeper"],
                "starter": [True, True],
            }
        )

        self.td_tracab["period_id"] = [1, 1, MISSING_INT, 2, 2]

        self.expected_game_tracab_opta = Game(
            tracking_data=self.td_tracab,
            event_data=self.corrected_ed,
            pitch_dimensions=self.md_tracab.pitch_dimensions,
            periods=self.expected_periods_tracab_opta,
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
            shot_events=SHOT_EVENTS_OPTA_TRACAB.copy(),
            dribble_events=DRIBBLE_EVENTS_OPTA_TRACAB.copy(),
            pass_events=PASS_EVENTS_OPTA_TRACAB.copy(),
            _tracking_timestamp_is_precise=True,
            _event_timestamp_is_precise=True,
            _periods_changed_playing_direction=[],
            allow_synchronise_tracking_and_event_data=True,
        )

        self.td_metrica_loc = "tests/test_data/metrica_tracking_data_test.txt"
        self.md_metrica_loc = "tests/test_data/metrica_metadata_test.xml"
        self.ed_metrica_loc = "tests/test_data/metrica_event_data_test.json"
        self.td_metrica, self.md_metrica = load_metrica_tracking_data(
            self.td_metrica_loc, self.md_metrica_loc
        )
        self.td_metrica = TrackingData(
            self.td_metrica,
            provider="metrica",
            frame_rate=1,
        )
        self.td_metrica["period_id"] = [1, 1, 1, 2, 2, 2]
        self.ed_metrica, md_metrica_ed, _ = load_metrica_event_data(
            self.ed_metrica_loc, self.md_metrica_loc
        )

        # add metrica even data timestamps
        self.md_metrica.periods_frames["start_datetime_ed"] = (
            md_metrica_ed.periods_frames["start_datetime_ed"]
        )
        self.md_metrica.periods_frames["end_datetime_ed"] = md_metrica_ed.periods_frames[
            "end_datetime_ed"
        ]

        self.expected_game_metrica = Game(
            tracking_data=self.td_metrica,
            event_data=EventData(self.ed_metrica, provider="metrica"),
            pitch_dimensions=self.md_metrica.pitch_dimensions,
            periods=self.md_metrica.periods_frames,
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
            pass_events=PASS_EVENTS_METRICA,
            shot_events=SHOT_EVENTS_METRICA,
            dribble_events=DRIBBLE_EVENTS_METRICA,
            _tracking_timestamp_is_precise=True,
            _event_timestamp_is_precise=True,
            _periods_changed_playing_direction=[],
        )

        self.td_inmotio_loc = "tests/test_data/inmotio_td_test.txt"
        self.md_inmotio_loc = "tests/test_data/inmotio_metadata_test.xml"
        self.td_inmotio, self.md_inmotio = load_inmotio_tracking_data(
            self.td_inmotio_loc, self.md_inmotio_loc, verbose=False
        )
        self.td_inmotio = TrackingData(
            self.td_inmotio,
            provider="inmotio",
            frame_rate=1,
        )
        self.ed_instat_loc = "tests/test_data/instat_ed_test.json"
        self.md_instat_loc = "tests/test_data/instat_md_test.json"
        self.ed_instat, self.md_instat, _ = load_instat_event_data(
            self.ed_instat_loc, self.md_instat_loc
        )

        self.expected_home_players_inmotio_instat = pd.DataFrame(
            {
                "id": [1, 2],
                "full_name": ["Player 1", "Player 2"],
                "shirt_num": [1, 2],
                "start_frame": [2, 2],
                "end_frame": [6, 6],
                "position": ["goalkeeper", "defender"],
                "starter": [True, True],
            }
        )

        self.expected_away_players_inmotio_instat = pd.DataFrame(
            {
                "id": [3, 4],
                "full_name": ["Player 11", "Player 12"],
                "shirt_num": [1, 2],
                "start_frame": [2, 2],
                "end_frame": [6, 6],
                "position": ["goalkeeper", "unspecified"],
                "starter": [True, True],
            }
        )

        self.expected_periods_inmotio_instat = pd.DataFrame(
            {
                "period_id": [1, 2, 3, 4, 5],
                "start_frame": [2, 5, MISSING_INT, MISSING_INT, MISSING_INT],
                "end_frame": [3, 6, MISSING_INT, MISSING_INT, MISSING_INT],
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
                    pd.to_datetime("2023-01-01 21:00:00").tz_localize(
                        "Europe/Amsterdam"
                    ),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                    pd.to_datetime("NaT"),
                ],
                "end_datetime_ed": [
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
            }
        )
        self.expected_game_inmotio_instat = Game(
            tracking_data=self.td_inmotio,
            event_data=EventData(self.ed_instat, provider="instat"),
            pitch_dimensions=self.md_inmotio.pitch_dimensions,
            periods=self.expected_periods_inmotio_instat,
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
            shot_events=pd.DataFrame(),
            pass_events=pd.DataFrame(),
            dribble_events=pd.DataFrame(),
            country=self.md_instat.country,
            _tracking_timestamp_is_precise=False,
            _periods_changed_playing_direction=[],
        )

        self.expected_game_inmotio_instat._periods_changed_playing_direction = [2]

        self.game_to_sync = get_game(
            tracking_data_loc="tests/test_data/sync/tracab_td_sync_test.dat",
            tracking_metadata_loc="tests/test_data/sync/tracab_metadata_sync_test.xml",
            tracking_data_provider="tracab",
            event_data_loc="tests/test_data/sync/opta_events_sync_test.xml",
            event_metadata_loc="tests/test_data/sync/opta_metadata_sync_test.xml",
            event_data_provider="opta",
            check_quality=False,
        )

        tracab_td = TrackingData(
            TD_TRACAB,
            provider="tracab",
            frame_rate=MD_TRACAB.frame_rate,
        )

        self.expected_game_tracab = Game(
            tracking_data=tracab_td,
            event_data=EventData(),
            pitch_dimensions=MD_TRACAB.pitch_dimensions,
            periods=MD_TRACAB.periods_frames,
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
            pass_events=pd.DataFrame(),
            shot_events=pd.DataFrame(),
            dribble_events=pd.DataFrame(),
            _tracking_timestamp_is_precise=True,
            _periods_changed_playing_direction=[],
        )

        self.sportec_event_loc = "tests/test_data/sportec_ed_test.xml"
        self.sportec_metadata_loc = "tests/test_data/sportec_metadata.xml"

        self.statsbomb_event_loc = "tests/test_data/statsbomb_event_test.json"
        self.statsbomb_match_loc = "tests/test_data/statsbomb_match_test.json"
        self.statsbomb_lineup_loc = "tests/test_data/statsbomb_lineup_test.json"

    def test_get_game_wrong_inputs(self):
        with self.assertRaises(ValueError):
            get_game(event_data_loc=self.ed_opta_loc, event_data_provider="opta")

        with self.assertRaises(ValueError):
            get_game(
                event_data_loc=self.ed_opta_loc, event_metadata_loc=self.md_opta_loc
            )

        with self.assertRaises(ValueError):
            get_game(event_metadata_loc=self.md_opta_loc)

        with self.assertRaises(ValueError):
            get_game(
                tracking_data_loc=self.td_tracab_loc,
                tracking_metadata_loc=self.md_tracab_loc,
            )

        with self.assertRaises(ValueError):
            get_game(
                tracking_data_loc=self.td_tracab_loc, tracking_data_provider="tracab"
            )

    def test_get_game_opta_tracab(self):
        game = get_game(
            tracking_data_loc=self.td_tracab_loc,
            tracking_metadata_loc=self.md_tracab_loc,
            event_data_loc=self.ed_opta_loc,
            event_metadata_loc=self.md_opta_loc,
            tracking_data_provider=self.td_provider,
            event_data_provider=self.ed_provider,
            check_quality=True,
        )
        assert game == self.expected_game_tracab_opta

        with self.assertWarns(DeprecationWarning):
            match = get_match(
                tracking_data_loc=self.td_tracab_loc,
                tracking_metadata_loc=self.md_tracab_loc,
                event_data_loc=self.ed_opta_loc,
                event_metadata_loc=self.md_opta_loc,
                tracking_data_provider=self.td_provider,
                event_data_provider=self.ed_provider,
                check_quality=True,
            )
        assert match == self.expected_game_tracab_opta

    def test_get_game_no_valid_input(self):
        with self.assertRaises(ValueError):
            get_game()

    def test_get_game_only_event_data(self):
        game = get_game(
            event_data_loc=self.ed_opta_loc,
            event_metadata_loc=self.md_opta_loc,
            event_data_provider="opta",
        )

        # expected_game pitch dimensions = [10, 10],
        # while opta pitch dimensions = [106, 68]
        x_cols = [col for col in game.event_data.columns if "_x" in col]
        y_cols = [col for col in game.event_data.columns if "_y" in col]
        game.event_data[x_cols] = game.event_data[x_cols] / 106.0 * 100
        game.event_data[y_cols] = game.event_data[y_cols] / 68.0 * 50
        game.pitch_dimensions = [100.0, 50.0]

        expected_game_opta = Game(
            tracking_data=TrackingData(),
            event_data=ED_OPTA,
            pitch_dimensions=MD_OPTA.pitch_dimensions,
            periods=MD_OPTA.periods_frames,
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
            shot_events=SHOT_EVENTS_OPTA,
            dribble_events=DRIBBLE_EVENTS_OPTA,
            pass_events=PASS_EVENTS_OPTA,
            _event_timestamp_is_precise=True,
        )

        assert game == expected_game_opta

    def test_get_game_only_tracking_data(self):
        game = get_game(
            tracking_data_loc=self.td_tracab_loc,
            tracking_metadata_loc=self.md_tracab_loc,
            tracking_data_provider="tracab",
            check_quality=False,
        )
        assert game == self.expected_game_tracab

    def test_get_game_wrong_provider(self):
        with self.assertRaises(ValueError):
            get_game(
                tracking_data_loc=self.td_tracab_loc,
                tracking_metadata_loc=self.md_tracab_loc,
                event_data_loc=self.ed_opta_loc,
                event_metadata_loc=self.md_opta_loc,
                tracking_data_provider=self.td_provider,
                event_data_provider="wrong",
                check_quality=False,
            )

        with self.assertRaises(ValueError):
            get_game(
                tracking_data_loc=self.td_tracab_loc,
                tracking_metadata_loc=self.md_tracab_loc,
                event_data_loc=self.ed_opta_loc,
                event_metadata_loc=self.md_opta_loc,
                tracking_data_provider="also wrong",
                event_data_provider=self.ed_provider,
                check_quality=False,
            )

    def test_get_game_inmotio_instat_unaligned_player_ids(self):
        game_instat_inmotio_unaligned_input = get_game(
            tracking_data_loc=self.td_inmotio_loc,
            tracking_metadata_loc=self.md_inmotio_loc,
            tracking_data_provider="inmotio",
            event_data_loc="tests/test_data/instat_ed_test_unaligned_player_ids.json",
            event_metadata_loc=self.md_instat_loc,
            event_data_provider="instat",
            check_quality=False,
        )
        expected_game = self.expected_game_inmotio_instat.copy()
        expected_game.home_players.loc[:, "id"] = [100, 200]
        expected_game.event_data.loc[[0, 1], "player_id"] = [200, 100]

        assert game_instat_inmotio_unaligned_input == expected_game

    def test_get_game_inmotio_instat(self):
        game_instat_inmotio = get_game(
            tracking_data_loc=self.td_inmotio_loc,
            tracking_metadata_loc=self.md_inmotio_loc,
            tracking_data_provider="inmotio",
            event_data_loc=self.ed_instat_loc,
            event_metadata_loc=self.md_instat_loc,
            event_data_provider="instat",
            check_quality=False,
        )

        assert game_instat_inmotio == self.expected_game_inmotio_instat

    def test_get_game_metrica_data(self):
        res = get_game(
            tracking_data_loc=self.td_metrica_loc,
            tracking_metadata_loc=self.md_metrica_loc,
            tracking_data_provider="metrica",
            event_data_loc=self.ed_metrica_loc,
            event_metadata_loc=self.md_metrica_loc,
            event_data_provider="metrica",
            check_quality=False,
        )
        assert res == self.expected_game_metrica

    def test_get_game_sportec(self):
        expected_game_sportec = Game(
            tracking_data=TrackingData(),
            event_data=SPORTEC_EVENT_DATA.copy(),
            pitch_dimensions=SPORTEC_METADATA_ED.pitch_dimensions,
            periods=SPORTEC_METADATA_ED.periods_frames,
            home_team_id=SPORTEC_METADATA_ED.home_team_id,
            home_formation=SPORTEC_METADATA_ED.home_formation,
            home_score=SPORTEC_METADATA_ED.home_score,
            home_team_name=SPORTEC_METADATA_ED.home_team_name,
            home_players=SPORTEC_METADATA_ED.home_players,
            away_team_id=SPORTEC_METADATA_ED.away_team_id,
            away_formation=SPORTEC_METADATA_ED.away_formation,
            away_score=SPORTEC_METADATA_ED.away_score,
            away_team_name=SPORTEC_METADATA_ED.away_team_name,
            away_players=SPORTEC_METADATA_ED.away_players,
            country=SPORTEC_METADATA_ED.country,
            _event_timestamp_is_precise=True,
            _periods_changed_playing_direction=None,
            pass_events=SPORTEC_PASS_EVENTS,
            shot_events=SPORTEC_SHOT_EVENTS,
            dribble_events=SPORTEC_DRIBBLE_EVENTS,
        )
        res = get_game(
            event_data_loc=self.sportec_event_loc,
            event_metadata_loc=self.sportec_metadata_loc,
            event_data_provider="sportec",
        )

        self.assertEqual(res, expected_game_sportec)

    def test_get_game_statsbomb(self):
        expected_game_statsbomb = Game(
            tracking_data=TrackingData(),
            event_data=ED_STATSBOMB,
            pitch_dimensions=MD_STATSBOMB.pitch_dimensions,
            periods=MD_STATSBOMB.periods_frames,
            home_team_id=MD_STATSBOMB.home_team_id,
            home_formation=MD_STATSBOMB.home_formation,
            home_score=MD_STATSBOMB.home_score,
            home_team_name=MD_STATSBOMB.home_team_name,
            home_players=MD_STATSBOMB.home_players,
            away_team_id=MD_STATSBOMB.away_team_id,
            away_formation=MD_STATSBOMB.away_formation,
            away_score=MD_STATSBOMB.away_score,
            away_team_name=MD_STATSBOMB.away_team_name,
            away_players=MD_STATSBOMB.away_players,
            country=MD_STATSBOMB.country,
            _tracking_timestamp_is_precise=False,
            _periods_changed_playing_direction=None,
            shot_events=SHOT_EVENTS_STATSBOMB,
            pass_events=PASS_EVENTS_STATSBOMB,
            dribble_events=DRIBBLE_EVENTS_STATSBOMB,
        )

        res = get_game(
            event_data_loc=self.statsbomb_event_loc,
            event_match_loc=self.statsbomb_match_loc,
            event_lineup_loc=self.statsbomb_lineup_loc,
            event_data_provider="statsbomb",
        )
        self.assertEqual(res, expected_game_statsbomb)

    def test_get_game_statsbomb_input_missing(self):
        with self.assertRaises(ValueError):
            get_game(
                event_data_loc=self.statsbomb_event_loc,
                event_match_loc=self.statsbomb_match_loc,
                event_data_provider="statsbomb",
            )

    @patch("databallpy.utils.get_game.load_sportec_open_event_data")
    @patch("databallpy.utils.get_game.load_sportec_open_tracking_data")
    @patch("databallpy.utils.get_game.os.remove")
    def test_get_open_game_sportec(
        self, mock_os_remove, mock_tracking_data, mock_event_data
    ):
        mock_tracking_data.return_value = (TRACAB_SPORTEC_XML_TD, SPORTEC_METADATA_TD)
        mock_event_data.return_value = (
            SPORTEC_EVENT_DATA,
            SPORTEC_METADATA_ED,
            SPORTEC_DATABALLPY_EVENTS,
        )
        mock_os_remove.return_value = "removed"
        game = get_open_game(use_cache=False)

        td = TrackingData(
            TRACAB_SPORTEC_XML_TD,
            provider="sportec",
            frame_rate=SPORTEC_METADATA_TD.frame_rate,
        )
        expected_game_sportec = Game(
            tracking_data=td,
            event_data=SPORTEC_EVENT_DATA.copy(),
            pitch_dimensions=SPORTEC_METADATA_ED.pitch_dimensions,
            periods=pd.merge(
                SPORTEC_METADATA_TD.periods_frames,
                SPORTEC_METADATA_ED.periods_frames[
                    ["start_datetime_ed", "end_datetime_ed"]
                ],
                left_index=True,
                right_index=True,
                how="outer",
            ),
            home_team_id=SPORTEC_METADATA_ED.home_team_id,
            home_formation=SPORTEC_METADATA_ED.home_formation,
            home_score=SPORTEC_METADATA_ED.home_score,
            home_team_name=SPORTEC_METADATA_ED.home_team_name,
            home_players=SPORTEC_METADATA_ED.home_players,
            away_team_id=SPORTEC_METADATA_ED.away_team_id,
            away_formation=SPORTEC_METADATA_ED.away_formation,
            away_score=SPORTEC_METADATA_ED.away_score,
            away_team_name=SPORTEC_METADATA_ED.away_team_name,
            away_players=SPORTEC_METADATA_ED.away_players,
            country=SPORTEC_METADATA_ED.country,
            _event_timestamp_is_precise=True,
            _tracking_timestamp_is_precise=True,
            _periods_changed_playing_direction=[1],
            pass_events=SPORTEC_PASS_EVENTS,
            shot_events=SPORTEC_SHOT_EVENTS,
            dribble_events=SPORTEC_DRIBBLE_EVENTS,
            allow_synchronise_tracking_and_event_data=True,
        )

        self.assertEqual(game, expected_game_sportec)
        self.assertEqual(mock_os_remove.call_count, 2)

        with self.assertWarns(DeprecationWarning):
            match = get_open_match()
        self.assertEqual(match, expected_game_sportec)

    @patch(
        "requests.get",
        side_effect=[
            Mock(text=TD_METRICA_RAW),
            Mock(text=MD_METRICA_RAW),
            Mock(text=ED_METRICA_RAW),
            Mock(text=MD_METRICA_RAW),
        ],
    )
    def test_get_open_game(self, _):
        game = get_open_game(provider="metrica", use_cache=False)
        pd.testing.assert_frame_equal(game.periods, self.expected_game_metrica.periods)
        expected_game = self.expected_game_metrica.copy()
        expected_game._periods_changed_playing_direction = []
        expected_game.allow_synchronise_tracking_and_event_data = True
        assert game == expected_game

        with self.assertRaises(ValueError):
            get_open_game(provider="wrong")

    def test_get_saved_game(self):
        expected_game = self.expected_game_metrica
        expected_game.save_game(
            name="test_game", path="tests/test_data", allow_overwrite=True
        )
        saved_game = get_saved_game(name="test_game", path="tests/test_data")
        assert expected_game == saved_game
        assert saved_game != self.expected_game_tracab_opta

        with self.assertWarns(DeprecationWarning):
            saved_match = get_saved_match(name="test_game", path="tests/test_data")
        assert saved_game == saved_match
        for file in os.listdir(os.path.join("tests", "test_data", "test_game")):
            os.remove(os.path.join("tests", "test_data", "test_game", file))
        os.rmdir(os.path.join("tests", "test_data", "test_game"))

    def test_get_saved_game_errors(self):
        with self.assertRaises(ValueError):
            get_saved_game(name="test_game2", path="tests/test_data")

    def test_get_game_scisports(self):
        res_game = get_game(
            event_data_loc="tests/test_data/scisports_test.json",
            event_data_provider="scisports",
            _check_game_class_=False,
        )
        pd.testing.assert_frame_equal(res_game.event_data, ED_SCISPORTS)
        pd.testing.assert_frame_equal(res_game.periods, MD_SCISPORTS.periods_frames)
        pd.testing.assert_frame_equal(res_game.home_players, MD_SCISPORTS.home_players)
        pd.testing.assert_frame_equal(res_game.away_players, MD_SCISPORTS.away_players)

        self.assertTrue(len(res_game.shot_events) == 2)
        self.assertTrue(len(res_game.pass_events) == 3)
        self.assertTrue(len(res_game.dribble_events) == 1)
