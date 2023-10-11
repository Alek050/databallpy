import datetime as dt
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from databallpy.get_match import get_match, get_open_match, get_saved_match
from databallpy.load_data.event_data.instat import load_instat_event_data
from databallpy.load_data.event_data.metrica_event_data import load_metrica_event_data
from databallpy.load_data.event_data.opta import load_opta_event_data
from databallpy.load_data.tracking_data.inmotio import load_inmotio_tracking_data
from databallpy.load_data.tracking_data.metrica_tracking_data import (
    load_metrica_tracking_data,
)
from databallpy.load_data.tracking_data.tracab import load_tracab_tracking_data
from databallpy.match import Match
from databallpy.utils.utils import MISSING_INT
from databallpy.warnings import DataBallPyWarning
from expected_outcomes import (
    DRIBBLE_EVENTS_METRICA,
    DRIBBLE_EVENTS_OPTA,
    DRIBBLE_EVENTS_OPTA_TRACAB,
    ED_OPTA,
    MD_OPTA,
    MD_TRACAB,
    PASS_EVENTS_METRICA,
    PASS_EVENTS_OPTA,
    PASS_EVENTS_OPTA_TRACAB,
    SHOT_EVENTS_METRICA,
    SHOT_EVENTS_OPTA,
    SHOT_EVENTS_OPTA_TRACAB,
    TD_TRACAB,
)
from tests.mocks import ED_METRICA_RAW, MD_METRICA_RAW, TD_METRICA_RAW


class TestGetMatch(unittest.TestCase):
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
        self.ed_opta, self.md_opta, self.dbp_events = load_opta_event_data(
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
                "starter": [True, False],
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
                "starter": [True, False],
            }
        )

        self.td_tracab["period"] = [1, 1, MISSING_INT, 2, 2]

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
            shot_events=SHOT_EVENTS_OPTA_TRACAB,
            dribble_events=DRIBBLE_EVENTS_OPTA_TRACAB,
            pass_events=PASS_EVENTS_OPTA_TRACAB,
            _tracking_timestamp_is_precise=True,
            _event_timestamp_is_precise=True,
        )

        self.td_metrica_loc = "tests/test_data/metrica_tracking_data_test.txt"
        self.md_metrica_loc = "tests/test_data/metrica_metadata_test.xml"
        self.ed_metrica_loc = "tests/test_data/metrica_event_data_test.json"
        self.td_metrica, self.md_metrica = load_metrica_tracking_data(
            self.td_metrica_loc, self.md_metrica_loc
        )
        self.td_metrica["period"] = [1, 1, 1, 2, 2, 2]
        self.ed_metrica, md_metrica_ed, dbe_metrica = load_metrica_event_data(
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
            pass_events=PASS_EVENTS_METRICA,
            shot_events=SHOT_EVENTS_METRICA,
            dribble_events=DRIBBLE_EVENTS_METRICA,
            _tracking_timestamp_is_precise=True,
            _event_timestamp_is_precise=True,
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
            _tracking_timestamp_is_precise=False,
        )
        self.expected_match_inmotio_instat.event_data["team_id"] = [
            "T-0001",
            "T-0001",
            "T-0002",
            None,
        ]

        self.match_to_sync = get_match(
            tracking_data_loc="tests/test_data/sync/tracab_td_sync_test.dat",
            tracking_metadata_loc="tests/test_data/sync/tracab_metadata_sync_test.xml",
            tracking_data_provider="tracab",
            event_data_loc="tests/test_data/sync/opta_events_sync_test.xml",
            event_metadata_loc="tests/test_data/sync/opta_metadata_sync_test.xml",
            event_data_provider="opta",
            check_quality=False,
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
            _tracking_timestamp_is_precise=True,
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
            shot_events=SHOT_EVENTS_OPTA,
            dribble_events=DRIBBLE_EVENTS_OPTA,
            pass_events=PASS_EVENTS_OPTA,
            _event_timestamp_is_precise=True,
        )

    def test_get_match_wrong_inputs(self):
        with self.assertRaises(ValueError):
            get_match(event_data_loc=self.ed_opta_loc, event_data_provider="opta")

        with self.assertRaises(ValueError):
            get_match(
                event_data_loc=self.ed_opta_loc, event_metadata_loc=self.md_opta_loc
            )

        with self.assertRaises(ValueError):
            get_match(event_metadata_loc=self.md_opta_loc)

        with self.assertRaises(ValueError):
            get_match(
                tracking_data_loc=self.td_tracab_loc,
                tracking_metadata_loc=self.md_tracab_loc,
            )

        with self.assertRaises(ValueError):
            get_match(
                tracking_data_loc=self.td_tracab_loc, tracking_data_provider="tracab"
            )

    def test_get_match_opta_tracab(self):
        match = get_match(
            tracking_data_loc=self.td_tracab_loc,
            tracking_metadata_loc=self.md_tracab_loc,
            event_data_loc=self.ed_opta_loc,
            event_metadata_loc=self.md_opta_loc,
            tracking_data_provider=self.td_provider,
            event_data_provider=self.ed_provider,
            check_quality=False,
        )
        assert match == self.expected_match_tracab_opta

    def test_get_match_no_valid_input(self):
        with self.assertRaises(ValueError):
            get_match()

    def test_get_match_only_event_data(self):
        match = get_match(
            event_data_loc=self.ed_opta_loc,
            event_metadata_loc=self.md_opta_loc,
            event_data_provider="opta",
        )

        # expected_match pitch dimensions = [10, 10],
        # while opta pitch dimensions = [106, 68]
        x_cols = [col for col in match.event_data.columns if "_x" in col]
        y_cols = [col for col in match.event_data.columns if "_y" in col]
        match.event_data[x_cols] = match.event_data[x_cols] / 106.0 * 10
        match.event_data[y_cols] = match.event_data[y_cols] / 68.0 * 10
        match.pitch_dimensions = [10.0, 10.0]

        assert match == self.expected_match_opta

    def test_get_match_only_tracking_data(self):
        match = get_match(
            tracking_data_loc=self.td_tracab_loc,
            tracking_metadata_loc=self.md_tracab_loc,
            tracking_data_provider="tracab",
            check_quality=False,
        )
        assert match == self.expected_match_tracab

    def test_get_match_wrong_provider(self):
        with self.assertRaises(AssertionError):
            get_match(
                tracking_data_loc=self.td_tracab_loc,
                tracking_metadata_loc=self.md_tracab_loc,
                event_data_loc=self.ed_opta_loc,
                event_metadata_loc=self.md_opta_loc,
                tracking_data_provider=self.td_provider,
                event_data_provider="wrong",
                check_quality=False,
            )

        with self.assertRaises(AssertionError):
            get_match(
                tracking_data_loc=self.td_tracab_loc,
                tracking_metadata_loc=self.md_tracab_loc,
                event_data_loc=self.ed_opta_loc,
                event_metadata_loc=self.md_opta_loc,
                tracking_data_provider="also wrong",
                event_data_provider=self.ed_provider,
                check_quality=False,
            )

    def test_get_match_inmotio_instat_unaligned_player_ids(self):
        match_instat_inmotio_unaligned_input = get_match(
            tracking_data_loc=self.td_inmotio_loc,
            tracking_metadata_loc=self.md_inmotio_loc,
            tracking_data_provider="inmotio",
            event_data_loc="tests/test_data/instat_ed_test_unaligned_player_ids.json",
            event_metadata_loc=self.md_instat_loc,
            event_data_provider="instat",
            check_quality=False,
        )
        assert (
            match_instat_inmotio_unaligned_input == self.expected_match_inmotio_instat
        )

    def test_get_match_inmotio_instat(self):
        match_instat_inmotio = get_match(
            tracking_data_loc=self.td_inmotio_loc,
            tracking_metadata_loc=self.md_inmotio_loc,
            tracking_data_provider="inmotio",
            event_data_loc=self.ed_instat_loc,
            event_metadata_loc=self.md_instat_loc,
            event_data_provider="instat",
            check_quality=False,
        )

        assert match_instat_inmotio == self.expected_match_inmotio_instat

    def test_get_match_metrica_data(self):
        res = get_match(
            tracking_data_loc=self.td_metrica_loc,
            tracking_metadata_loc=self.md_metrica_loc,
            tracking_data_provider="metrica",
            event_data_loc=self.ed_metrica_loc,
            event_metadata_loc=self.md_metrica_loc,
            event_data_provider="metrica",
            check_quality=False,
        )
        assert res == self.expected_match_metrica

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
        expected_match = self.expected_match_metrica.copy()
        expected_match.allow_synchronise_tracking_and_event_data = True
        assert match == expected_match

    def test_get_saved_match(self):
        expected_match = self.expected_match_metrica
        expected_match.save_match(name="test_match", path="tests/test_data")
        saved_match = get_saved_match(name="test_match", path="tests/test_data")
        assert expected_match == saved_match
        assert saved_match != self.expected_match_tracab_opta

    def test_get_match_call_quality_check(self):
        # does not check functionality since the tracking data is not valid
        with self.assertRaises(ZeroDivisionError), self.assertWarns(DataBallPyWarning):
            get_match(
                tracking_data_loc=self.td_tracab_loc,
                tracking_metadata_loc=self.md_tracab_loc,
                event_data_loc=self.ed_opta_loc,
                event_metadata_loc=self.md_opta_loc,
                tracking_data_provider=self.td_provider,
                event_data_provider=self.ed_provider,
                check_quality=True,
            )

        with self.assertRaises(ZeroDivisionError), self.assertWarns(DataBallPyWarning):
            get_match(
                tracking_data_loc=self.td_tracab_loc,
                tracking_metadata_loc=self.md_tracab_loc,
                tracking_data_provider=self.td_provider,
                check_quality=True,
            )
