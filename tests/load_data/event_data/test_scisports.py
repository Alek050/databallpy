import unittest

import numpy as np
import pandas as pd

from databallpy.load_data.event_data.dribble_event import DribbleEvent
from databallpy.load_data.event_data.pass_event import PassEvent
from databallpy.load_data.event_data.scisports import (
    _add_team_possessions_to_tracking_data,
    _find_event,
    _find_location,
    _find_set_piece,
    _get_all_events,
    _get_dribble_instances,
    _get_pass_instances,
    _get_shot_instances,
    _load_event_data,
    _update_scisports_event_data_with_metadata,
    load_scisports_event_data,
)
from databallpy.load_data.event_data.shot_event import ShotEvent
from databallpy.load_data.metadata import Metadata
from databallpy.utils.utils import MISSING_INT
from databallpy.warnings import DataBallPyWarning


class TestScisports(unittest.TestCase):
    def setUp(self):
        self.scisports_event_loc = "tests/test_data/scisports_test.xml"
        self.expected_possessions = pd.DataFrame(
            {
                "team": ["Team 1", "Team 1", "Team 2"],
                "start_seconds": [0.0, 1, 5],
                "end_seconds": [3.0, 6, 9],
                "possession_type": ["free_kick", "free_kick", "open_play"],
            }
        )
        self.expected_event_data = (
            pd.DataFrame(
                {
                    "seconds": [5.75, 3.0, 5.0, 6.5, 6.5, 8.5, 9.5],
                    "team": [
                        "Team 1",
                        "Team 1",
                        "Team 2",
                        "Team 1",
                        "Team 2",
                        "Team 2",
                        "Team 2",
                    ],
                    "player_name": [
                        "F. Team1Player11",
                        "H. Team1Player1",
                        "Z. Team2Player8",
                        "J. Team1Player2",
                        "J. Team2Player3",
                        "J. Team2Player3",
                        "J. Team2Player3",
                    ],
                    "jersey_number": [11, 1, 8, 2, 3, 3, 3],
                    "event_id": [2, 0, 1, 3, 4, 5, 6],
                    "scisports_event": [
                        "Deep Run",
                        "GK Pass",
                        "Cross",
                        "Pass",
                        "Unknown",
                        "Pass",
                        "Shot",
                    ],
                    "event_location": [
                        None,
                        "Unknown",
                        "Unknown",
                        "Own Half",
                        "Att. Half",
                        "Att. Half",
                        "Att. Half",
                    ],
                    "set_piece": [
                        None,
                        "Open Play",
                        "Free Kick",
                        "Free Kick",
                        "Free Kick",
                        "Throw In",
                        "Penalty",
                    ],
                    "outcome": [None, 1, None, None, None, None, None],
                    "run_type": ["High Run", None, None, None, None, None, None],
                    "period_id": [2, 1, 2, 2, 2, 2, 2],
                    "databallpy_event": [
                        None,
                        "pass",
                        "pass",
                        "pass",
                        None,
                        "pass",
                        "shot",
                    ],
                }
            )
            .sort_values(by="seconds")
            .reset_index(drop=True)
        )

        self.processed_event_data = self.expected_event_data.copy()
        self.processed_event_data["minutes"] = [3, 4, 2, 6, 7, 12, 45]
        self.processed_event_data["datetime"] = [
            "2021-01-02 00:10:10",
            "2021-01-02 00:01:10",
            "2021-01-03 00:12:10",
            "2021-01-05 06:00:10",
            "2021-01-01 00:08:10",
            "2021-01-01 02:00:10",
            "2021-01-01 00:00:10",
        ]
        self.processed_event_data["datetime"] = pd.to_datetime(
            self.processed_event_data["datetime"]
        ).dt.tz_localize("Europe/Amsterdam")
        self.processed_event_data["start_x"] = [np.nan] * 7
        self.processed_event_data["start_y"] = [np.nan] * 7
        self.processed_event_data.rename(columns={"team": "team_id"}, inplace=True)
        self.processed_event_data["player_id"] = [101, 108, 111, 102, 103, 103, 103]

    def test_load_scisports_event_data(self):
        with self.assertRaises(TypeError):
            load_scisports_event_data(1)

        with self.assertWarns(DataBallPyWarning):
            res_ed, res_pos = load_scisports_event_data(self.scisports_event_loc)

        pd.testing.assert_frame_equal(res_ed, self.expected_event_data)
        pd.testing.assert_frame_equal(res_pos, self.expected_possessions)

    def test_load_event_data(self):
        with self.assertWarns(DataBallPyWarning):
            res_ed, res_pos = _load_event_data(self.scisports_event_loc)
        pd.testing.assert_frame_equal(
            res_ed, self.expected_event_data.drop(columns=["databallpy_event"])
        )
        pd.testing.assert_frame_equal(res_pos, self.expected_possessions)

    def test_get_all_events(self):
        expected_all_events = (
            pd.DataFrame(
                {
                    "seconds": [5.75, 3.0, 3.0, 5.0, 6.5, 6.5, 8.5, 9.5, 9.5],
                    "team": [
                        "Team 1",
                        "Team 1",
                        "Team 1",
                        "Team 2",
                        "Team 1",
                        "Team 2",
                        "Team 2",
                        "Team 2",
                        "Team 2",
                    ],
                    "jersey_number": [11, 1, 1, 8, 2, 3, 3, 3, 3],
                    "player_name": [
                        "F. Team1Player11",
                        "H. Team1Player1",
                        "H. Team1Player1",
                        "Z. Team2Player8",
                        "J. Team1Player2",
                        "J. Team2Player3",
                        "J. Team2Player3",
                        "J. Team2Player3",
                        "J. Team2Player3",
                    ],
                    "event_type": [
                        "Physical",
                        "GK Pass",
                        "Pass (Successful)",
                        "Set Piece",
                        "Set Piece",
                        "Set Piece",
                        "Set Piece",
                        "Set Piece",
                        None,
                    ],
                    "event_name": [
                        "Deep Run",
                        "Open Play",
                        "Open Play",
                        "Crossed Free Kick",
                        "Free Kick",
                        "Free Kick",
                        "Throw In",
                        "Penalty",
                        "Unknown Event",
                    ],
                    "identifier": [
                        None,
                        None,
                        None,
                        None,
                        "Own Half",
                        "Att. Half",
                        "Att. Half",
                        "Att. Half",
                        None,
                    ],
                    "run_type": [
                        "High Run",
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ],
                }
            )
            .sort_values(by="seconds")
            .reset_index(drop=True)
        )

        expected_match_periods = [0.5, 3.5, 5.5, 11.5]

        res_pos, res_events, res_periods = _get_all_events(self.scisports_event_loc)
        assert res_periods == expected_match_periods
        pd.testing.assert_frame_equal(res_pos, self.expected_possessions)
        pd.testing.assert_frame_equal(res_events, expected_all_events)

    def test_find_set_piece(self):
        identifiers_1 = ["Pass"]
        identifiers_2 = ["Kick Off", "Pass"]
        identifiers_3 = ["Set Piece", "Free Kick", "Pass"]

        assert _find_set_piece(identifiers_1) == "Open Play"
        assert _find_set_piece(identifiers_2) == "Kick Off"
        assert _find_set_piece(identifiers_3) == "Free Kick"

    def test_find_location(self):
        identifiers_1 = ["Pass"]
        identifiers_2 = ["Kick Off", "Pass", "Att. Half"]
        identifiers_3 = ["Set Piece", "Free Kick", "Pass", "Att. Half", "Score Box"]

        assert _find_location(identifiers_1) == "Unknown"
        assert _find_location(identifiers_2) == "Att. Half"
        assert _find_location(identifiers_3) == "Att. Half - Score Box"

    def test_find_event_shots(self):
        identifiers_1 = ["Goal", "Shot", "on Target"]
        identifiers_2 = ["Pass", "Own Goal"]
        identifiers_3 = ["Shot", "Wide"]

        assert _find_event(identifiers_1) == ("Goal", 1)
        assert _find_event(identifiers_2) == ("Own Goal", 1)
        assert _find_event(identifiers_3) == ("Shot Wide", 0)

    def test_find_event_pass(self):
        identifiers_1 = ["Assist", "Pass (Successful)", "GK Throw", "Att. Half"]
        identifiers_2 = ["Pass", "Cross"]
        identifiers_3 = ["Own Half", "Free Kick", "Pass", "Possession Loss"]

        assert _find_event(identifiers_1) == ("Assist", 1)
        assert _find_event(identifiers_2) == ("Cross", None)
        assert _find_event(identifiers_3) == ("Pass", 0)

    def test_find_event_duel(self):
        identifiers_1 = ["Take On", "Possession Loss"]
        identifiers_2 = ["Take On Faced", "Duel Defensive"]

        assert _find_event(identifiers_1) == ("Take On", 0)
        assert _find_event(identifiers_2) == ("Take On Faced", None)

    def test_find_event_defensive(self):
        identifiers_1 = ["GK Save"]
        identifiers_2 = ["GK Save", "GK Punch", "Defensive"]

        assert _find_event(identifiers_1) == ("GK Save", None)
        assert _find_event(identifiers_2) == ("GK Punch", None)

    def test_find_event_others(self):
        identifiers_1 = ["Penetration"]
        identifiers_2 = ["Unknown", "Possession Loss"]
        identifiers_3 = ["2nd Ball"]
        identifiers_4 = ["Substitute"]
        identifiers_5 = ["new_kind_of_event"]

        assert _find_event(identifiers_1) == ("Penetration", None)
        assert _find_event(identifiers_2) == ("Possession Loss", 1)
        assert _find_event(identifiers_3) == ("2nd Ball", None)
        assert _find_event(identifiers_4) == ("Substitute", None)
        assert _find_event(identifiers_5) == ("Unknown", None)

    def test_get_shot_instances(self):
        expected_shots = {
            6: ShotEvent(
                event_id=6,
                period_id=2,
                minutes=45,
                seconds=9,
                datetime=pd.to_datetime("2021-01-01 00:00:10").tz_localize(
                    "Europe/Amsterdam"
                ),
                start_x=np.nan,
                start_y=np.nan,
                y_target=np.nan,
                z_target=np.nan,
                team_id="Team 2",
                player_id=103,
                shot_outcome="not_specified",
                type_of_play=None,
                body_part=None,
                created_oppertunity=None,
            )
        }

        res_shots = _get_shot_instances(self.processed_event_data)
        assert res_shots == expected_shots

    def test_get_pass_instances(self):
        expected_passes = {
            0: PassEvent(
                event_id=0,
                period_id=1,
                minutes=3,
                seconds=3,
                datetime=pd.to_datetime("2021-01-02 00:10:10").tz_localize(
                    "Europe/Amsterdam"
                ),
                start_x=np.nan,
                start_y=np.nan,
                team_id="Team 1",
                outcome="successful",
                player_id=101,
                end_x=np.nan,
                end_y=np.nan,
                pass_type="not_specified",
                set_piece="no_set_piece",
            )
        }

        res_passes = _get_pass_instances(self.processed_event_data.loc[[0, 4, 6]])
        assert res_passes == expected_passes

    def test_get_dribble_instance(self):
        event_data = self.processed_event_data.copy()
        event_data.loc[3, "databallpy_event"] = "dribble"
        expected_dribbles = {
            3: DribbleEvent(
                event_id=3,
                period_id=2,
                minutes=6,
                seconds=6,
                datetime=pd.to_datetime("2021-01-05 06:00:10").tz_localize(
                    "Europe/Amsterdam"
                ),
                start_x=np.nan,
                start_y=np.nan,
                team_id="Team 1",
                outcome=None,
                player_id=102,
                related_event_id=MISSING_INT,
                duel_type=None,
                has_opponent=True,
            )
        }

        res_dribbles = _get_dribble_instances(event_data)
        assert res_dribbles == expected_dribbles

    def test_update_scisports_event_data_with_metadata(self):
        period_frames = pd.DataFrame(
            {
                "period_id": [1, 2, 3, 4, 5],
                "start_datetime_ed": [
                    "2021-01-05 06:00:00",
                    "2021-01-05 07:00:00",
                    "NaT",
                    "NaT",
                    "NaT",
                ],
                "end_datetime_ed": [
                    "2021-01-05 06:45:00",
                    "2021-01-05 07:45:00",
                    "NaT",
                    "NaT",
                    "NaT",
                ],
            }
        )
        period_frames["start_datetime_ed"] = pd.to_datetime(
            period_frames["start_datetime_ed"]
        ).dt.tz_localize("Europe/Amsterdam")
        period_frames["end_datetime_ed"] = pd.to_datetime(
            period_frames["end_datetime_ed"]
        ).dt.tz_localize("Europe/Amsterdam")

        home_players = pd.DataFrame(
            {
                "id": [211, 202, 201],
                "full_name": [
                    "Frank Team1Player11",
                    "Jonas Team1Player2",
                    "Hendrik the big one Team1Player1",
                ],
                "shirt_num": [11, 2, 1],
            }
        )

        away_players = pd.DataFrame(
            {
                "id": [208, 203],
                "full_name": ["Zacharias Team2Player8", "Johan Team2Player3"],
                "shirt_num": [8, 3],
            }
        )

        metadata_ed = Metadata(
            match_id=12345,
            pitch_dimensions=[106.0, 68.0],
            periods_frames=period_frames,
            frame_rate=1,
            home_team_id=11111,
            home_team_name="Team 1",
            home_players=home_players,
            home_score=MISSING_INT,
            home_formation="1442",
            away_team_id=22222,
            away_team_name="Team 2",
            away_players=away_players,
            away_score=MISSING_INT,
            away_formation="1433",
            country="Netherlands",
        )

        expected_event_data = self.processed_event_data.copy()
        expected_event_data = expected_event_data.reindex(
            columns=[
                "event_id",
                "databallpy_event",
                "period_id",
                "minutes",
                "seconds",
                "player_id",
                "team_id",
                "outcome",
                "start_x",
                "start_y",
                "datetime",
                "scisports_event",
                "event_location",
                "run_type",
                "player_name",
                "set_piece",
            ]
        )
        expected_event_data.loc[:5, "minutes"] = 0
        expected_event_data["team_id"] = [
            11111.0,
            22222,
            11111,
            11111,
            22222,
            22222,
            22222,
        ]
        expected_event_data["player_id"] = [
            201,
            208,
            211,
            202,
            203,
            203,
            203,
        ]

        names_map = {
            "F. Team1Player11": "Frank Team1Player11",
            "H. Team1Player1": "Hendrik the big one Team1Player1",
            "Z. Team2Player8": "Zacharias Team2Player8",
            "J. Team1Player2": "Jonas Team1Player2",
            "J. Team2Player3": "Johan Team2Player3",
        }
        expected_event_data["player_name"] = expected_event_data["player_name"].map(
            names_map
        )
        expected_event_data["datetime"] = pd.to_datetime(
            [
                "2021-01-05 06:00:00.000000+01:00",
                "2021-01-05 06:00:02.000000+01:00",
                "2021-01-05 06:00:02.750000+01:00",
                "2021-01-05 06:00:03.500000+01:00",
                "2021-01-05 06:00:03.500000+01:00",
                "2021-01-05 06:00:05.500000+01:00",
                "2021-01-05 06:45:06.500000+01:00",
            ],
            format="%Y-%m-%d %H:%M:%S.%f%z",
        ).tz_convert("Europe/Amsterdam")

        input_data = self.expected_event_data.copy()
        input_data.loc[6, "seconds"] = input_data.loc[6, "seconds"] + 45 * 60.0
        res_ed = _update_scisports_event_data_with_metadata(input_data, metadata_ed)

        pd.testing.assert_frame_equal(res_ed, expected_event_data)

        metadata_td = metadata_ed.copy()
        metadata_td.periods_frames = metadata_td.periods_frames.rename(
            columns={
                "start_datetime_ed": "start_datetime_td",
                "end_datetime_ed": "end_datetime_td",
            }
        )

        res_td = _update_scisports_event_data_with_metadata(input_data, metadata_td)
        pd.testing.assert_frame_equal(res_td, expected_event_data)

    def test_add_team_possessions_to_tracking_data_all_dead(self):
        tracking_data = pd.DataFrame(
            {
                "ball_status": ["dead"] * 10,
                "ball_possession": [None] * 10,
            }
        )
        possesions = pd.DataFrame(
            {
                "team": ["Team 1", "Team 1", "Team 2"],
                "start_seconds": [0.0, 1, 4],
                "end_seconds": [3.0, 6, 9],
            }
        )

        expected_tracking_data = tracking_data.copy()
        expected_tracking_data["ball_possession"] = [
            "home",
            "home",
            "home",
            "home",
            "home",
            "away",
            "away",
            "away",
            "away",
            "away",
        ]

        res = _add_team_possessions_to_tracking_data(
            tracking_data, possesions, frame_rate=1, home_team_name="Team 1"
        )
        pd.testing.assert_frame_equal(res, expected_tracking_data)

        # add where the ball is alive, thus the start of the match

    def test_add_team_possessions_to_tracking_data_alive(self):
        tracking_data = pd.DataFrame(
            {
                "ball_status": [
                    "dead",
                    "alive",
                    "alive",
                    "alive",
                    "alive",
                    "alive",
                    "alive",
                    "alive",
                    "alive",
                    "alive",
                ],
                "ball_possession": [None] * 10,
            }
        )
        possesions = pd.DataFrame(
            {
                "team": ["Team 1", "Team 1", "Team 2"],
                "start_seconds": [0.0, 1, 4],
                "end_seconds": [3.0, 6, 9],
            }
        )

        expected_tracking_data = tracking_data.copy()
        expected_tracking_data["ball_possession"] = [
            None,
            "home",
            "home",
            "home",
            "home",
            "home",
            "away",
            "away",
            "away",
            "away",
        ]

        res = _add_team_possessions_to_tracking_data(
            tracking_data, possesions, frame_rate=1, home_team_name="Team 1"
        )
        pd.testing.assert_frame_equal(res, expected_tracking_data)
