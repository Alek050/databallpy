import json
import unittest

import numpy as np
import pandas as pd

from databallpy.data_parsers.event_data_parsers.scisports_parser import (
    _get_dribble_event,
    _get_pass_event,
    _get_periods_frames,
    _get_players,
    _get_shot_event,
    _load_event_data,
    _load_metadata,
    load_scisports_event_data,
)
from databallpy.events import DribbleEvent, PassEvent, ShotEvent
from databallpy.utils.utils import MISSING_INT
from tests.expected_outcomes import ED_SCISPORTS, MD_SCISPORTS


class TestSciSportsParser(unittest.TestCase):
    def setUp(self):
        self.json_loc = "tests/test_data/scisports_test.json"
        self.expected_home_players = MD_SCISPORTS.home_players.copy()
        self.expected_away_players = MD_SCISPORTS.away_players.copy()
        self.expected_periods = MD_SCISPORTS.periods_frames.copy()
        self.expected_metadata = MD_SCISPORTS.copy()
        with open(self.json_loc, "r") as f:
            self.json_data = json.load(f)
        self.expected_event_data = ED_SCISPORTS.copy()

    def test_load_scisports_event_data(self):
        res_ed, res_md, res_dbe = load_scisports_event_data(self.json_loc)
        pd.testing.assert_frame_equal(res_ed, self.expected_event_data)
        self.assertEqual(res_md, self.expected_metadata)

        for key, event in {
            **res_dbe["shot_events"],
            **res_dbe["pass_events"],
            **res_dbe["dribble_events"],
        }.items():
            row = self.expected_event_data.loc[
                self.expected_event_data["event_id"] == key
            ].iloc[0]
            databallpy_event = row["databallpy_event"]

            self.assertAlmostEqual(event.event_id, key)
            self.assertAlmostEqual(event.period_id, row["period_id"])
            self.assertAlmostEqual(event.minutes, row["minutes"])
            self.assertAlmostEqual(event.seconds, row["seconds"])
            self.assertAlmostEqual(event.datetime, row["datetime"])
            self.assertAlmostEqual(event.team_id, row["team_id"])
            self.assertEqual(
                event.team_side, "home" if row["team_id"] == 100 else "away"
            )
            self.assertAlmostEqual(event.pitch_size, (106.0, 68.0))

            if databallpy_event == "shot":
                self.assertIsInstance(event, ShotEvent)
                self.assertEqual(event.start_x, row["start_x"])
            elif databallpy_event == "pass":
                self.assertIsInstance(event, PassEvent)
                self.assertTrue(event.end_y in [2.01, -20.8, 32.5])
            elif databallpy_event == "dribble":
                self.assertIsInstance(event, DribbleEvent)
                self.assertEqual(event.start_y, row["start_y"])

    def test_load_scisports_event_data_errors(self):
        with self.assertRaises(FileNotFoundError):
            load_scisports_event_data("non_existent_file.json")

        with self.assertRaises(ValueError):
            load_scisports_event_data(self.json_loc, pitch_dimensions={106.0, 68.0})

    def test_load_metadata(self):
        result = _load_metadata(self.json_loc, (106.0, 68.0))
        self.assertEqual(result, self.expected_metadata)

    def test_get_players(self):
        result = _get_players(self.json_data, 100)
        pd.testing.assert_frame_equal(result[0], self.expected_home_players)
        pd.testing.assert_frame_equal(result[1], self.expected_away_players)

    def test_get_periods_frames(self):
        result = _get_periods_frames(
            self.json_data, pd.to_datetime("2023-01-01"), "Europe/Amsterdam"
        )
        pd.testing.assert_frame_equal(result, self.expected_periods)

    def test_load_event_data(self):
        res_event_data, res_databallpy_events = _load_event_data(
            self.json_loc, self.expected_metadata
        )
        pd.testing.assert_frame_equal(res_event_data, self.expected_event_data)

        for key, event in {
            **res_databallpy_events["shot_events"],
            **res_databallpy_events["pass_events"],
            **res_databallpy_events["dribble_events"],
        }.items():
            row = self.expected_event_data.loc[
                self.expected_event_data["event_id"] == key
            ].iloc[0]
            databallpy_event = row["databallpy_event"]

            self.assertAlmostEqual(event.event_id, key)
            self.assertAlmostEqual(event.period_id, row["period_id"])
            self.assertAlmostEqual(event.minutes, row["minutes"])
            self.assertAlmostEqual(event.seconds, row["seconds"])
            self.assertAlmostEqual(event.datetime, row["datetime"])
            self.assertAlmostEqual(event.team_id, row["team_id"])
            self.assertEqual(
                event.team_side, "home" if row["team_id"] == 100 else "away"
            )
            self.assertAlmostEqual(event.pitch_size, [106.0, 68.0])

            if databallpy_event == "shot":
                self.assertIsInstance(event, ShotEvent)
                self.assertEqual(event.start_x, row["start_x"])
            elif databallpy_event == "pass":
                self.assertIsInstance(event, PassEvent)
                self.assertTrue(event.end_y in [2.01, -20.8, 32.5])
            elif databallpy_event == "dribble":
                self.assertIsInstance(event, DribbleEvent)
                self.assertEqual(event.start_y, row["start_y"])

    def test_get_shot_event(self):
        event = {
            "eventId": 10,
            "playerId": 201,
            "playerName": "away player 1",
            "groupId": 2,
            "groupName": "AWAY",
            "teamId": 200,
            "teamName": "Team 2",
            "receiverId": 201,
            "receiverName": "away player 1",
            "receiverTeamId": 200,
            "receiverTeamName": "Team 2",
            "baseTypeId": 6,
            "baseTypeName": "SHOT",
            "subTypeId": 800,
            "subTypeName": "PENALTY",
            "resultId": 1,
            "resultName": "SUCCESSFUL",
            "bodyPartId": 3,
            "bodyPartName": "LEFT_FOOT",
            "shotTypeId": 1,
            "shotTypeName": "ON_TARGET",
            "foulTypeId": -1,
            "foulTypeName": "NOT_APPLICABLE",
            "positionTypeId": 12,
            "positionTypeName": "CMF",
            "formationTypeId": -2,
            "formationTypeName": "UNKNOWN",
            "partId": 2,
            "partName": "SECOND_HALF",
            "startTimeMs": 12700,
            "endTimeMs": 13500,
            "startPosXM": 42.5,
            "startPosYM": 0.0,
            "endPosXM": 53.5,
            "endPosYM": -3.5,
            "sequenceId": 3,
            "sequenceEvent": 1,
            "possessionTypeId": 2,
            "possessionTypeName": "FREE_KICK",
            "sequenceStart": 0,
            "sequenceEnd": 1,
            "synced": True,
        }
        expected_shot = ShotEvent(
            event_id=14,
            period_id=2,
            minutes=MISSING_INT,
            seconds=MISSING_INT,
            datetime=pd.to_datetime("NaT"),
            start_x=42.5,
            start_y=0.0,
            team_id=200,
            team_side="away",
            pitch_size=(106.0, 68.0),
            _xt=-1.0,
            player_id=201,
            shot_outcome="goal",
            y_target=np.nan,
            z_target=np.nan,
            body_part="left_foot",
        )
        result = _get_shot_event(event, 14)
        self.assertEqual(result, expected_shot)

    def test_get_pass_event(self):
        event = {
            "eventId": 6,
            "playerId": 102,
            "playerName": "home player 2",
            "groupId": 1,
            "groupName": "HOME",
            "teamId": 100,
            "teamName": "Team 1",
            "receiverId": 201,
            "receiverName": "away player 1",
            "receiverTeamId": 200,
            "receiverTeamName": "Team 2",
            "baseTypeId": 2,
            "baseTypeName": "CROSS",
            "subTypeId": 200,
            "subTypeName": "CROSS",
            "resultId": 0,
            "resultName": "UNSUCCESSFUL",
            "bodyPartId": 0,
            "bodyPartName": "FEET",
            "shotTypeId": -1,
            "shotTypeName": "NOT_APPLICABLE",
            "foulTypeId": -1,
            "foulTypeName": "NOT_APPLICABLE",
            "positionTypeId": 4,
            "positionTypeName": "CB",
            "formationTypeId": -2,
            "formationTypeName": "UNKNOWN",
            "partId": 1,
            "partName": "FIRST_HALF",
            "startTimeMs": 6600,
            "endTimeMs": 9300,
            "startPosXM": 24.01,
            "startPosYM": 25.97,
            "endPosXM": -35.1,
            "endPosYM": -20.8,
            "sequenceId": 1,
            "sequenceEvent": 1,
            "possessionTypeId": 2,
            "possessionTypeName": "FREE_KICK",
            "sequenceStart": 0,
            "sequenceEnd": 1,
            "synced": True,
        }
        expected_pass = PassEvent(
            event_id=13,
            period_id=1,
            minutes=MISSING_INT,
            seconds=MISSING_INT,
            datetime=pd.to_datetime("NaT"),
            start_x=24.01,
            start_y=25.97,
            team_id=100,
            team_side="home",
            pitch_size=(106.0, 68.0),
            _xt=-1.0,
            outcome="unsuccessful",
            player_id=102,
            receiver_id=MISSING_INT,
            end_x=-35.1,
            end_y=-20.8,
            pass_type="cross",
            set_piece="no_set_piece",
        )
        result = _get_pass_event(event, 13)
        self.assertEqual(result, expected_pass)

    def test_get_dribble_event(self):
        event = {
            "eventId": 6,
            "playerId": 102,
            "playerName": "home player 2",
            "groupId": 1,
            "groupName": "HOME",
            "teamId": 100,
            "teamName": "Team 1",
            "receiverId": -1,
            "receiverName": "NOT_APPLICABLE",
            "receiverTeamId": -1,
            "receiverTeamName": "NOT_APPLICABLE",
            "baseTypeId": 3,
            "baseTypeName": "DRIBBLE",
            "subTypeId": 300,
            "subTypeName": "CARRY",
            "resultId": 1,
            "resultName": "SUCCESSFUL",
            "bodyPartId": 0,
            "bodyPartName": "FEET",
            "shotTypeId": -1,
            "shotTypeName": "NOT_APPLICABLE",
            "foulTypeId": -1,
            "foulTypeName": "NOT_APPLICABLE",
            "positionTypeId": 4,
            "positionTypeName": "CB",
            "formationTypeId": -2,
            "formationTypeName": "UNKNOWN",
            "partId": 1,
            "partName": "FIRST_HALF",
            "startTimeMs": 3400,
            "endTimeMs": 6600,
            "startPosXM": -17.68,
            "startPosYM": 2.01,
            "endPosXM": 24.01,
            "endPosYM": 25.97,
            "sequenceId": 1,
            "sequenceEvent": 1,
            "possessionTypeId": 2,
            "possessionTypeName": "FREE_KICK",
            "sequenceStart": 0,
            "sequenceEnd": 1,
            "synced": True,
        }
        expected_dribble = DribbleEvent(
            event_id=12,
            period_id=1,
            minutes=MISSING_INT,
            seconds=MISSING_INT,
            datetime=pd.to_datetime("NaT"),
            start_x=-17.68,
            start_y=2.01,
            team_id=100,
            team_side="home",
            pitch_size=(106.0, 68.0),
            _xt=-1.0,
            player_id=102,
            related_event_id=MISSING_INT,
            duel_type="offensive",
            outcome=True,
            has_opponent=False,
        )

        result = _get_dribble_event(event, 12)
        self.assertEqual(result, expected_dribble)
