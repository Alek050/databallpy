import copy
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from databallpy.data_parsers.event_data_parsers.opta_parser import (
    _get_opta_end_location,
    _get_opta_spadl_action,
    _get_player_info,
    _get_valid_opta_datetime,
    _load_event_data,
    _load_metadata,
    _make_dribble_event_instance,
    _make_pass_instance,
    _make_shot_event_instance,
    _rescale_opta_dimensions,
    _update_opta_spadl_actions,
    _update_pass_outcome,
    load_opta_event_data,
)
from databallpy.events import DribbleEvent, PassEvent, ShotEvent
from databallpy.schemas import EventData
from databallpy.utils.constants import SPADL_ACTION_TYPES, SPADL_RESULTS
from databallpy.utils.utils import MISSING_INT
from databallpy.utils.warnings import DataBallPyWarning
from tests.expected_outcomes import (
    DRIBBLE_INSTANCES_OPTA,
    DRIBBLE_INSTANCES_OPTA_MA13,
    ED_OPTA,
    ED_OPTA_MA13,
    MD_OPTA,
    MD_OPTA_MA2,
    PASS_INSTANCES_OPTA,
    PASS_INSTANCES_OPTA_MA13,
    SHOT_INSTANCES_OPTA,
    SHOT_INSTANCES_OPTA_MA13,
)
from tests.pandas_testing import assert_frame_equal, to_list

ED_OPTA = pd.DataFrame(ED_OPTA.drop(columns=["team_name"]).copy())
ED_OPTA_MA13 = pd.DataFrame(ED_OPTA_MA13.drop(columns=["team_name"]).copy())


class TestOptaParser(unittest.TestCase):
    def setUp(self):
        self.f7_loc = "tests/test_data/f7_test.xml"
        self.f7_loc_no_timestamps = "tests/test_data/f7_test_no_timestamps.xml"
        self.f7_loc_no_timestamps_and_date = (
            "tests/test_data/f7_test_no_timestamps_and_date.xml"
        )
        self.f7_loc_multiple_games = "tests/test_data/f7_test_multiple_games.xml"
        self.f24_loc = "tests/test_data/f24_test.xml"
        self.ma2_loc = "tests/test_data/ma2_test.xml"
        self.ma13_loc = "tests/test_data/ma13_test.xml"

    def test_load_opta_event_data(self):
        event_data, metadata, dbp_events = load_opta_event_data(
            self.f7_loc, self.f24_loc, pitch_dimensions=[100.0, 50.0]
        )
        assert_frame_equal(event_data, ED_OPTA)
        assert metadata == MD_OPTA

        # SHOT_INSTANCES_OPTA is scaled to a pitch of [106, 68],
        # while here [100, 50] is expected.
        expected_shot_events_opta = {}
        for shot_id, shot_event in SHOT_INSTANCES_OPTA.items():
            expected_shot_events_opta[shot_id] = shot_event.copy()
            expected_shot_events_opta[shot_id].start_x = shot_event.start_x / 106.0 * 100
            expected_shot_events_opta[shot_id].start_y = shot_event.start_y / 68.0 * 50
            expected_shot_events_opta[shot_id]._xt = -1
            expected_shot_events_opta[shot_id]._xt = expected_shot_events_opta[
                shot_id
            ].xt
            shot_event._xt = -1
            shot_event._xt = shot_event.xt
            expected_shot_events_opta[shot_id].pitch_size = [100.0, 50.0]
            expected_shot_events_opta[shot_id]._update_shot_angle()
            expected_shot_events_opta[shot_id]._update_ball_goal_distance()
            expected_shot_events_opta[shot_id].xg = expected_shot_events_opta[
                shot_id
            ].get_xg()

        assert "shot_events" in dbp_events.keys()
        for key, event in dbp_events["shot_events"].items():
            assert key in expected_shot_events_opta.keys()
            assert event == expected_shot_events_opta[key]

    def test_load_opta_event_data_ma(self):
        # same game as test_load_opta_event_data, but via the new MA2/MA13 opta
        # format (opaque alphanumeric ids) instead of the legacy F7/F24 format.
        event_data, metadata, dbp_events = load_opta_event_data(
            self.ma2_loc, self.ma13_loc, pitch_dimensions=[100.0, 50.0]
        )
        assert_frame_equal(event_data, ED_OPTA_MA13)
        assert metadata == MD_OPTA_MA2

        # SHOT_INSTANCES_OPTA_MA13 is scaled to a pitch of [106, 68],
        # while here [100, 50] is expected.
        expected_shot_events_opta = {}
        for shot_id, shot_event in SHOT_INSTANCES_OPTA_MA13.items():
            expected_shot_events_opta[shot_id] = shot_event.copy()
            expected_shot_events_opta[shot_id].start_x = shot_event.start_x / 106.0 * 100
            expected_shot_events_opta[shot_id].start_y = shot_event.start_y / 68.0 * 50
            expected_shot_events_opta[shot_id]._xt = -1
            expected_shot_events_opta[shot_id]._xt = expected_shot_events_opta[
                shot_id
            ].xt
            shot_event._xt = -1
            shot_event._xt = shot_event.xt
            expected_shot_events_opta[shot_id].pitch_size = [100.0, 50.0]
            expected_shot_events_opta[shot_id]._update_shot_angle()
            expected_shot_events_opta[shot_id]._update_ball_goal_distance()
            expected_shot_events_opta[shot_id].xg = expected_shot_events_opta[
                shot_id
            ].get_xg()

        assert "shot_events" in dbp_events.keys()
        for key, event in dbp_events["shot_events"].items():
            assert key in expected_shot_events_opta.keys()
            assert event == expected_shot_events_opta[key]

    def test_load_event_data_pass(self):
        event_data, metadata, dbp_events = load_opta_event_data(
            self.f7_loc, "pass", pitch_dimensions=[100.0, 50.0]
        )
        assert metadata == MD_OPTA
        assert_frame_equal(event_data, pd.DataFrame())
        assert dbp_events == {}

    def test_load_opta_event_data_errors(self):
        with self.assertRaises(TypeError):
            load_opta_event_data(3, self.f24_loc, pitch_dimensions=[100.0, 50.0])
        with self.assertRaises(TypeError):
            load_opta_event_data(
                self.f7_loc, [self.f24_loc], pitch_dimensions=[100.0, 50.0]
            )
        with self.assertRaises(ValueError):
            load_opta_event_data(
                self.f7_loc + ".json", self.f24_loc, pitch_dimensions=[100.0, 50.0]
            )
        with self.assertRaises(ValueError):
            load_opta_event_data(
                self.f7_loc, self.f24_loc[:-3], pitch_dimensions=[100.0, 50.0]
            )

    def test_load_metadata(self):
        metadata = _load_metadata(self.f7_loc, [100.0, 50.0])
        assert metadata == MD_OPTA

    def test_load_metadata_ma2(self):
        metadata = _load_metadata(self.ma2_loc, [100.0, 50.0])
        assert metadata == MD_OPTA_MA2

    def test_load_metadata_multiple_games(self):
        metadata = _load_metadata(self.f7_loc_multiple_games, [100.0, 50.0])
        # the second game metadata is dropped
        assert metadata == MD_OPTA

    def test_load_metadata_no_timestamps_and_date(self):
        with self.assertRaises(ValueError):
            _load_metadata(self.f7_loc_no_timestamps_and_date, [100.0, 50.0])

    def test_load_metadata_no_timestamps(self):
        with self.assertWarns(DataBallPyWarning):
            metadata = _load_metadata(self.f7_loc_no_timestamps, [100.0, 50.0])
        expected_md = MD_OPTA.copy()
        pf = expected_md.periods_frames
        pf.loc[0, "start_datetime_ed"] = pd.to_datetime("20230122T111500+0000", utc=True)
        pf.loc[0, "end_datetime_ed"] = pd.to_datetime("20230122T120000+0000", utc=True)
        pf.loc[1, "start_datetime_ed"] = pd.to_datetime("20230122T121500+0000", utc=True)
        pf.loc[1, "end_datetime_ed"] = pd.to_datetime("20230122T130000+0000", utc=True)

        assert metadata == expected_md

    def test_get_player_info(self):
        player_data = [
            {
                "PlayerRef": "s123",
                "Formation_Place": "0",
                "Position": "Substitute",
                "SubPosition": "Midfielder",
                "Status": "Substitute",
                "ShirtNumber": "33",
            },
            {
                "PlayerRef": "s234",
                "Formation_Place": "1",
                "Position": "GoalKeeper",
                "Status": "Start",
                "ShirtNumber": "2",
            },
        ]
        player_names = {"123": "Sven Kerhoffs", "234": "Niels Smits"}

        expected_result = pd.DataFrame(
            {
                "id": [123, 234],
                "full_name": ["Sven Kerhoffs", "Niels Smits"],
                "formation_place": [0, 1],
                "position": ["midfielder", "goalkeeper"],
                "starter": [False, True],
                "shirt_num": [33, 2],
            }
        )

        result = _get_player_info(player_data, player_names)
        assert_frame_equal(result, expected_result)

    def test_load_event_data(self):
        players = pd.concat(
            [MD_OPTA.home_players, MD_OPTA.away_players], ignore_index=True
        )
        event_data, dbp_events = _load_event_data(
            self.f24_loc, away_team_id=194, country="Netherlands", players=players
        )

        # player name is added in other function later in the pipeline
        expected_event_data = ED_OPTA.copy().drop("player_name", axis=1)

        # away team coordinates are changed later on in the pipeline
        expected_event_data.loc[
            expected_event_data["team_id"] == 194,
            ["start_x", "start_y", "end_x", "end_y"],
        ] *= -1

        # scaling of pitch dimension is done later on in the pipeline
        expected_event_data.loc[:, ["start_x", "end_x"]] += 50
        expected_event_data.loc[:, ["start_y", "end_y"]] = (
            expected_event_data.loc[:, ["start_y", "end_y"]] + 25
        ) * (100 / 50)

        assert_frame_equal(event_data, expected_event_data)

        assert "shot_events" in dbp_events.keys()
        for key, event in dbp_events["shot_events"].items():
            assert key in SHOT_INSTANCES_OPTA.keys()
            assert event == SHOT_INSTANCES_OPTA[key]

        assert "dribble_events" in dbp_events.keys()
        for key, event in dbp_events["dribble_events"].items():
            assert key in DRIBBLE_INSTANCES_OPTA.keys()
            assert event == DRIBBLE_INSTANCES_OPTA[key]

        assert "pass_events" in dbp_events.keys()
        for key, event in dbp_events["pass_events"].items():
            assert key in PASS_INSTANCES_OPTA.keys()
            assert event == PASS_INSTANCES_OPTA[key]

    def test_load_event_data_ma13(self):
        players = pd.DataFrame(
            {
                "id": ["hplayermaid01", "hplayermaid02", "aplayermaid01"],
                "shirt_num": [1, 2, 1],
            }
        )
        event_data, dbp_events = _load_event_data(
            self.ma13_loc,
            away_team_id="ateammaid02",
            country="Netherlands",
            players=players,
        )

        # player name is added in other function later in the pipeline
        expected_event_data = ED_OPTA_MA13.copy().drop("player_name", axis=1)

        # away team coordinates are changed later on in the pipeline
        expected_event_data.loc[
            expected_event_data["team_id"] == "ateammaid02",
            ["start_x", "start_y", "end_x", "end_y"],
        ] *= -1

        # scaling of pitch dimension is done later on in the pipeline
        expected_event_data.loc[:, ["start_x", "end_x"]] += 50
        expected_event_data.loc[:, ["start_y", "end_y"]] = (
            expected_event_data.loc[:, ["start_y", "end_y"]] + 25
        ) * (100 / 50)

        assert_frame_equal(event_data, expected_event_data)

        assert "shot_events" in dbp_events.keys()
        for key, event in dbp_events["shot_events"].items():
            assert key in SHOT_INSTANCES_OPTA_MA13.keys()
            assert event == SHOT_INSTANCES_OPTA_MA13[key]

        assert "dribble_events" in dbp_events.keys()
        for key, event in dbp_events["dribble_events"].items():
            assert key in DRIBBLE_INSTANCES_OPTA_MA13.keys()
            assert event == DRIBBLE_INSTANCES_OPTA_MA13[key]

        assert "pass_events" in dbp_events.keys()
        for key, event in dbp_events["pass_events"].items():
            assert key in PASS_INSTANCES_OPTA_MA13.keys()
            assert event == PASS_INSTANCES_OPTA_MA13[key]

    def test_make_shot_event_instance(self):
        event_xml = """
            <Event event_id="15" id="2529877443" last_modified="2023-04-08T02:39:48"
                   min="0" outcome="1" period_id="1" player_id="223345" sec="31"
                   team_id="325" timestamp="2023-04-07T19:02:07.364"
                   timestamp_utc="2023-04-07T18:02:07.364" type_id="16"
                   version="1680917987858" x="46.2" y="6.1">
                <Q id="4171346209" qualifier_id="464"/>
                <Q id="4170359433" qualifier_id="233" value="34"/>
                <Q id="4170356073" qualifier_id="286"/>
                <Q id="4170356071" qualifier_id="56" value="Back"/>
                <Q id="4170356075" qualifier_id="102" value="22.8"/>
                <Q id="4170356077" qualifier_id="103" value="7.2"/>
                <Q id="4170356079" qualifier_id="20"/>
            </Event>
            """

        event = BeautifulSoup(event_xml, "xml").find("Event")

        expected_output = ShotEvent(
            event_id=22,
            period_id=1,
            minutes=0,
            seconds=31,
            datetime=pd.to_datetime("2023-04-07T18:02:07.364", utc=True),
            start_x=-4.0279999,
            start_y=-29.852,
            team_side="home",
            team_id=325,
            pitch_size=[106.0, 68.0],
            player_id=223345,
            jersey=3,
            outcome=True,
            related_event_id=34,
            body_part="right_foot",
            possession_type="open_play",
            set_piece="no_set_piece",
            _xt=-1.0,
            outcome_str="goal",
            y_target=-1.991040,
            z_target=0.17568,
            first_touch=False,
        )
        players = pd.DataFrame(
            {
                "id": [223345],
                "full_name": ["Player 1"],
                "team_id": [325],
                "shirt_num": [3],
            }
        )
        actual_output = _make_shot_event_instance(event, 111, players=players, id=22)
        self.assertEqual(actual_output, expected_output)

    def test_make_dribble_event_instance(self):
        event_xml = """
            <Event type_id="3" event_id="15" id="2529877443"
            last_modified="2023-04-08T02:39:48" min="0" outcome="1" period_id="1"
            player_id="223345" sec="31" team_id="325"
            timestamp="2023-04-07T19:02:07.364" timestamp_utc="2023-04-07T18:02:07.364"
            version="1680917987858" x="46.2" y="6.1">
                <Q id="4171346209" qualifier_id="464"/>
                <Q id="4170356073" qualifier_id="286"/>
                <Q id="4170356071" qualifier_id="56" value="Back"/>
            </Event>
        """
        event = BeautifulSoup(event_xml, "xml").find("Event")

        players = pd.DataFrame(
            {
                "id": [223345],
                "full_name": ["Player 1"],
                "team_id": [325],
                "shirt_num": [3],
            }
        )
        dribble_event = _make_dribble_event_instance(
            event, away_team_id=325, players=players, id=12
        )

        expected_dribble_event = DribbleEvent(
            event_id=12,
            period_id=1,
            minutes=0,
            seconds=31,
            datetime=pd.to_datetime("2023-04-07T18:02:07.364", utc=True),
            start_x=4.0279999,
            start_y=29.852,
            pitch_size=[106.0, 68.0],
            player_id=223345,
            jersey=3,
            team_side="away",
            team_id=325,
            _xt=-1.0,
            related_event_id=MISSING_INT,
            set_piece="no_set_piece",
            possession_type="open_play",
            body_part="foot",
            outcome=True,
            with_opponent=True,
            duel_type="offensive",
        )
        assert dribble_event == expected_dribble_event

    def test_make_pass_event_instance(self):
        event_xml = """
        <Event id="1" type_id="1" period_id="1" min="0" sec="0" team_id="1"
         player_id="1" outcome="1" x="50.0" y="50.0"
         timestamp="2022-01-01T00:00:00.000Z">
            <Q id="1" qualifier_id="210"/>
            <Q id="2" qualifier_id="1"/>
            <Q id="3" qualifier_id="157"/>
        </Event>"""

        event = BeautifulSoup(event_xml, "xml").find("Event")
        players = players = pd.DataFrame(
            {"id": [1], "full_name": ["Player 1"], "team_id": [1], "shirt_num": [4]}
        )
        pass_event = _make_pass_instance(event, away_team_id=1, players=players, id=13)

        expected_pass_event = PassEvent(
            event_id=13,
            period_id=1,
            minutes=0,
            seconds=0,
            datetime=pd.to_datetime("2022-01-01T00:00:00.000Z", utc=True),
            start_x=0.0,
            start_y=0.0,
            pitch_size=[106.0, 68.0],
            team_side="away",
            team_id=1,
            player_id=1,
            jersey=4,
            _xt=-1.0,
            outcome=True,
            related_event_id=MISSING_INT,
            set_piece="no_set_piece",
            possession_type="open_play",
            body_part="unspecified",
            outcome_str="results_in_shot",
            end_x=np.nan,
            end_y=np.nan,
            pass_type="long_ball",
        )
        assert pass_event == expected_pass_event

    def test_rescale_opta_dimensions(self):
        # test with default pitch dimensions
        x, y = _rescale_opta_dimensions(50, 50)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)

        # test with custom pitch dimensions
        x, y = _rescale_opta_dimensions(50, 100, [100.0, 50.0])
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 25.0)

        # test with different coordinates
        x, y = _rescale_opta_dimensions(0, 0)
        self.assertAlmostEqual(x, -53.0)
        self.assertAlmostEqual(y, -34.0)

        x, y = _rescale_opta_dimensions(100, 100)
        self.assertAlmostEqual(x, 53.0)
        self.assertAlmostEqual(y, 34.0)

    def test_update_pass_outcome_no_related_event_id(self):
        shot_events = copy.deepcopy(SHOT_INSTANCES_OPTA)

        pass_events = copy.deepcopy(PASS_INSTANCES_OPTA)
        event_data = ED_OPTA.copy()

        for event in shot_events.values():
            event.related_event_id = MISSING_INT

        res_passes = _update_pass_outcome(event_data, shot_events, pass_events)

        for key, event in res_passes.items():
            assert key in pass_events.keys()
            assert event == pass_events[key]

    def test_update_pass_outcome_pass_not_found(self):
        shot_events = copy.deepcopy(SHOT_INSTANCES_OPTA)

        pass_events = copy.deepcopy(PASS_INSTANCES_OPTA)
        event_data = ED_OPTA.copy()

        for event in shot_events.values():
            # none of the pass events have event_id 120
            event.related_event_id = 120

        res_passes = _update_pass_outcome(event_data, shot_events, pass_events)

        for key, event in res_passes.items():
            assert key in pass_events.keys()
            assert event == pass_events[key]

    def test_update_pass_outcome_multiple_options(self):
        shot_events = copy.deepcopy(SHOT_INSTANCES_OPTA)

        pass_events = copy.deepcopy(PASS_INSTANCES_OPTA)
        pass_events[4].outcome_str = "successful"
        expected_passes = copy.deepcopy(pass_events)
        event_data = ED_OPTA.copy()

        for event in shot_events.values():
            if event.outcome_str == "goal":
                event.related_event_id = 120

        event_data.loc[event_data["databallpy_event"] == "pass", "original_event_id"] = (
            120
        )

        # the pass event with event_id 120 has two options
        assert len(event_data.loc[event_data["original_event_id"] == 120]) == 2

        res_passes = _update_pass_outcome(event_data, shot_events, pass_events)

        expected_passes[4].outcome_str = "assist"
        for key, event in res_passes.items():
            assert key in expected_passes.keys()
            assert event == expected_passes[key]

    def test_get_valid_opta_datetime(self):
        event_xml = """
            <Event event_id="15" id="2529877443" last_modified="2023-04-08T02:39:48"
                   min="0" outcome="1" period_id="1" player_id="223345" sec="31"
                   team_id="325" timestamp="2023-04-07T19:02:07.364"
                   timestamp_utc="2023-04-07T18:02:07.364" type_id="16"
                   version="1680917987858" x="46.2" y="6.1">
                <Q id="4171346209" qualifier_id="464"/>
                <Q id="4170359433" qualifier_id="233" value="34"/>
                <Q id="4170356073" qualifier_id="286"/>
                <Q id="4170356071" qualifier_id="56" value="Back"/>
                <Q id="4170356075" qualifier_id="102" value="22.8"/>
                <Q id="4170356077" qualifier_id="103" value="7.2"/>
                <Q id="4170356079" qualifier_id="20"/>
            </Event>
            """
        event = BeautifulSoup(event_xml, "xml").find("Event")
        res_dt = _get_valid_opta_datetime(event)
        assert res_dt == pd.to_datetime("2023-04-07T18:02:07.364", utc=True)

        event_xml = """
            <Event event_id="15" id="2529877443" last_modified="2023-04-08T02:39:48"
                   min="0" outcome="1" period_id="1" player_id="223345" sec="31"
                   team_id="325" timestamp="2023-04-07T19:02:07.364Z"
                   type_id="16"
                   version="1680917987858" x="46.2" y="6.1">
                <Q id="4171346209" qualifier_id="464"/>
                <Q id="4170359433" qualifier_id="233" value="34"/>
                <Q id="4170356073" qualifier_id="286"/>
                <Q id="4170356071" qualifier_id="56" value="Back"/>
                <Q id="4170356075" qualifier_id="102" value="22.8"/>
                <Q id="4170356077" qualifier_id="103" value="7.2"/>
                <Q id="4170356079" qualifier_id="20"/>
            </Event>
            """
        event = BeautifulSoup(event_xml, "xml").find("Event")
        res_dt = _get_valid_opta_datetime(event)
        assert res_dt == pd.to_datetime("2023-04-07T19:02:07.364", utc=True)

        event_xml = """
            <Event event_id="15" id="2529877443" last_modified="2023-04-08T02:39:48"
                   min="0" outcome="1" period_id="1" player_id="223345" sec="31"
                   team_id="325" timestamp="2023-04-07T19:02:07.364"
                   type_id="16"
                   version="1680917987858" x="46.2" y="6.1">
                <Q id="4171346209" qualifier_id="464"/>
                <Q id="4170359433" qualifier_id="233" value="34"/>
                <Q id="4170356073" qualifier_id="286"/>
                <Q id="4170356071" qualifier_id="56" value="Back"/>
                <Q id="4170356075" qualifier_id="102" value="22.8"/>
                <Q id="4170356077" qualifier_id="103" value="7.2"/>
                <Q id="4170356079" qualifier_id="20"/>
            </Event>
            """
        event = BeautifulSoup(event_xml, "xml").find("Event")
        res_dt = _get_valid_opta_datetime(event)
        assert res_dt == pd.to_datetime("2023-04-07T18:02:07.364", utc=True)


def _opta_event(outcome: int = 1, qualifiers: dict | None = None) -> ET.Element:
    event = ET.Element("Event", attrib={"outcome": str(outcome)})
    for qualifier_id, value in (qualifiers or {}).items():
        attrib = {"qualifier_id": str(qualifier_id)}
        if value is not None:
            attrib["value"] = str(value)
        ET.SubElement(event, "Q", attrib=attrib)
    return event


class TestOptaSpadl(unittest.TestCase):
    def setUp(self):
        self.f7_loc = "tests/test_data/f7_test.xml"
        self.f24_loc = "tests/test_data/f24_test.xml"

    def test_get_opta_spadl_action_passes(self):
        cases = [
            ("pass", 1, {}, ("pass", "success", "foot")),
            ("pass", 0, {107: None}, ("throw_in", "fail", "other")),
            ("pass", 1, {5: None, 2: None}, ("freekick_crossed", "success", "foot")),
            ("pass", 1, {5: None, 155: None}, ("freekick_crossed", "success", "foot")),
            ("pass", 0, {5: None}, ("freekick_short", "fail", "foot")),
            ("pass", 1, {6: None, 2: None}, ("corner_crossed", "success", "foot")),
            ("pass", 1, {6: None, 1: None}, ("corner_short", "success", "foot")),
            ("pass", 1, {2: None, 72: None}, ("cross", "success", "foot_left")),
            ("pass", 1, {124: None, 20: None}, ("goalkick", "success", "foot_right")),
            ("pass", 1, {15: None}, ("pass", "success", "head")),
            ("pass", 1, {168: None}, ("pass", "success", "head")),
            ("pass", 1, {21: None}, ("pass", "success", "other")),
            ("offside pass", 0, {}, ("pass", "offside", "foot")),
            ("pass", 1, {238: None}, (None, None, None)),
        ]
        for event_name, outcome, qualifiers, expected in cases:
            with self.subTest(event_name=event_name, qualifiers=qualifiers):
                event = _opta_event(outcome, qualifiers)
                self.assertEqual(_get_opta_spadl_action(event, event_name), expected)

    def test_get_opta_spadl_action_shots(self):
        cases = [
            ("goal", {}, ("shot", "success", "foot")),
            ("goal", {9: None}, ("shot_penalty", "success", "foot")),
            ("miss", {26: None, 15: None}, ("shot_freekick", "fail", "head")),
            ("post", {}, ("shot", "fail", "foot")),
            ("attempt saved", {28: None}, ("shot", "fail", "foot")),
            ("goal", {28: None}, ("bad_touch", "owngoal", "foot")),
            ("own goal", {}, ("bad_touch", "owngoal", "foot")),
        ]
        for event_name, qualifiers, expected in cases:
            with self.subTest(event_name=event_name, qualifiers=qualifiers):
                event = _opta_event(1, qualifiers)
                self.assertEqual(_get_opta_spadl_action(event, event_name), expected)

    def test_get_opta_spadl_action_other_events(self):
        cases = [
            ("take on", 0, {}, ("take_on", "fail", "foot")),
            ("tackle", 1, {}, ("tackle", "success", "foot")),
            ("interception", 1, {}, ("interception", "success", "foot")),
            ("blocked pass", 1, {}, ("interception", "success", "foot")),
            ("foul", 0, {}, ("foul", "fail", "foot")),
            ("foul", 1, {}, (None, None, None)),
            ("save", 1, {}, ("keeper_save", "success", "other")),
            ("save", 1, {94: None}, (None, None, None)),
            ("smother", 1, {}, ("keeper_save", "success", "other")),
            ("claim", 0, {}, ("keeper_claim", "fail", "other")),
            ("punch", 1, {}, ("keeper_punch", "success", "other")),
            ("keeper pick-up", 1, {}, ("keeper_pick_up", "success", "other")),
            ("keeper sweeper", 1, {}, ("clearance", "success", "foot")),
            ("keeper sweeper", 0, {}, (None, None, None)),
            ("clearance", 1, {15: None}, ("clearance", "success", "head")),
            ("ball touch", 0, {}, ("bad_touch", "fail", "foot")),
            ("ball touch", 1, {}, (None, None, None)),
            ("ball recovery", 1, {}, ("non_action", "success", "foot")),
            ("aerial", 1, {}, (None, None, None)),
            ("card", 1, {31: None}, (None, "yellow_card", None)),
            ("card", 1, {32: None}, (None, "red_card", None)),
            ("card", 1, {33: None}, (None, "red_card", None)),
            ("card", 1, {}, (None, None, None)),
        ]
        for event_name, outcome, qualifiers, expected in cases:
            with self.subTest(event_name=event_name, outcome=outcome):
                event = _opta_event(outcome, qualifiers)
                self.assertEqual(_get_opta_spadl_action(event, event_name), expected)

    def test_get_opta_end_location(self):
        cases = [
            ({140: 30.0, 141: 40.0}, False, (30.0, 40.0)),
            ({146: 80.0, 147: 45.0, 102: 50.0}, False, (80.0, 45.0)),
            ({102: 54.0, 103: 10.0}, False, (100.0, 54.0)),
            ({102: 54.0}, True, (0.0, 46.0)),
            ({140: 30.0}, False, (30.0, np.nan)),
            ({140: None, 141: "x"}, False, (np.nan, np.nan)),
            ({}, False, (np.nan, np.nan)),
        ]
        for qualifiers, own_goal, expected in cases:
            with self.subTest(qualifiers=qualifiers, own_goal=own_goal):
                res = _get_opta_end_location(_opta_event(1, qualifiers), own_goal)
                np.testing.assert_allclose(res, expected)

    def test_update_opta_spadl_actions(self):
        event_data = pd.DataFrame(
            {
                "original_event": [
                    "interception",
                    "ball recovery",
                    "pass",
                    "ball touch",
                    "pass",
                    "ball recovery",
                    "interception",
                ],
                "team_id": [1, 1, 1, 2, 1, 1, 2],
                "period_id": [1] * 7,
                "minutes": [0] * 7,
                "seconds": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "start_x": [0.0, 0.0, 10.0, 15.0, 25.0, 25.0, 26.0],
                "start_y": [0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 6.0],
                "end_x": [np.nan, np.nan, 20.0, np.nan, np.nan, np.nan, np.nan],
                "end_y": [np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, np.nan],
                "action_type": [
                    "interception",
                    "dribble",
                    "pass",
                    None,
                    "pass",
                    "dribble",
                    "interception",
                ],
                "action_result": [
                    "success",
                    "success",
                    "fail",
                    None,
                    "success",
                    "success",
                    "success",
                ],
                "action_bodypart": ["foot"] * 3 + [None] + ["foot"] * 3,
            }
        )
        res = _update_opta_spadl_actions(event_data)

        # ball recoveries are no actions, the carries are added in to_spadl
        self.assertListEqual(
            to_list(res["action_type"]),
            [
                "interception",
                "non_action",
                "pass",
                None,
                "pass",
                "non_action",
                "interception",
            ],
        )
        self.assertTrue(res.loc[[1, 5], "end_x"].isna().all())
        # the deflected pass still reached a teammate
        self.assertListEqual(res.loc[2, ["end_x", "end_y"]].tolist(), [25.0, 5.0])
        self.assertEqual(res.loc[2, "action_result"], "success")
        # interception results are fixed in to_spadl
        self.assertEqual(res.loc[0, "action_result"], "success")
        self.assertEqual(res.loc[6, "action_result"], "success")

    def test_update_opta_spadl_actions_cards(self):
        event_data = pd.DataFrame(
            {
                "original_event": ["foul", "foul", "card"],
                "team_id": [1, 2, 1],
                "player_id": [5, 6, 5],
                "period_id": [1] * 3,
                "minutes": [0] * 3,
                "seconds": [1.0, 1.0, 20.0],
                "datetime": pd.to_datetime(
                    [
                        "2023-01-01 00:00:01",
                        "2023-01-01 00:00:01",
                        "2023-01-01 00:00:20",
                    ]
                ),
                "start_x": [0.0, 0.0, 0.0],
                "start_y": [0.0, 0.0, 0.0],
                "end_x": [np.nan] * 3,
                "end_y": [np.nan] * 3,
                "action_type": ["foul", None, None],
                "action_result": ["fail", None, "yellow_card"],
                "action_bodypart": ["foot", None, None],
            }
        )
        res = _update_opta_spadl_actions(event_data)

        # the card is no action, but the result of the foul of the same player
        self.assertListEqual(to_list(res["action_result"]), ["yellow_card", None, None])

    def test_to_spadl(self):
        event_data, metadata, _ = load_opta_event_data(self.f7_loc, self.f24_loc)
        # the second half events in the test data have first half minutes
        event_data.loc[event_data["period_id"] == 2, "minutes"] += 45
        res = EventData(event_data, provider="opta").to_spadl(
            home_team_id=metadata.home_team_id,
            pitch_dimensions=metadata.pitch_dimensions,
            coordinate_system="spadl",
        )
        res = res[res["original_event_id"].notna()].reset_index(drop=True)
        self.assertListEqual(
            [SPADL_ACTION_TYPES[i] for i in res["type_id"]],
            ["pass", "pass", "bad_touch", "bad_touch", "shot", "take_on", "tackle"],
        )
        self.assertListEqual(
            [SPADL_RESULTS[i] for i in res["result_id"]],
            ["success", "fail", "owngoal", "owngoal", "fail", "fail", "success"],
        )
