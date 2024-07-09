import copy
import unittest

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from databallpy.data_parsers.event_data_parsers.opta_parser import (
    _get_player_info,
    _get_valid_opta_datetime,
    _load_event_data,
    _load_metadata,
    _make_dribble_event_instance,
    _make_pass_instance,
    _make_shot_event_instance,
    _rescale_opta_dimensions,
    _update_pass_outcome,
    load_opta_event_data,
)
from databallpy.events import DribbleEvent, PassEvent, ShotEvent
from databallpy.utils.utils import MISSING_INT
from databallpy.utils.warnings import DataBallPyWarning
from tests.expected_outcomes import (
    DRIBBLE_EVENTS_OPTA,
    ED_OPTA,
    MD_OPTA,
    PASS_EVENTS_OPTA,
    SHOT_EVENTS_OPTA,
)


class TestOptaParser(unittest.TestCase):
    def setUp(self):
        self.f7_loc = "tests/test_data/f7_test.xml"
        self.f7_loc_no_timestamps = "tests/test_data/f7_test_no_timestamps.xml"
        self.f7_loc_no_timestamps_and_date = (
            "tests/test_data/f7_test_no_timestamps_and_date.xml"
        )
        self.f7_loc_multiple_matches = "tests/test_data/f7_test_multiple_matches.xml"
        self.f24_loc = "tests/test_data/f24_test.xml"

    def test_load_opta_event_data(self):
        event_data, metadata, dbp_events = load_opta_event_data(
            self.f7_loc, self.f24_loc, pitch_dimensions=[10.0, 10.0]
        )
        pd.testing.assert_frame_equal(event_data, ED_OPTA)
        assert metadata == MD_OPTA

        # SHOT_EVENTS_OPTA is scaled to a pitch of [106, 68],
        # while here [10, 10] is expected.
        expected_shot_events_opta = {}
        for shot_id, shot_event in SHOT_EVENTS_OPTA.items():
            expected_shot_events_opta[shot_id] = shot_event.copy()
            expected_shot_events_opta[shot_id].start_x = shot_event.start_x / 106.0 * 10
            expected_shot_events_opta[shot_id].start_y = shot_event.start_y / 68.0 * 10
            expected_shot_events_opta[shot_id].pitch_size = [10.0, 10.0]
            expected_shot_events_opta[shot_id]._update_shot_angle()
            expected_shot_events_opta[shot_id]._update_ball_goal_distance()
            expected_shot_events_opta[shot_id].xG = expected_shot_events_opta[
                shot_id
            ].get_xG()

        assert "shot_events" in dbp_events.keys()
        for key, event in dbp_events["shot_events"].items():
            assert key in expected_shot_events_opta.keys()
            assert event == expected_shot_events_opta[key]

    def test_load_event_data_pass(self):
        event_data, metadata, dbp_events = load_opta_event_data(
            self.f7_loc, "pass", pitch_dimensions=[10.0, 10.0]
        )
        assert metadata == MD_OPTA
        pd.testing.assert_frame_equal(event_data, pd.DataFrame())
        assert dbp_events == {}

    def test_load_opta_event_data_errors(self):
        with self.assertRaises(TypeError):
            load_opta_event_data(3, self.f24_loc, pitch_dimensions=[10.0, 10.0])
        with self.assertRaises(TypeError):
            load_opta_event_data(
                self.f7_loc, [self.f24_loc], pitch_dimensions=[10.0, 10.0]
            )
        with self.assertRaises(ValueError):
            load_opta_event_data(
                self.f7_loc + ".json", self.f24_loc, pitch_dimensions=[10.0, 10.0]
            )
        with self.assertRaises(ValueError):
            load_opta_event_data(
                self.f7_loc, self.f24_loc[:-3], pitch_dimensions=[10.0, 10.0]
            )

    def test_load_metadata(self):
        metadata = _load_metadata(self.f7_loc, [10.0, 10.0])
        assert metadata == MD_OPTA

    def test_load_metadata_multiple_matches(self):
        metadata = _load_metadata(self.f7_loc_multiple_matches, [10.0, 10.0])
        # the second match metadata is dropped
        assert metadata == MD_OPTA

    def test_load_metadata_no_timestamps_and_date(self):
        with self.assertRaises(ValueError):
            _load_metadata(self.f7_loc_no_timestamps_and_date, [10.0, 10.0])

    def test_load_metadata_no_timestamps(self):
        with self.assertWarns(DataBallPyWarning):
            metadata = _load_metadata(self.f7_loc_no_timestamps, [10.0, 10.0])
        expected_md = MD_OPTA.copy()
        pf = expected_md.periods_frames
        pf.loc[0, "start_datetime_ed"] = pd.to_datetime(
            "20230122T111500+0000", utc=True
        )
        pf.loc[0, "end_datetime_ed"] = pd.to_datetime("20230122T120000+0000", utc=True)
        pf.loc[1, "start_datetime_ed"] = pd.to_datetime(
            "20230122T121500+0000", utc=True
        )
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
        pd.testing.assert_frame_equal(result, expected_result)

    def test_load_event_data(self):
        event_data, dbp_events = _load_event_data(
            self.f24_loc, away_team_id=194, country="Netherlands"
        )

        # player name is added in other function later in the pipeline
        expected_event_data = ED_OPTA.copy().drop("player_name", axis=1)

        # away team coordinates are changed later on in the pipeline
        expected_event_data.loc[[0, 2, 4, 6, 8, 10, 11], ["start_x", "start_y"]] *= -1

        # scaling of pitch dimension is done later on in the pipeline
        expected_event_data.loc[:, ["start_x", "start_y"]] = (
            expected_event_data.loc[:, ["start_x", "start_y"]] + 5
        ) * 10

        pd.testing.assert_frame_equal(event_data, expected_event_data)

        assert "shot_events" in dbp_events.keys()
        for key, event in dbp_events["shot_events"].items():
            assert key in SHOT_EVENTS_OPTA.keys()
            assert event == SHOT_EVENTS_OPTA[key]

        assert "dribble_events" in dbp_events.keys()
        for key, event in dbp_events["dribble_events"].items():
            assert key in DRIBBLE_EVENTS_OPTA.keys()
            assert event == DRIBBLE_EVENTS_OPTA[key]

        assert "pass_events" in dbp_events.keys()
        for key, event in dbp_events["pass_events"].items():
            assert key in PASS_EVENTS_OPTA.keys()
            assert event == PASS_EVENTS_OPTA[key]

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
            player_id=223345,
            event_id=2529877443,
            period_id=1,
            minutes=0,
            seconds=31,
            datetime=pd.to_datetime("2023-04-07T18:02:07.364", utc=True),
            start_x=-4.0279999,
            start_y=-29.852,
            pitch_size=[106.0, 68.0],
            team_side="home",
            _xt=-1.0,
            team_id=325,
            shot_outcome="goal",
            y_target=-1.991040,
            z_target=0.17568,
            body_part="right_foot",
            type_of_play="regular_play",
            first_touch=False,
            created_oppertunity="regular_play",
            related_event_id=MISSING_INT,
        )
        actual_output = _make_shot_event_instance(event, 111)
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

        dribble_event = _make_dribble_event_instance(event, away_team_id=325)

        expected_dribble_event = DribbleEvent(
            player_id=223345,
            event_id=2529877443,
            period_id=1,
            minutes=0,
            seconds=31,
            datetime=pd.to_datetime("2023-04-07T18:02:07.364", utc=True),
            start_x=4.0279999,
            start_y=29.852,
            pitch_size=[106.0, 68.0],
            team_side="away",
            _xt=-1.0,
            team_id=325,
            related_event_id=MISSING_INT,
            duel_type="offensive",
            outcome=True,
            has_opponent=True,
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
        pass_event = _make_pass_instance(event, away_team_id=1)

        expected_pass_event = PassEvent(
            event_id=1,
            period_id=1,
            minutes=0,
            seconds=0,
            datetime=pd.to_datetime("2022-01-01T00:00:00.000Z", utc=True),
            start_x=0.0,
            start_y=0.0,
            pitch_size=[106.0, 68.0],
            team_side="away",
            _xt=-1.0,
            team_id=1,
            outcome="results_in_shot",
            player_id=1,
            end_x=np.nan,
            end_y=np.nan,
            pass_type="long_ball",
            set_piece="no_set_piece",
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
        shot_events = copy.deepcopy(SHOT_EVENTS_OPTA)

        pass_events = copy.deepcopy(PASS_EVENTS_OPTA)
        event_data = ED_OPTA.copy()

        for event in shot_events.values():
            event.related_event_id = MISSING_INT

        res_passes = _update_pass_outcome(event_data, shot_events, pass_events)

        for key, event in res_passes.items():
            assert key in pass_events.keys()
            assert event == pass_events[key]

    def test_update_pass_outcome_pass_not_found(self):
        shot_events = copy.deepcopy(SHOT_EVENTS_OPTA)

        pass_events = copy.deepcopy(PASS_EVENTS_OPTA)
        event_data = ED_OPTA.copy()

        for event in shot_events.values():
            # none of the pass events have event_id 120
            event.related_event_id = 120

        res_passes = _update_pass_outcome(event_data, shot_events, pass_events)

        for key, event in res_passes.items():
            assert key in pass_events.keys()
            assert event == pass_events[key]

    def test_update_pass_outcome_multiple_options(self):
        shot_events = copy.deepcopy(SHOT_EVENTS_OPTA)

        pass_events = copy.deepcopy(PASS_EVENTS_OPTA)
        event_data = ED_OPTA.copy()

        for event in shot_events.values():
            if event.shot_outcome == "goal":
                event.related_event_id = 120

        event_data.loc[event_data["databallpy_event"] == "pass", "opta_id"] = 120

        # the pass event with event_id 120 has two options
        assert len(event_data.loc[event_data["opta_id"] == 120]) == 2

        res_passes = _update_pass_outcome(event_data, shot_events, pass_events)
        expected_passes = pass_events.copy()
        expected_passes[2499594243].outcome = "assist"
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
