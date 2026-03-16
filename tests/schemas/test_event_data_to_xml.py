import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import pandas as pd

from databallpy import Event, LabelDict, events_to_xml
from databallpy.schemas.event_data import EventData

_T0 = pd.Timestamp("2023-01-14 12:00:00", tz="UTC")


def make_event_data():
    """Minimal 4-row EventData spanning 2 periods.

    Datetimes are anchored at _T0 (first pass in H1 = 0 s):
      event 1 (H1 pass,  player 10): _T0          →  t =    0 s
      event 2 (H1 shot,  player 20): _T0 + 2640 s →  t = 2640 s
      event 3 (H2 pass,  player 10): _T0 + 3600 s →  t = 3600 s  (~60 min incl. break)
      event 4 (H2 tackle,player 20): _T0 + 5455 s →  t = 5455 s
    """
    data = {
        "event_id": [1, 2, 3, 4],
        "databallpy_event": ["pass", "shot", "pass", None],
        "period_id": [1, 1, 2, 2],
        "minutes": [2, 44, 47, 89],
        "seconds": [30.0, 0.0, 5.0, 55.0],
        "player_id": [10, 20, 10, 20],
        "player_name": ["Jan", "Piet", "Jan", "Piet"],
        "team_id": [1, 2, 1, 2],
        "team_name": ["Ajax", "PSV", "Ajax", "PSV"],
        "is_successful": pd.array([True, False, True, None], dtype=pd.BooleanDtype()),
        "start_x": [0.0, 10.0, -5.0, 20.0],
        "start_y": [0.0, 5.0, -3.0, 10.0],
        "datetime": [
            _T0,
            _T0 + pd.Timedelta(seconds=2640),
            _T0 + pd.Timedelta(seconds=3600),
            _T0 + pd.Timedelta(seconds=5455),
        ],
        "original_event_id": [100, 200, 300, 400],
        "original_event": ["pass", "shot", "pass", "tackle"],
    }
    return EventData(data)


class TestEventDataToXml(unittest.TestCase):
    def setUp(self):
        self.ed = make_event_data()

    def _parse(self, xml_str):
        # Strip the XML declaration before parsing
        if xml_str.startswith("<?xml"):
            xml_str = xml_str[xml_str.index("?>") + 2 :].lstrip()
        return ET.fromstring(xml_str)

    def _instance_codes(self, root):
        return [el.text for el in root.findall(".//instance/code")]

    def test_to_xml_no_filters(self):
        xml_str = self.ed.to_video_analysis_xml()
        root = self._parse(xml_str)
        instances = root.findall(".//instance")
        # 4 events + 2 period markers
        self.assertEqual(len(instances), 6)
        codes = self._instance_codes(root)
        self.assertIn("1H", codes)
        self.assertIn("2H", codes)

    def test_to_xml_team_filter(self):
        xml_str = self.ed.to_video_analysis_xml(team_id=1, tag_period_starts=False)
        root = self._parse(xml_str)
        instances = root.findall(".//instance")
        self.assertEqual(len(instances), 2)

    def test_to_xml_team_filter_list(self):
        xml_str = self.ed.to_video_analysis_xml(team_id=[1, 2], tag_period_starts=False)
        root = self._parse(xml_str)
        instances = root.findall(".//instance")
        self.assertEqual(len(instances), 4)

    def test_to_xml_player_filter(self):
        xml_str = self.ed.to_video_analysis_xml(player_id=20, tag_period_starts=False)
        root = self._parse(xml_str)
        instances = root.findall(".//instance")
        self.assertEqual(len(instances), 2)

    def test_to_xml_minute_range(self):
        xml_str = self.ed.to_video_analysis_xml(
            min_minute=40, max_minute=50, tag_period_starts=False
        )
        root = self._parse(xml_str)
        instances = root.findall(".//instance")
        self.assertEqual(len(instances), 2)

    def test_to_xml_databallpy_events_filter(self):
        xml_str = self.ed.to_video_analysis_xml(
            databallpy_events=["shot"], tag_period_starts=False
        )
        root = self._parse(xml_str)
        instances = root.findall(".//instance")
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].find("code").text, "shot")

    def test_to_xml_is_successful_filter(self):
        xml_str = self.ed.to_video_analysis_xml(
            is_successful=True, tag_period_starts=False
        )
        root = self._parse(xml_str)
        instances = root.findall(".//instance")
        self.assertEqual(len(instances), 2)

    def test_to_xml_time_window(self):
        xml_str = self.ed.to_video_analysis_xml(
            player_id=10,
            tag_period_starts=False,
            before_seconds=5.0,
            after_seconds=10.0,
        )
        root = self._parse(xml_str)
        instances = root.findall(".//instance")
        self.assertEqual(len(instances), 2)
        by_id = {inst.find("ID").text: inst for inst in instances}
        for inst in instances:
            end = float(inst.find("end").text)
            start = float(inst.find("start").text)
            self.assertGreater(end, start)
        # Event 1 (H1 anchor, t=0): start = 0 - 5 = -5, end = 0 + 10 = 10 → window = 15 s
        start1 = float(by_id["1"].find("start").text)
        end1 = float(by_id["1"].find("end").text)
        self.assertAlmostEqual(end1 - start1, 15.0, places=1)
        # Event 3 (H2, t=3600 s): start = 3600 - 5 = 3595, end = 3600 + 10 = 3610
        start3 = float(by_id["3"].find("start").text)
        end3 = float(by_id["3"].find("end").text)
        self.assertAlmostEqual(start3, 3595.0, places=2)
        self.assertAlmostEqual(end3, 3610.0, places=1)

    def test_to_xml_period_offsets(self):
        """H2 events must be positioned well after H1 on the video timeline."""
        xml_str = self.ed.to_video_analysis_xml(
            player_id=10,
            tag_period_starts=False,
            before_seconds=0.0,
            after_seconds=5.0,
        )
        root = self._parse(xml_str)
        # Event 1 (H1 anchor): t = 0 s
        # Event 3 (H2, _T0 + 3600 s): t = 3600 s — H2 does NOT reset to 0
        instances = root.findall(".//instance")
        starts = {
            inst.find("ID").text: float(inst.find("start").text) for inst in instances
        }
        self.assertAlmostEqual(starts["1"], 0.0, places=2)
        self.assertAlmostEqual(starts["3"], 3600.0, places=2)
        self.assertGreater(starts["3"], starts["1"])

    def test_to_xml_no_period_starts(self):
        xml_str = self.ed.to_video_analysis_xml(tag_period_starts=False)
        root = self._parse(xml_str)
        codes = self._instance_codes(root)
        self.assertNotIn("1H", codes)
        self.assertNotIn("2H", codes)

    def test_to_xml_period_marker_ids(self):
        xml_str = self.ed.to_video_analysis_xml()
        root = self._parse(xml_str)
        ids = [el.text for el in root.findall(".//instance/ID")]
        self.assertIn("p1", ids)
        self.assertIn("p2", ids)

    def test_to_xml_null_databallpy_event_falls_back_to_original(self):
        """Row 4 has null databallpy_event; code should fall back to original_event."""
        xml_str = self.ed.to_video_analysis_xml(
            player_id=20, is_successful=None, tag_period_starts=False
        )
        root = self._parse(xml_str)
        instances = root.findall(".//instance")
        codes = {inst.find("ID").text: inst.find("code").text for inst in instances}
        self.assertEqual(codes["4"], "tackle")

    def test_to_xml_invalid_before_seconds(self):
        with self.assertRaises(ValueError):
            self.ed.to_video_analysis_xml(before_seconds=-1.0)

    def test_to_xml_invalid_after_seconds(self):
        with self.assertRaises(ValueError):
            self.ed.to_video_analysis_xml(after_seconds=0.0)

    def test_to_xml_invalid_code_column(self):
        with self.assertRaises(ValueError):
            self.ed.to_video_analysis_xml(code_column="nonexistent_column")

    def test_to_xml_invalid_minute_range(self):
        with self.assertRaises(ValueError):
            self.ed.to_video_analysis_xml(min_minute=50, max_minute=40)

    def test_to_xml_unknown_databallpy_events_warns(self):
        with self.assertWarns(UserWarning):
            self.ed.to_video_analysis_xml(databallpy_events=["unknown_event"])

    def test_to_xml_custom_code_column(self):
        xml_str = self.ed.to_video_analysis_xml(
            code_column="original_event", tag_period_starts=False
        )
        root = self._parse(xml_str)
        codes = self._instance_codes(root)
        self.assertIn("pass", codes)
        self.assertIn("shot", codes)
        self.assertIn("tackle", codes)

    def test_to_xml_returns_string(self):
        result = self.ed.to_xml()
        self.assertIsInstance(result, str)
        self.assertIn("<?xml", result)

    def test_to_xml_writes_file_and_returns_path(self):
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            path = f.name
        try:
            result = self.ed.to_video_analysis_xml(output=path)
            self.assertEqual(result, path)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("<?xml", content)
        finally:
            os.unlink(path)

    def test_events_to_xml_user_defined(self):
        """Test calling events_to_xml directly with a hand-built dict."""
        my_events = {
            "0": Event(
                id="0",
                code="Press",
                start_t=120.0,
                end_t=125.0,
                labels=[LabelDict(group="Player", name="De Bruyne")],
            ),
            "1": Event(
                id="1",
                code="Shot",
                start_t=200.0,
                end_t=205.0,
                labels=[],
            ),
        }
        xml_str = events_to_xml(my_events)
        self.assertIsInstance(xml_str, str)
        root = ET.fromstring(xml_str[xml_str.index("?>") + 2 :].lstrip())
        instances = root.findall(".//instance")
        self.assertEqual(len(instances), 2)
        codes = [inst.find("code").text for inst in instances]
        self.assertIn("Press", codes)
        self.assertIn("Shot", codes)

    def test_team_name_label_used(self):
        """Team label should show team_name when the column is present."""
        xml_str = self.ed.to_video_analysis_xml(tag_period_starts=False)
        root = self._parse(xml_str)
        team_labels = [
            el.find("text").text
            for el in root.findall(".//instance/label")
            if el.find("group") is not None and el.find("group").text == "Team"
        ]
        self.assertIn("Ajax", team_labels)
        self.assertIn("PSV", team_labels)
        self.assertNotIn("1", team_labels)
        self.assertNotIn("2", team_labels)

    def test_team_name_fallback_to_id(self):
        """Team label should fall back to team_id when team_name column is absent."""
        data = make_event_data().drop(columns=["team_name"])
        xml_str = data.to_video_analysis_xml(tag_period_starts=False)
        root = self._parse(xml_str)
        team_labels = [
            el.find("text").text
            for el in root.findall(".//instance/label")
            if el.find("group") is not None and el.find("group").text == "Team"
        ]
        self.assertIn("1", team_labels)
        self.assertIn("2", team_labels)


if __name__ == "__main__":
    unittest.main()
