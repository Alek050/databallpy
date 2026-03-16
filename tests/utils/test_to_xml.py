import unittest
import xml.etree.ElementTree as ET

from databallpy.utils.to_xml import Event, LabelDict, _color_for_code, events_to_xml


def _parse(xml_str: str) -> ET.Element:
    if xml_str.startswith("<?xml"):
        xml_str = xml_str[xml_str.index("?>") + 2 :].lstrip()
    return ET.fromstring(xml_str)


class TestColorForCode(unittest.TestCase):
    def test_returns_tuple_of_three_ints(self):
        result = _color_for_code("pass")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        for v in result:
            self.assertIsInstance(v, int)

    def test_values_in_range_128_255(self):
        for code in ["pass", "shot", "1H", "ET1", ""]:
            r, g, b = _color_for_code(code)
            for v in (r, g, b):
                self.assertGreaterEqual(v, 128)
                self.assertLessEqual(v, 255)

    def test_deterministic(self):
        self.assertEqual(_color_for_code("pass"), _color_for_code("pass"))

    def test_different_codes_differ(self):
        self.assertNotEqual(_color_for_code("pass"), _color_for_code("shot"))


class TestEventsToXml(unittest.TestCase):
    def _make_events(self):
        return {
            "0": Event(
                id="0",
                code="Pass",
                start_t=10.0,
                end_t=15.0,
                labels=[
                    LabelDict(group="Player", name="Alice"),
                    LabelDict(group="Team", name="home"),
                ],
            ),
            "1": Event(
                id="1",
                code="Shot",
                start_t=90.0,
                end_t=95.0,
                labels=[],
            ),
        }

    def test_returns_string(self):
        result = events_to_xml(self._make_events())
        self.assertIsInstance(result, str)

    def test_xml_declaration_present(self):
        result = events_to_xml(self._make_events())
        self.assertIn("<?xml", result)

    def test_root_element_is_file(self):
        root = _parse(events_to_xml(self._make_events()))
        self.assertEqual(root.tag, "file")

    def test_sort_info_present(self):
        root = _parse(events_to_xml(self._make_events()))
        sort_info = root.find("SORT_INFO")
        self.assertIsNotNone(sort_info)
        self.assertEqual(sort_info.find("sort_type").text, "sort order")

    def test_instance_count(self):
        root = _parse(events_to_xml(self._make_events()))
        instances = root.findall(".//instance")
        self.assertEqual(len(instances), 2)

    def test_instance_fields(self):
        root = _parse(events_to_xml(self._make_events()))
        instances = {inst.find("ID").text: inst for inst in root.findall(".//instance")}
        inst = instances["0"]
        self.assertEqual(inst.find("ID").text, "0")
        self.assertEqual(inst.find("code").text, "Pass")
        self.assertEqual(inst.find("start").text, "10.00")
        self.assertEqual(inst.find("end").text, "15.00")

    def test_labels_serialised(self):
        root = _parse(events_to_xml(self._make_events()))
        instances = {inst.find("ID").text: inst for inst in root.findall(".//instance")}
        labels = instances["0"].findall("label")
        self.assertEqual(len(labels), 2)
        self.assertEqual(labels[0].find("group").text, "Player")
        self.assertEqual(labels[0].find("text").text, "Alice")
        self.assertEqual(labels[1].find("group").text, "Team")
        self.assertEqual(labels[1].find("text").text, "home")

    def test_no_labels(self):
        root = _parse(events_to_xml(self._make_events()))
        instances = {inst.find("ID").text: inst for inst in root.findall(".//instance")}
        self.assertEqual(len(instances["1"].findall("label")), 0)

    def test_rows_contain_distinct_codes(self):
        root = _parse(events_to_xml(self._make_events()))
        rows = root.findall(".//row")
        codes = {row.find("code").text for row in rows}
        self.assertEqual(codes, {"Pass", "Shot"})

    def test_rows_sorted_alphabetically(self):
        root = _parse(events_to_xml(self._make_events()))
        row_codes = [row.find("code").text for row in root.findall(".//row")]
        self.assertEqual(row_codes, sorted(row_codes))

    def test_rows_have_color_and_sort_order(self):
        root = _parse(events_to_xml(self._make_events()))
        for i, row in enumerate(root.findall(".//row")):
            self.assertIsNotNone(row.find("R"))
            self.assertIsNotNone(row.find("G"))
            self.assertIsNotNone(row.find("B"))
            self.assertEqual(row.find("sort_order").text, str(i))

    def test_same_code_appears_once_in_rows(self):
        events = {
            "0": Event(id="0", code="Pass", start_t=0.0, end_t=5.0, labels=[]),
            "1": Event(id="1", code="Pass", start_t=10.0, end_t=15.0, labels=[]),
        }
        root = _parse(events_to_xml(events))
        row_codes = [row.find("code").text for row in root.findall(".//row")]
        self.assertEqual(row_codes.count("Pass"), 1)

    def test_time_decimals(self):
        events = {"0": Event(id="0", code="X", start_t=1.0, end_t=2.0, labels=[])}
        root = _parse(events_to_xml(events, time_decimals=4))
        inst = root.find(".//instance")
        self.assertEqual(inst.find("start").text, "1.0000")
        self.assertEqual(inst.find("end").text, "2.0000")

    def test_end_t_equals_start_t_is_valid(self):
        events = {"0": Event(id="0", code="X", start_t=5.0, end_t=5.0, labels=[])}
        result = events_to_xml(events)
        self.assertIsInstance(result, str)

    def test_end_t_less_than_start_t_raises(self):
        events = {"0": Event(id="0", code="X", start_t=10.0, end_t=5.0, labels=[])}
        with self.assertRaises(ValueError):
            events_to_xml(events)

    def test_empty_events_dict(self):
        result = events_to_xml({})
        root = _parse(result)
        self.assertEqual(len(root.findall(".//instance")), 0)
        self.assertEqual(len(root.findall(".//row")), 0)

    def test_color_values_are_integers_in_xml(self):
        events = {"0": Event(id="0", code="Pass", start_t=0.0, end_t=1.0, labels=[])}
        root = _parse(events_to_xml(events))
        row = root.find(".//row")
        for tag in ("R", "G", "B"):
            int(row.find(tag).text)  # should not raise


if __name__ == "__main__":
    unittest.main()
