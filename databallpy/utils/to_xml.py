from __future__ import annotations

import hashlib
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from typing import Dict, List, TypedDict


class LabelDict(TypedDict):
    group: str
    name: str


class Event(TypedDict):
    id: str
    code: str
    start_t: float
    end_t: float
    labels: List[LabelDict]


def _color_for_code(code: str) -> tuple[int, int, int]:
    """
    Deterministically map a code string to a hex color (#RRGGBB).

    This is a simple approach based on a hash; adjust if you want a fixed palette
    or specific colors per code.
    """
    h = hashlib.sha1(code.encode("utf-8")).digest()
    r, g, b = h[0], h[1], h[2]
    r = 128 + (r // 2)
    g = 128 + (g // 2)
    b = 128 + (b // 2)
    return r, g, b


def events_to_xml(
    events: Dict[str, Event],
    *,
    time_decimals: int = 2,
) -> str:
    """
    Convert a dict of events into a XML string.

    Structure::

        <?xml version="1.0" encoding="UTF-8"?>
          <SORT_INFO>...</SORT_INFO>
          <ALL_INSTANCES>
            <instance>
              <ID>...</ID>
              <start>...</start>
              <end>...</end>
              <code>...</code>
              <label>
                <group>...</group>
                <text>...</text>
              </label>
              ...
            </instance>
            ...
          </ALL_INSTANCES>
          <ROWS>
            <row>
              <code>...</code>
              <r>int</r>
              <g>int</g>
              <b>int</b>
              <sort_order>int</sort_order>
            </row>
            ...
          </ROWS>
        </file>

    """
    root = ET.Element("file")

    sort_info_el = ET.SubElement(root, "SORT_INFO")
    sort_type_el = ET.SubElement(sort_info_el, "sort_type")
    sort_type_el.text = "sort order"

    all_instances_el = ET.SubElement(root, "ALL_INSTANCES")
    distinct_codes: set[str] = set()

    for key, data in events.items():
        event_id = str(data["id"])
        code = data["code"]
        start_t = float(data["start_t"])
        end_t = float(data["end_t"])
        labels = data.get("labels", [])

        if end_t < start_t:
            raise ValueError(
                f"Event with key '{key}' has end_t < start_t ({end_t} < {start_t})"
            )

        start_str = f"{start_t:.{time_decimals}f}"
        end_str = f"{end_t:.{time_decimals}f}"

        instance_el = ET.SubElement(all_instances_el, "instance")

        id_el = ET.SubElement(instance_el, "ID")
        id_el.text = event_id

        start_el = ET.SubElement(instance_el, "start")
        start_el.text = start_str

        end_el = ET.SubElement(instance_el, "end")
        end_el.text = end_str

        code_el = ET.SubElement(instance_el, "code")
        code_el.text = code

        distinct_codes.add(code)

        for label in labels:
            group = label["group"]
            name = label["name"]

            label_el = ET.SubElement(instance_el, "label")

            group_el = ET.SubElement(label_el, "group")
            group_el.text = str(group)

            text_el = ET.SubElement(label_el, "text")
            text_el.text = str(name)

    rows_el = ET.SubElement(root, "ROWS")
    for sort_order, code in enumerate(sorted(distinct_codes)):
        row_el = ET.SubElement(rows_el, "row")

        code_el = ET.SubElement(row_el, "code")
        code_el.text = code

        r, g, b = _color_for_code(code)

        r_el = ET.SubElement(row_el, "R")
        r_el.text = str(int(r / 255 * 99999))

        g_el = ET.SubElement(row_el, "G")
        g_el.text = str(int(g / 255 * 99999))

        b_el = ET.SubElement(row_el, "B")
        b_el.text = str(int(b / 255 * 99999))

        sort_el = ET.SubElement(row_el, "sort_order")
        sort_el.text = str(sort_order)

    rough_string = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(rough_string)
    pretty_xml = parsed.toprettyxml(indent="\t", encoding="utf-8")

    return pretty_xml.decode("utf-8")
