import os
import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd

from databallpy.data_parsers.tracking_data_parsers.tracab_parser import (
    _get_metadata,
    _get_players_metadata_v1,
    _get_tracking_data_txt,
    fallback_fill_data_dict,
    load_sportec_open_tracking_data,
    load_tracab_tracking_data,
)
from databallpy.data_parsers.tracking_data_parsers.utils import _insert_missing_rows
from databallpy.utils.constants import MISSING_INT
from tests.expected_outcomes import (
    MD_TRACAB,
    SPORTEC_METADATA_TD,
    TD_TRACAB,
    TRACAB_SPORTEC_XML_TD,
)


class TestTracabParser(unittest.TestCase):
    def setUp(self):
        self.tracking_data_dat_loc = "tests/test_data/tracab_td_test.dat"
        self.metadata_json_loc = "tests/test_data/tracab_metadata_test.json"
        self.metadata_xml_loc = "tests/test_data/tracab_metadata_test.xml"
        self.metadata_sportec_loc = "tests/test_data/sportec_metadata.xml"
        self.td_sportec_loc = "tests/test_data/tracab_xml_td_test.xml"

    def test_load_tracab_tracking_data(self):
        tracking_data, metadata = load_tracab_tracking_data(
            self.tracking_data_dat_loc, self.metadata_json_loc, verbose=False
        )
        assert metadata == MD_TRACAB
        pd.testing.assert_frame_equal(tracking_data, TD_TRACAB)

        tracking_data, metadata = load_tracab_tracking_data(
            self.tracking_data_dat_loc, self.metadata_xml_loc, verbose=False
        )
        assert metadata == MD_TRACAB
        pd.testing.assert_frame_equal(tracking_data, TD_TRACAB)

    def test_load_tracab_tracking_data_errors(self):
        with self.assertRaises(ValueError):
            load_tracab_tracking_data(
                self.tracking_data_dat_loc[:-3], self.metadata_json_loc, verbose=False
            )

        with self.assertRaises(FileNotFoundError):
            load_tracab_tracking_data(
                self.tracking_data_dat_loc,
                self.metadata_json_loc + ".json",
                verbose=False,
            )

        with self.assertRaises(ValueError):
            load_tracab_tracking_data(
                "tests/test_data/tracab_td_wrong_format.json", self.metadata_json_loc
            )

        with self.assertRaises(ValueError):
            load_tracab_tracking_data(
                self.tracking_data_dat_loc[:-3], self.metadata_xml_loc, verbose=False
            )

        with self.assertRaises(FileNotFoundError):
            load_tracab_tracking_data(
                self.tracking_data_dat_loc,
                self.metadata_xml_loc + ".sml",
                verbose=False,
            )
        with self.assertRaises(ValueError):
            load_tracab_tracking_data(
                "tests/test_data/tracab_td_wrong_format.json", self.metadata_xml_loc
            )

    def test_get_metadata(self):
        metadata = _get_metadata(self.metadata_json_loc)
        expected_metadata = MD_TRACAB.copy()
        expected_metadata.periods_changed_playing_direction = None
        self.assertEqual(metadata, expected_metadata)

        metadata = _get_metadata(self.metadata_xml_loc)
        expected_metadata = MD_TRACAB.copy()
        expected_metadata.periods_changed_playing_direction = None
        self.assertEqual(metadata, expected_metadata)

        metadata = _get_metadata(self.metadata_sportec_loc)
        expected_metadata = SPORTEC_METADATA_TD.copy()
        expected_metadata.periods_frames["start_datetime_td"] = pd.to_datetime("NaT")
        expected_metadata.periods_frames["end_datetime_td"] = pd.to_datetime("NaT")
        expected_metadata.periods_frames["start_frame"] = MISSING_INT
        expected_metadata.periods_frames["end_frame"] = MISSING_INT
        expected_metadata.periods_changed_playing_direction = None
        expected_metadata.frame_rate = MISSING_INT
        self.assertEqual(metadata, expected_metadata)

        with self.assertRaises(ValueError):
            _get_metadata("tests/test_data/sportec_metadata_wrong.xml")

    def test_get_tracking_data_xml(self):
        track_dat, meta_dat = load_tracab_tracking_data(
            self.td_sportec_loc, self.metadata_sportec_loc
        )

        self.assertEqual(len(track_dat), 100003 - 10000)
        track_dat = track_dat[track_dat["period_id"] != MISSING_INT].reset_index(
            drop=True
        )
        pd.testing.assert_frame_equal(track_dat, TRACAB_SPORTEC_XML_TD)

        exp_metadata = SPORTEC_METADATA_TD.copy()
        exp_metadata.frame_rate = 1.0
        self.assertEqual(meta_dat, exp_metadata)

    def test_get_tracking_data_txt(self):
        tracking_data = _get_tracking_data_txt(self.tracking_data_dat_loc, verbose=False)
        expected_td = TD_TRACAB.drop(["matchtime_td", "period_id", "datetime"], axis=1)
        pd.testing.assert_frame_equal(tracking_data, expected_td)

    def test_fallback_fill_data_dict(self):
        data_dict = fallback_fill_data_dict(self.tracking_data_dat_loc, verbose=False)
        df = pd.DataFrame(data_dict)
        mask = df.columns.str.contains("_x|_y|_z")
        df.loc[:, mask] = np.round(df.loc[:, mask] / 100, 3)
        df = _insert_missing_rows(df, "frame")
        first_cols = [
            "frame",
            "ball_x",
            "ball_y",
            "ball_z",
            "ball_status",
            "ball_possession",
        ]
        other_cols = [x for x in df.sort_index(axis=1).columns if x not in first_cols]
        expected_td = TD_TRACAB.drop(["matchtime_td", "period_id", "datetime"], axis=1)
        pd.testing.assert_frame_equal(df[[*first_cols, *other_cols]], expected_td)

    @patch("databallpy.data_parsers.tracking_data_parsers.tracab_parser.requests.get")
    @patch(
        "databallpy.data_parsers.tracking_data_parsers.tracab_parser.requests.Session"
    )
    @patch("databallpy.data_parsers.tracking_data_parsers.tracab_parser.os.makedirs")
    @patch("databallpy.data_parsers.tracking_data_parsers.tracab_parser.os.path.exists")
    @patch(
        "databallpy.data_parsers.tracking_data_parsers.tracab_parser.open",
        new_callable=mock_open,
    )
    @patch("databallpy.data_parsers.tracking_data_parsers.tracab_parser.os.rename")
    @patch(
        "databallpy.data_parsers.tracking_data_parsers.tracab_parser.load_tracab_tracking_data"
    )
    def test_load_sportec_open_tracking_data(
        self,
        mock_load_tracab_tracking_data,
        mock_rename,
        mock_open,
        mock_exists,
        mock_makedirs,
        mock_session,
        mock_requests_get,
    ):
        # Setup mock responses
        mock_exists.side_effect = [
            False,
            False,
            True,
            True,
        ]  # metadata.xml does not exist, tracking_data.xml does not exist, then both exist
        mock_requests_get.return_value = MagicMock(content=b"mock content")
        mock_session.return_value.get.return_value = MagicMock(
            headers={"content-length": "1024"},
            iter_content=lambda chunk_size: [b"mock content"],
        )
        mock_load_tracab_tracking_data.return_value = (pd.DataFrame(), "mock_metadata")
        mock_rename.return_value = "triggered"

        match_id = "J03WMX"
        verbose = True
        expected_metadata_path = os.path.join(
            os.getcwd(), "datasets", "IDSSE", match_id, "metadata.xml"
        )
        expected_tracking_data_path = os.path.join(
            os.getcwd(), "datasets", "IDSSE", match_id, "tracking_data.xml"
        )

        # Call the function
        result = load_sportec_open_tracking_data(match_id, verbose)

        # Verify the function calls
        mock_makedirs.assert_called_once_with(
            os.path.join(os.getcwd(), "datasets", "IDSSE", match_id), exist_ok=True
        )
        self.assertEqual(mock_requests_get.call_count, 1)
        self.assertEqual(mock_session.return_value.get.call_count, 1)
        mock_open.assert_any_call(expected_metadata_path, "wb")
        mock_open.assert_any_call(
            os.path.join(
                os.getcwd(), "datasets", "IDSSE", match_id, "tracking_data_temp.xml"
            ),
            "wb",
        )
        self.assertEqual(mock_rename.call_count, 1)
        mock_load_tracab_tracking_data.assert_called_once_with(
            expected_tracking_data_path, expected_metadata_path, verbose=verbose
        )

        # Verify the return value
        pd.testing.assert_frame_equal(result[0], pd.DataFrame())
        self.assertEqual(result[1], "mock_metadata")

    def test_get_players_metadata(self):
        input_players_info = [
            {
                "PlayerId": "1234",
                "FirstName": "Bart",
                "LastName": "Bakker",
                "JerseyNo": "4",
                "StartFrameCount": "1212",
                "EndFrameCount": "2323",
            },
            {
                "PlayerId": "1235",
                "FirstName": "Bram",
                "LastName": "Slager",
                "JerseyNo": "3",
                "StartFrameCount": "1218",
                "EndFrameCount": "2327",
            },
        ]

        expected_df_players = pd.DataFrame(
            {
                "id": [1234, 1235],
                "full_name": ["Bart Bakker", "Bram Slager"],
                "shirt_num": [4, 3],
                "start_frame": [1212, 1218],
                "end_frame": [2323, 2327],
            }
        )
        df_players = _get_players_metadata_v1(input_players_info)
        pd.testing.assert_frame_equal(df_players, expected_df_players)
