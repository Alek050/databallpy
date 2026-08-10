import unittest

import numpy as np
import pandas as pd

from databallpy.data_parsers.tracking_data_parsers.utils._get_frame_selection import (
    _get_frame_selection,
    _get_period_start_frames,
)
from databallpy.utils.constants import MISSING_INT


class TestGetFrameSelection(unittest.TestCase):
    def setUp(self):
        self.periods_frames = pd.DataFrame(
            {
                "period_id": [1, 2, 3, 4, 5],
                "start_frame": [2, 5, MISSING_INT, MISSING_INT, MISSING_INT],
                "end_frame": [3, 6, MISSING_INT, MISSING_INT, MISSING_INT],
            }
        )

    def test_get_period_start_frames(self):
        assert _get_period_start_frames(self.periods_frames) == {2, 5}

    def test_get_period_start_frames_no_periods(self):
        periods_frames = self.periods_frames.copy()
        periods_frames["start_frame"] = MISSING_INT
        assert _get_period_start_frames(periods_frames) == set()

    def test_get_frame_selection_no_selection(self):
        assert _get_frame_selection(self.periods_frames, None, None) is None

    def test_get_frame_selection_period_id(self):
        assert _get_frame_selection(self.periods_frames, 1, None) == (2, 3)
        assert _get_frame_selection(self.periods_frames, np.int64(2), None) == (5, 6)
        assert _get_frame_selection(self.periods_frames, [1, 2], None) == (2, 6)

    def test_get_frame_selection_period_id_wrong_type(self):
        with self.assertRaises(TypeError):
            _get_frame_selection(self.periods_frames, "1", None)
        with self.assertRaises(TypeError):
            _get_frame_selection(self.periods_frames, [1, "2"], None)

    def test_get_frame_selection_period_id_not_available(self):
        with self.assertRaises(ValueError):
            _get_frame_selection(self.periods_frames, 3, None)
        with self.assertRaises(ValueError):
            _get_frame_selection(self.periods_frames, [1, 4], None)

    def test_get_frame_selection_frames(self):
        assert _get_frame_selection(self.periods_frames, None, (3, 5)) == (3, 5)
        assert _get_frame_selection(self.periods_frames, None, [3, 5]) == (3, 5)
        assert _get_frame_selection(self.periods_frames, None, (-10, 100)) == (2, 6)
        assert _get_frame_selection(
            self.periods_frames, None, (np.int64(4), np.int64(6))
        ) == (4, 6)

    def test_get_frame_selection_frames_wrong_type(self):
        with self.assertRaises(TypeError):
            _get_frame_selection(self.periods_frames, None, 3)
        with self.assertRaises(TypeError):
            _get_frame_selection(self.periods_frames, None, (1, 2, 3))
        with self.assertRaises(TypeError):
            _get_frame_selection(self.periods_frames, None, (1.0, 2.0))

    def test_get_frame_selection_frames_wrong_order(self):
        with self.assertRaises(ValueError):
            _get_frame_selection(self.periods_frames, None, (5, 3))

    def test_get_frame_selection_period_id_and_frames(self):
        assert _get_frame_selection(self.periods_frames, 1, (3, 10)) == (3, 3)

    def test_get_frame_selection_no_frames_left(self):
        with self.assertRaises(ValueError):
            _get_frame_selection(self.periods_frames, 1, (5, 6))
