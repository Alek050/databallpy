import unittest

import pandas as pd

from databallpy.load_data.event_data._normalize_playing_direction_events import (
    _normalize_playing_direction_events,
)


class TestNormalizePlayingDirection(unittest.TestCase):
    def setUp(self) -> None:
        self.event_data = pd.DataFrame(
            {
                "event": [
                    "pass",
                    "pass",
                    "shot",
                    "goal",
                    "miss",
                    "goal",
                    "shot",
                    "miss",
                ],
                "period_id": [1, 1, 1, 1, 2, 2, 2, 2],
                "start_x": [12, 40, -30, -30, 15, 16, 20, 25],
                "start_y": [1, 1, 1, 1, 1, 1, 1, 1],
                "team_id": [12, 33, 33, 12, 33, 33, 12, 12],
            }
        )

    def test_normalize_playing_direction(self):
        expected = pd.DataFrame(
            {
                "event": [
                    "pass",
                    "pass",
                    "shot",
                    "goal",
                    "miss",
                    "goal",
                    "shot",
                    "miss",
                ],
                "period_id": [1, 1, 1, 1, 2, 2, 2, 2],
                "start_x": [12, -40, 30, -30, 15, 16, -20, -25],
                "start_y": [1, -1, -1, 1, 1, 1, -1, -1],
                "team_id": [12, 33, 33, 12, 33, 33, 12, 12],
            }
        )

        res = _normalize_playing_direction_events(self.event_data, 33, 12)

        pd.testing.assert_frame_equal(res, expected)
