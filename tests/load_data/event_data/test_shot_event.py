import unittest

import pandas as pd

from databallpy.load_data.event_data.shot_event import ShotEvent


class TestShotEvent(unittest.TestCase):
    def setUp(self) -> None:
        self.shot_event = ShotEvent(
            event_id=2512690515,
            period_id=1,
            minutes=9,
            seconds=17,
            datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
            start_x=50.0,
            start_y=20.0,
            z_target=15.0,
            y_target=3.5,
            player_id=45849,
            shot_outcome="own_goal",
            body_part="head",
            type_of_play="corner_kick",
            first_touch=False,
            created_oppertunity="regular_play",
            related_event=123,
        )

    def test_shot_event_eq(self):
        assert self.shot_event == self.shot_event

        shot_event_changed_event_attr = self.shot_event.copy()
        shot_event_changed_event_attr.event_id = 123
        assert self.shot_event != shot_event_changed_event_attr

        shot_event_changed_shot_attr = self.shot_event.copy()
        shot_event_changed_shot_attr.shot_outcome = "goal"
        assert self.shot_event != shot_event_changed_shot_attr

        assert self.shot_event != 1

    def test_shot_event_copy(self):
        shot_event_copy = self.shot_event.copy()
        assert shot_event_copy == self.shot_event
