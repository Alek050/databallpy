import unittest

import pandas as pd

from databallpy.load_data.event_data.base_event import BaseEvent


class TestBaseEvent(unittest.TestCase):
    def setUp(self):
        self.base_event = BaseEvent(
            event_id=1,
            period_id=1,
            minutes=1,
            seconds=10,
            datetime=pd.to_datetime("2020-01-01 00:00:00"),
            start_x=10.0,
            start_y=11.0,
            team_id=1,
        )

    def test_base_event__eq__(self):
        assert self.base_event == self.base_event

        assert self.base_event != 1
        assert self.base_event != BaseEvent(
            event_id=2,
            period_id=1,
            minutes=1,
            seconds=10,
            datetime=pd.to_datetime("2020-01-01 00:00:00"),
            start_x=10.0,
            start_y=11.0,
            team_id=1,
        )

    def test_base_event_post_init(self):
        # event_id
        with self.assertRaises(TypeError):
            BaseEvent(
                event_id=1.3,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
            )
        # period_id
        with self.assertRaises(TypeError):
            BaseEvent(
                event_id=1,
                period_id="1.3",
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
            )
        # minutes
        with self.assertRaises(TypeError):
            BaseEvent(
                event_id=1,
                period_id=1,
                minutes="1.3",
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
            )
        # seconds
        with self.assertRaises(TypeError):
            BaseEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=[10],
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
            )
        # datetime
        with self.assertRaises(TypeError):
            BaseEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime="2020-01-01 00:00:00",
                start_x=10.0,
                start_y=11.0,
                team_id=1,
            )
        # start_x
        with self.assertRaises(TypeError):
            BaseEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10,
                start_y=11.0,
                team_id=1,
            )
        # start_y
        with self.assertRaises(TypeError):
            BaseEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y={11.0},
                team_id=1,
            )
        # team_id
        with self.assertRaises(TypeError):
            BaseEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id="1",
            )
