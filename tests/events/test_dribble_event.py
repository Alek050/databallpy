import unittest

import pandas as pd

from databallpy.events.dribble_event import DribbleEvent


class TestDribbleEvent(unittest.TestCase):
    def setUp(self) -> None:
        self.dribble_event = DribbleEvent(
            event_id=2512690515,
            period_id=1,
            minutes=9,
            seconds=17,
            datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
            start_x=50.0,
            start_y=20.0,
            pitch_size=[105.0, 68.0],
            team_side="home",
            _xt=0.02,
            team_id=123,
            player_id=45849,
            related_event_id=123,
            duel_type="offensive",
            outcome=True,
        )

    def test_post_init(self):
        # super() call
        with self.assertRaises(TypeError):
            DribbleEvent(
                event_id="2512690515",
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=[105.0, 68.0],
                team_side="home",
                _xt=0.02,
                team_id=123,
                player_id=45849,
                related_event_id=123,
                duel_type="offensive",
                outcome=True,
            )

        # player_id
        with self.assertRaises(TypeError):
            DribbleEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=[105.0, 68.0],
                team_side="home",
                _xt=0.02,
                team_id=123,
                player_id=[22],
                related_event_id=123,
                duel_type="offensive",
                outcome=True,
            )

        # related_event_id
        with self.assertRaises(TypeError):
            DribbleEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=[105.0, 68.0],
                team_side="home",
                _xt=0.02,
                team_id=123,
                player_id=45849,
                related_event_id=[123],
                duel_type="offensive",
                outcome=True,
            )

        # duel_type
        with self.assertRaises(TypeError):
            DribbleEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=[105.0, 68.0],
                team_side="home",
                _xt=0.02,
                team_id=123,
                player_id=45849,
                related_event_id=123,
                duel_type=1,
                outcome=True,
            )

        # outcome
        with self.assertRaises(TypeError):
            DribbleEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=[105.0, 68.0],
                team_side="home",
                _xt=0.02,
                team_id=123,
                player_id=45849,
                related_event_id=123,
                duel_type="offensive",
                outcome=1,
            )

    def test_dribble_event_eq(self):
        assert self.dribble_event == self.dribble_event

        dribble_event_changed_event_attr = self.dribble_event.copy()
        dribble_event_changed_event_attr.event_id = 123
        assert self.dribble_event != dribble_event_changed_event_attr

        dribble_event_changed_outcome_attr = self.dribble_event.copy()
        dribble_event_changed_outcome_attr.outcome = False
        assert self.dribble_event != dribble_event_changed_outcome_attr

        assert self.dribble_event != 1

    def test_dribble_event_copy(self):
        dribble_event_copy = self.dribble_event.copy()
        assert dribble_event_copy == self.dribble_event

        dribble_event_copy.event_id = 999
        assert dribble_event_copy != self.dribble_event
