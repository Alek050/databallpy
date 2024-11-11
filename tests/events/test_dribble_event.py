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
            team_id=123,
            team_side="home",
            pitch_size=[105.0, 68.0],
            player_id=45849,
            jersey=10,
            outcome=True,
            related_event_id=123,
            body_part="foot",
            possession_type="unspecified",
            set_piece="no_set_piece",
            _xt=0.02,
            duel_type="offensive",
            with_opponent=False,
        )

    def test_post_init(self):
        # super() call CloseToBallEvent
        with self.assertRaises(TypeError):
            DribbleEvent(
                event_id="2512690515",
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                team_id=123,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=45849,
                jersey=10,
                outcome=True,
                related_event_id=123,
                body_part="foot",
                possession_type="unspecified",
                set_piece="no_set_piece",
                _xt=0.02,
                duel_type="offensive",
                with_opponent=False,
            )

        # super() call OnBallEvent
        with self.assertRaises(ValueError):
            DribbleEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                team_id=123,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=45849,
                jersey=10,
                outcome=True,
                related_event_id=123,
                body_part="foot",
                possession_type="test",
                set_piece="no_set_piece",
                _xt=0.02,
                duel_type="offensive",
                with_opponent=False,
            )

        # duel type
        with self.assertRaises(TypeError):
            DribbleEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                team_id=123,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=45849,
                jersey=10,
                outcome=True,
                related_event_id=123,
                body_part="foot",
                possession_type="unspecified",
                set_piece="no_set_piece",
                _xt=0.02,
                duel_type=3,
                with_opponent=False,
            )

        with self.assertRaises(ValueError):
            DribbleEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                team_id=123,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=45849,
                jersey=10,
                outcome=True,
                related_event_id=123,
                body_part="foot",
                possession_type="unspecified",
                set_piece="no_set_piece",
                _xt=0.02,
                duel_type="invalid",
                with_opponent=False,
            )

        # with opponent
        with self.assertRaises(TypeError):
            DribbleEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                team_id=123,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=45849,
                jersey=10,
                outcome=True,
                related_event_id=123,
                body_part="foot",
                possession_type="unspecified",
                set_piece="no_set_piece",
                _xt=0.02,
                duel_type="offensive",
                with_opponent=1,
            )

        # _xt update
        event = DribbleEvent(
            event_id=2512690515,
            period_id=1,
            minutes=9,
            seconds=17,
            datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
            start_x=50.0,
            start_y=20.0,
            team_id=123,
            team_side="home",
            pitch_size=[105.0, 68.0],
            player_id=45849,
            jersey=10,
            outcome=True,
            related_event_id=123,
            body_part="foot",
            possession_type="unspecified",
            set_piece="no_set_piece",
            _xt=-1,
            duel_type="offensive",
            with_opponent=False,
        )
        self.assertAlmostEqual(event.xt, 0.0369635, places=6)

    def test_dribble_event__eq_and_copy__(self):
        dribble_event = self.dribble_event.copy()
        assert self.dribble_event == dribble_event

        dribble_event.event_id = 123
        assert self.dribble_event != dribble_event

        dribble_event = self.dribble_event.copy()
        dribble_event.duel_type = "defensive"
        assert self.dribble_event != dribble_event

        assert self.dribble_event != 1

    def test_dribble_event_df_attributes(self):
        assert self.dribble_event.df_attributes == [
            "event_id",
            "period_id",
            "minutes",
            "seconds",
            "datetime",
            "start_x",
            "start_y",
            "team_id",
            "team_side",
            "player_id",
            "jersey",
            "outcome",
            "related_event_id",
            "xt",
            "body_part",
            "possession_type",
            "set_piece",
            "duel_type",
            "with_opponent",
        ]
