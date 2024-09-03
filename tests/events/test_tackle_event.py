import unittest

import pandas as pd

from databallpy.events import IndividualCloseToBallEvent, TackleEvent


class TestTackleEvent(unittest.TestCase):
    def setUp(self) -> None:
        self.tackle_event = TackleEvent(
            event_id=1,
            period_id=1,
            minutes=1,
            seconds=1,
            datetime=pd.to_datetime("2021-01-01 00:00:00"),
            start_x=1.0,
            start_y=1.0,
            team_id=1,
            team_side="home",
            pitch_size=(105, 68),
            player_id=1,
            jersey=1,
            outcome=True,
            related_event_id=-999,
        )

    def test_eq_copy(self):
        tackle_event_copy = self.tackle_event.copy()
        self.assertEqual(self.tackle_event, tackle_event_copy)

        tackle_event_copy.event_id = 2
        self.assertNotEqual(self.tackle_event, tackle_event_copy)

        close_to_ball_event = IndividualCloseToBallEvent(
            event_id=1,
            period_id=1,
            minutes=1,
            seconds=1,
            datetime=pd.to_datetime("2021-01-01 00:00:00"),
            start_x=1.0,
            start_y=1.0,
            team_id=1,
            team_side="home",
            pitch_size=(105, 68),
            player_id=1,
            jersey=1,
            outcome=True,
            related_event_id=-999,
        )
        self.assertNotEqual(self.tackle_event, close_to_ball_event)

    def test_df_attributes(self):
        self.assertEqual(
            self.tackle_event.df_attributes,
            [
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
            ],
        )

    def test_post_init(self):
        # super call
        with self.assertRaises(TypeError):
            TackleEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=1,
                datetime="2021-01-01 00:00:00",
                start_x=1.0,
                start_y=1.0,
                team_id=1,
                team_side="home",
                pitch_size=(105, 68),
                player_id=1,
                jersey=1,
                outcome=True,
                related_event_id=None,
            )
