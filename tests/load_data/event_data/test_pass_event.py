import unittest

import numpy as np
import pandas as pd

from databallpy.load_data.event_data.pass_event import PassEvent


class TestPassEvent(unittest.TestCase):
    def setUp(self) -> None:
        self.pass_event = PassEvent(
            event_id=1,
            period_id=1,
            minutes=1,
            seconds=1,
            datetime=pd.to_datetime("2021-01-01 00:00:00", utc=True),
            start_x=1.0,
            start_y=1.0,
            team_id=1,
            outcome="successful",
            player_id=1,
            end_x=1.0,
            end_y=1.0,
            length=np.nan,
            angle=np.nan,
            pass_type="pull_back",
            set_piece="free_kick",
        )

    def test_pass_event_copy(self):
        self.assertEqual(self.pass_event, self.pass_event.copy())
        not_eq_pass_event = self.pass_event.copy()
        not_eq_pass_event.event_id = 2
        self.assertNotEqual(self.pass_event, not_eq_pass_event)

    def test_pass_event_eq(self):
        self.assertEqual(self.pass_event, self.pass_event)
        not_eq_pass_event = self.pass_event.copy()

        # base event different
        not_eq_pass_event.event_id = 2
        self.assertNotEqual(self.pass_event, not_eq_pass_event)

        # pass event different
        not_eq_pass_event = self.pass_event.copy()
        not_eq_pass_event.outcome = "unsuccesful"
        self.assertNotEqual(self.pass_event, not_eq_pass_event)

        self.assertNotEqual(self.pass_event, 1)

    def test_pass_event_datatypes(self):
        # super call
        with self.assertRaises(TypeError):
            PassEvent(
                event_id="1",
                period_id=1,
                minutes=1,
                seconds=1,
                datetime=pd.to_datetime("2021-01-01 00:00:00", utc=True),
                start_x=1.0,
                start_y=1.0,
                team_id=1,
                outcome="successful",
                player_id=1,
                end_x=1.0,
                end_y=1.0,
                length=np.nan,
                angle=np.nan,
                pass_type="pull_back",
                set_piece="free_kick",
            )

        # outcome
        with self.assertRaises(TypeError):
            PassEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=1,
                datetime=pd.to_datetime("2021-01-01 00:00:00", utc=True),
                start_x=1.0,
                start_y=1.0,
                team_id=1,
                outcome=1,
                player_id=1,
                end_x=1.0,
                end_y=1.0,
                length=np.nan,
                angle=np.nan,
                pass_type="pull_back",
                set_piece="free_kick",
            )

        with self.assertRaises(ValueError):
            PassEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=1,
                datetime=pd.to_datetime("2021-01-01 00:00:00", utc=True),
                start_x=1.0,
                start_y=1.0,
                team_id=1,
                outcome="rubben_schaken_cross",
                player_id=1,
                end_x=1.0,
                end_y=1.0,
                length=np.nan,
                angle=np.nan,
                pass_type="pull_back",
                set_piece="free_kick",
            )

        # player id
        with self.assertRaises(TypeError):
            PassEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=1,
                datetime=pd.to_datetime("2021-01-01 00:00:00", utc=True),
                start_x=1.0,
                start_y=1.0,
                team_id=1,
                outcome="successful",
                player_id=[1],
                end_x=1.0,
                end_y=1.0,
                length=np.nan,
                angle=np.nan,
                pass_type="pull_back",
                set_piece="free_kick",
            )

        # end x
        with self.assertRaises(TypeError):
            PassEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=1,
                datetime=pd.to_datetime("2021-01-01 00:00:00", utc=True),
                start_x=1.0,
                start_y=1.0,
                team_id=1,
                outcome="successful",
                player_id=1,
                end_x=10,
                end_y=1.0,
                length=np.nan,
                angle=np.nan,
                pass_type="pull_back",
                set_piece="free_kick",
            )

        # end y
        with self.assertRaises(TypeError):
            PassEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=1,
                datetime=pd.to_datetime("2021-01-01 00:00:00", utc=True),
                start_x=1.0,
                start_y=1.0,
                team_id=1,
                outcome="successful",
                player_id=1,
                end_x=1.0,
                end_y="1.",
                length=np.nan,
                angle=np.nan,
                pass_type="pull_back",
                set_piece="free_kick",
            )

        # length
        with self.assertRaises(TypeError):
            PassEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=1,
                datetime=pd.to_datetime("2021-01-01 00:00:00", utc=True),
                start_x=1.0,
                start_y=1.0,
                team_id=1,
                outcome="successful",
                player_id=1,
                end_x=1.0,
                end_y=1.0,
                length="1.",
                angle=np.nan,
                pass_type="pull_back",
                set_piece="free_kick",
            )

        # angle
        with self.assertRaises(TypeError):
            PassEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=1,
                datetime=pd.to_datetime("2021-01-01 00:00:00", utc=True),
                start_x=1.0,
                start_y=1.0,
                team_id=1,
                outcome="successful",
                player_id=1,
                end_x=1.0,
                end_y=1.0,
                length=np.nan,
                angle="1.",
                pass_type="pull_back",
                set_piece="free_kick",
            )

        # pass type
        with self.assertRaises(TypeError):
            PassEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=1,
                datetime=pd.to_datetime("2021-01-01 00:00:00", utc=True),
                start_x=1.0,
                start_y=1.0,
                team_id=1,
                outcome="successful",
                player_id=1,
                end_x=1.0,
                end_y=1.0,
                length=np.nan,
                angle=np.nan,
                pass_type=1,
                set_piece="free_kick",
            )
        with self.assertRaises(ValueError):
            PassEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=1,
                datetime=pd.to_datetime("2021-01-01 00:00:00", utc=True),
                start_x=1.0,
                start_y=1.0,
                team_id=1,
                outcome="successful",
                player_id=1,
                end_x=1.0,
                end_y=1.0,
                length=np.nan,
                angle=np.nan,
                pass_type="failed_shot",
                set_piece="free_kick",
            )

        # set piece
        with self.assertRaises(TypeError):
            PassEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=1,
                datetime=pd.to_datetime("2021-01-01 00:00:00", utc=True),
                start_x=1.0,
                start_y=1.0,
                team_id=1,
                outcome="successful",
                player_id=1,
                end_x=1.0,
                end_y=1.0,
                length=np.nan,
                angle=np.nan,
                pass_type="pull_back",
                set_piece=1,
            )
        with self.assertRaises(ValueError):
            PassEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=1,
                datetime=pd.to_datetime("2021-01-01 00:00:00", utc=True),
                start_x=1.0,
                start_y=1.0,
                team_id=1,
                outcome="successful",
                player_id=1,
                end_x=1.0,
                end_y=1.0,
                length=np.nan,
                angle=np.nan,
                pass_type="pull_back",
                set_piece="tackle",
            )
