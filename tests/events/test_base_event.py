import unittest

import numpy as np
import pandas as pd

from databallpy.events import IndividualCloseToBallEvent, IndividualOnBallEvent
from databallpy.utils.constants import MISSING_INT


class TestBaseOnBallEvent(unittest.TestCase):
    def setUp(self):
        self.close_to_ball_event = IndividualCloseToBallEvent(
            event_id=1,
            period_id=1,
            minutes=1,
            seconds=10,
            datetime=pd.to_datetime("2020-01-01 00:00:00"),
            start_x=10.0,
            start_y=11.0,
            team_id=1,
            team_side="home",
            pitch_size=[105.0, 68.0],
            player_id=1,
            jersey=10,
            outcome=True,
            related_event_id=MISSING_INT,
        )

        self.on_ball_event = IndividualOnBallEvent(
            event_id=1,
            period_id=1,
            minutes=1,
            seconds=10,
            datetime=pd.to_datetime("2020-01-01 00:00:00"),
            start_x=10.0,
            start_y=11.0,
            team_id=1,
            team_side="home",
            pitch_size=[105.0, 68.0],
            player_id=1,
            jersey=10,
            outcome=True,
            related_event_id=MISSING_INT,
            body_part="foot",
            possession_type="open_play",
            set_piece="unspecified",
            _xt=0.02,
        )

    def test_individual_close_to_ball_event__eq__(self):
        copy_event = self.close_to_ball_event.copy()
        assert self.close_to_ball_event == copy_event

        assert self.close_to_ball_event != 1

        copy_event.event_id = 2
        assert self.close_to_ball_event != copy_event

    def test_individual_close_to_ball_event_post_init(self):
        # event_id
        with self.assertRaises(TypeError):
            IndividualCloseToBallEvent(
                event_id=1.3,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
            )
        # period_id
        with self.assertRaises(TypeError):
            IndividualCloseToBallEvent(
                event_id=1,
                period_id="1.3",
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
            )
        # minutes
        with self.assertRaises(TypeError):
            IndividualCloseToBallEvent(
                event_id=1,
                period_id=1,
                minutes="1.3",
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
            )
        # seconds
        with self.assertRaises(TypeError):
            IndividualCloseToBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=[10],
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
            )

        # datetime
        with self.assertRaises(TypeError):
            IndividualCloseToBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime="2020-01-01 00:00:00",
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
            )

        # start_x
        with self.assertRaises(TypeError):
            IndividualCloseToBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
            )

        # start_y
        with self.assertRaises(TypeError):
            IndividualCloseToBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y={11.0},
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
            )

        # team_id
        with self.assertRaises(TypeError):
            IndividualCloseToBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=[1],
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
            )

        # team_side
        with self.assertRaises(TypeError):
            IndividualCloseToBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side=1,
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
            )

        with self.assertRaises(ValueError):
            IndividualCloseToBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="test",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
            )

        # pitch size
        with self.assertRaises(TypeError):
            IndividualCloseToBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=105.0,
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
            )

        with self.assertRaises(ValueError):
            IndividualCloseToBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0, 10.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
            )

        with self.assertRaises(TypeError):
            IndividualCloseToBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, "68.0"],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
            )

        # player_id
        with self.assertRaises(TypeError):
            IndividualCloseToBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=[1],
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
            )

        # jersey
        with self.assertRaises(TypeError):
            IndividualCloseToBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey="10",
                outcome=True,
                related_event_id=MISSING_INT,
            )

        # outcome
        with self.assertRaises(TypeError):
            IndividualCloseToBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=1,
                related_event_id=MISSING_INT,
            )

        # related_event_id
        with self.assertRaises(TypeError):
            IndividualCloseToBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id={22},
            )

    def test_individual_close_to_ball_base_df_attributes(self):
        assert self.close_to_ball_event.base_df_attributes == [
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
        ]

    def test_individual_close_to_ball_event_copy(self):
        copy_event = self.close_to_ball_event.copy()
        assert self.close_to_ball_event == copy_event

        copy_event.pitch_size = [10.0, 105.0]
        assert self.close_to_ball_event != copy_event

    def test_individual_on_ball_event__eq__(self):
        copy_event = self.on_ball_event.copy()
        assert self.on_ball_event == copy_event

        assert self.on_ball_event != 1

        copy_event.event_id = 2
        assert self.on_ball_event != copy_event

        copy_event.event_id = 1
        assert self.on_ball_event == copy_event

        copy_event.body_part = "head"
        assert self.on_ball_event != copy_event

        copy_event.body_part = "foot"
        assert self.on_ball_event == copy_event

        copy_event._xt = 0.03
        assert self.on_ball_event != copy_event

    def test_individual_on_ball_event_post_init(self):
        # super call
        with self.assertRaises(TypeError):
            IndividualOnBallEvent(
                event_id=1.3,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
                body_part="foot",
                possession_type="open_play",
                set_piece="unspecified",
                _xt=0.02,
            )

        # body_part
        with self.assertRaises(TypeError):
            IndividualOnBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
                body_part=1,
                possession_type="open_play",
                set_piece="unspecified",
                _xt=0.02,
            )

        with self.assertRaises(ValueError):
            IndividualOnBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
                body_part="test",
                possession_type="open_play",
                set_piece="unspecified",
                _xt=0.02,
            )

        # possession_type
        with self.assertRaises(TypeError):
            IndividualOnBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
                body_part="foot",
                possession_type=1,
                set_piece="unspecified",
                _xt=0.02,
            )

        with self.assertRaises(ValueError):
            IndividualOnBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
                body_part="foot",
                possession_type="test",
                set_piece="unspecified",
                _xt=0.02,
            )

        # set_piece
        with self.assertRaises(TypeError):
            IndividualOnBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
                body_part="foot",
                possession_type="open_play",
                set_piece=1,
                _xt=0.02,
            )

        with self.assertRaises(ValueError):
            IndividualOnBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
                body_part="foot",
                possession_type="open_play",
                set_piece="test",
                _xt=0.02,
            )

        # _xt
        with self.assertRaises(TypeError):
            IndividualOnBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
                body_part="foot",
                possession_type="open_play",
                set_piece="unspecified",
                _xt="0.02",
            )

        # test recalculation of xT
        event = IndividualOnBallEvent(
            event_id=1,
            period_id=1,
            minutes=1,
            seconds=10,
            datetime=pd.to_datetime("2020-01-01 00:00:00"),
            start_x=10.0,
            start_y=11.0,
            team_id=1,
            team_side="home",
            pitch_size=[105.0, 68.0],
            player_id=1,
            jersey=10,
            outcome=True,
            related_event_id=MISSING_INT,
            body_part="foot",
            possession_type="open_play",
            set_piece="unspecified",
            _xt=np.nan,
        )
        self.assertAlmostEqual(event.xT, 0.008168, places=5)

    def test_individual_on_ball_base_df_attributes(self):
        assert self.on_ball_event.base_df_attributes == [
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
            "xT",
            "body_part",
            "possession_type",
            "set_piece",
        ]

    def test_individual_on_ball_event_xT(self):
        event = self.on_ball_event.copy()
        event._xt = -1
        event.body_part = "foot"
        event.set_piece = "penalty"

        assert event.xT == 0.797

        event._xt = -1
        event.set_piece = "corner_kick"
        assert event.xT == 0.049

        event.set_piece = "goal_kick"
        event._xt = -1
        assert event.xT == 0.0

        event.set_piece = "kick_off"
        event._xt = -1
        assert event.xT == 0.001

        event.set_piece = "throw_in"
        event._xt = -1
        np.testing.assert_almost_equal(event.xT, 0.0)

        event.set_piece = "free_kick"
        event._xt = -1
        np.testing.assert_almost_equal(event.xT, 0.0139, decimal=4)

        event.set_piece = "no_set_piece"
        event._xt = -1
        np.testing.assert_almost_equal(event.xT, 0.0082, decimal=4)

        event.set_piece = "unspecified"
        event._xt = -1
        event.team_side = "away"
        np.testing.assert_almost_equal(event.xT, 0.0041, decimal=4)

        with self.assertRaises(ValueError):
            event.set_piece = "test"
            event._xt = -1
            event.xT

    def test_individual_on_ball_event_copy(self):
        copy_event = self.on_ball_event.copy()
        assert self.on_ball_event == copy_event

        copy_event.pitch_size = [10.0, 105.0]
        assert self.on_ball_event != copy_event
