import unittest

import numpy as np
import pandas as pd

from databallpy.events import BaseOnBallEvent, PassEvent


class TestBaseOnBallEvent(unittest.TestCase):
    def setUp(self):
        self.base_event = BaseOnBallEvent(
            event_id=1,
            period_id=1,
            minutes=1,
            seconds=10,
            datetime=pd.to_datetime("2020-01-01 00:00:00"),
            start_x=10.0,
            start_y=11.0,
            team_id=1,
            pitch_size=[105.0, 68.0],
            team_side="home",
            _xt=0.02,
        )

    def test_base_on_ball_event_xt(self):
        pass_event = PassEvent(
            event_id=1,
            period_id=1,
            minutes=1,
            seconds=10,
            datetime=pd.to_datetime("2020-01-01 00:00:00"),
            start_x=10.0,
            start_y=11.0,
            pitch_size=[105.0, 68.0],
            team_side="home",
            _xt=-1,
            team_id=1,
            outcome="successful",
            end_x=20.0,
            end_y=21.0,
            player_id=1,
            pass_type="not_specified",
            set_piece="penalty",
            pass_length=10.0,
        )

        assert pass_event.xT == 0.797

        pass_event._xt = -1
        pass_event.set_piece = "corner_kick"
        assert pass_event.xT == 0.049

        pass_event.set_piece = "goal_kick"
        pass_event._xt = -1
        assert pass_event.xT == 0.0

        pass_event.set_piece = "kick_off"
        pass_event._xt = -1
        assert pass_event.xT == 0.001

        pass_event.set_piece = "throw_in"
        pass_event._xt = -1
        np.testing.assert_almost_equal(pass_event.xT, 0.0)

        pass_event.set_piece = "free_kick"
        pass_event._xt = -1
        np.testing.assert_almost_equal(pass_event.xT, 0.0139, decimal=4)

        pass_event.set_piece = "no_set_piece"
        pass_event._xt = -1
        np.testing.assert_almost_equal(pass_event.xT, 0.0082, decimal=4)

        pass_event.set_piece = "unspecified_set_piece"
        pass_event._xt = -1
        pass_event.team_side = "away"
        np.testing.assert_almost_equal(pass_event.xT, 0.0041, decimal=4)

        with self.assertRaises(ValueError):
            pass_event.set_piece = "test"
            pass_event._xt = -1
            pass_event.xT

    def test_base_on_ball_event__eq__(self):
        assert self.base_event == self.base_event

        assert self.base_event != 1
        assert self.base_event != BaseOnBallEvent(
            event_id=2,
            period_id=1,
            minutes=1,
            seconds=10,
            datetime=pd.to_datetime("2020-01-01 00:00:00"),
            start_x=10.0,
            start_y=11.0,
            team_id=1,
            pitch_size=[105.0, 68.0],
            team_side="home",
            _xt=0.02,
        )

    def test_base_on_ball_event_post_init(self):
        # event_id
        with self.assertRaises(TypeError):
            BaseOnBallEvent(
                event_id=1.3,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                pitch_size=[105.0, 68.0],
                team_side="home",
                _xt=0.02,
            )
        # period_id
        with self.assertRaises(TypeError):
            BaseOnBallEvent(
                event_id=1,
                period_id="1.3",
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                pitch_size=[105.0, 68.0],
                team_side="home",
                _xt=0.02,
            )
        # minutes
        with self.assertRaises(TypeError):
            BaseOnBallEvent(
                event_id=1,
                period_id=1,
                minutes="1.3",
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                pitch_size=[105.0, 68.0],
                team_side="home",
                _xt=0.02,
            )
        # seconds
        with self.assertRaises(TypeError):
            BaseOnBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=[10],
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                pitch_size=[105.0, 68.0],
                team_side="home",
                _xt=0.02,
            )
        # datetime
        with self.assertRaises(TypeError):
            BaseOnBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime="2020-01-01 00:00:00",
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                pitch_size=[105.0, 68.0],
                team_side="home",
                _xt=0.02,
            )
        # start_x
        with self.assertRaises(TypeError):
            BaseOnBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10,
                start_y=11.0,
                team_id=1,
                pitch_size=[105.0, 68.0],
                team_side="home",
                _xt=0.02,
            )
        # start_y
        with self.assertRaises(TypeError):
            BaseOnBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y={11.0},
                team_id=1,
                pitch_size=[105.0, 68.0],
                team_side="home",
                _xt=0.02,
            )
        # team_id
        with self.assertRaises(TypeError):
            BaseOnBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=[1],
                pitch_size=[105.0, 68.0],
                team_side="home",
                _xt=0.02,
            )

        # pitch_size
        with self.assertRaises(TypeError):
            BaseOnBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                pitch_size=[105.0, "68.0"],
                team_side="home",
                _xt=0.02,
            )
        with self.assertRaises(ValueError):
            BaseOnBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                pitch_size=[105.0, 68.0, 10.0],
                team_side="home",
                _xt=0.02,
            )
        with self.assertRaises(TypeError):
            BaseOnBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                pitch_size=100.68,
                team_side="home",
                _xt=0.02,
            )
        # team_side
        with self.assertRaises(TypeError):
            BaseOnBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                pitch_size=[105.0, 68.0],
                team_side=1,
                _xt=0.02,
            )
        with self.assertRaises(ValueError):
            BaseOnBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                pitch_size=[105.0, 68.0],
                team_side="test",
                _xt=0.02,
            )
        # _xt
        with self.assertRaises(TypeError):
            BaseOnBallEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=10,
                datetime=pd.to_datetime("2020-01-01 00:00:00"),
                start_x=10.0,
                start_y=11.0,
                team_id=1,
                pitch_size=[105.0, 68.0],
                team_side="home",
                _xt="0.02",
            )
