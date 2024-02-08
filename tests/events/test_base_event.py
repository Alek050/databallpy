import math
import os
import unittest

import joblib
import pandas as pd

from databallpy.events import BaseOnBallEvent, PassEvent
from databallpy.features.angle import get_smallest_angle


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
        distance_to_goal = math.dist([10, 11], [52.5, 0])

        angle_to_goal = get_smallest_angle(
            [10 - 52.5, 11 - 3.66], [10 - 52.5, 11 + 3.66], angle_format="degree"
        )
        dist_ang = distance_to_goal * angle_to_goal
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "databallpy", "models"
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

        throw_in_model = joblib.load(f"{path}/xT_throw_ins.pkl")
        pass_event.set_piece = "throw_in"
        pass_event._xt = -1
        assert (
            pass_event.xT == throw_in_model.predict([[distance_to_goal / 107.313]])[0]
        )

        free_kick_model = joblib.load(f"{path}/xT_free_kicks.pkl")
        pass_event.set_piece = "free_kick"
        pass_event._xt = -1
        assert (
            pass_event.xT
            == free_kick_model.predict(
                [
                    [
                        angle_to_goal / 24.265,
                        distance_to_goal / 107.313,
                        dist_ang / 419.069,
                    ]
                ]
            )[0]
        )

        open_play_model = joblib.load(f"{path}/xT_open_play.pkl")
        pass_event.set_piece = "no_set_piece"
        pass_event._xt = -1
        assert (
            pass_event.xT
            == open_play_model.predict(
                [
                    [
                        angle_to_goal / 125.493,
                        distance_to_goal / 109.313,
                        dist_ang / 419.195,
                    ]
                ]
            )[0]
        )

        pass_event.set_piece = "unspecified_set_piece"
        pass_event._xt = -1
        pass_event.team_side = "away"
        distance_to_goal = math.dist([10, 11], [-52.5, 0])
        angle_to_goal = get_smallest_angle(
            [10 + 52.5, 11 + 3.66], [10 + 52.5, 11 - 3.66], angle_format="degree"
        )
        dist_ang = distance_to_goal * angle_to_goal
        assert pass_event.xT == max(
            open_play_model.predict(
                [
                    [
                        angle_to_goal / 125.493,
                        distance_to_goal / 109.313,
                        dist_ang / 419.195,
                    ]
                ]
            )[0],
            0.0,
        )

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

    def test_base_on_ball_eventt_post_init(self):
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
