import unittest

import numpy as np
import pandas as pd

from databallpy.events import ShotEvent
from databallpy.features.angle import get_smallest_angle


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
            pitch_size=(106, 68),
            team_side="home",
            _xt=0.1,
            team_id=123,
            z_target=15.0,
            y_target=3.5,
            player_id=45849,
            shot_outcome="own_goal",
            body_part="head",
            type_of_play="free_kick",
            first_touch=False,
            created_oppertunity="regular_play",
            related_event_id=123,
        )

        self.tracking_data_frame = pd.Series(
            {
                "frame": 1,
                "home_1_x": 0,
                "home_1_y": 0,
                "away_1_x": 10,
                "away_1_y": 10,
                "away_2_x": 30,
                "away_2_y": 0,
                "ball_x": 0,
                "ball_y": 0,
                "event_id": 2512690515,
            }
        )

    def test_post_init(self):
        # super() call
        with self.assertRaises(TypeError):
            ShotEvent(
                event_id="2512690515",
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=(106, 68),
                team_side="home",
                _xt=0.1,
                team_id=123,
                z_target=15.0,
                y_target=3.5,
                player_id=45849,
                shot_outcome="own_goal",
                body_part="head",
                type_of_play="corner_kick",
                first_touch=False,
                created_oppertunity="regular_play",
                related_event_id=123,
            )

        # player_id
        with self.assertRaises(TypeError):
            ShotEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=(106, 68),
                team_side="home",
                _xt=0.1,
                team_id=123,
                z_target=15.0,
                y_target=3.5,
                player_id=[22],
                shot_outcome="own_goal",
                body_part="head",
                type_of_play="corner_kick",
                first_touch=False,
                created_oppertunity="regular_play",
                related_event_id=123,
            )

        # shot_outcome
        with self.assertRaises(TypeError):
            ShotEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=(106, 68),
                team_side="home",
                _xt=0.1,
                team_id=123,
                z_target=15.0,
                y_target=3.5,
                player_id=45849,
                shot_outcome=1,
                body_part="head",
                type_of_play="corner_kick",
                first_touch=False,
                created_oppertunity="regular_play",
                related_event_id=123,
            )

        with self.assertRaises(ValueError):
            ShotEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=(106, 68),
                team_side="home",
                _xt=0.1,
                team_id=123,
                z_target=15.0,
                y_target=3.5,
                player_id=45849,
                shot_outcome="wrong_outcome",
                body_part="head",
                type_of_play="corner_kick",
                first_touch=False,
                created_oppertunity="regular_play",
                related_event_id=123,
            )

        # y_target and z_target
        with self.assertRaises(TypeError):
            ShotEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=(106, 68),
                team_side="home",
                _xt=0.1,
                team_id=123,
                z_target="15.0",
                y_target=3.5,
                player_id=45849,
                shot_outcome="own_goal",
                body_part="head",
                type_of_play="corner_kick",
                first_touch=False,
                created_oppertunity="regular_play",
                related_event_id=123,
            )
        with self.assertRaises(TypeError):
            ShotEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=(106, 68),
                team_side="home",
                _xt=0.1,
                team_id=123,
                z_target=15.0,
                y_target=3,
                player_id=45849,
                shot_outcome="own_goal",
                body_part="head",
                type_of_play="corner_kick",
                first_touch=False,
                created_oppertunity="regular_play",
                related_event_id=123,
            )

        # body_part
        with self.assertRaises(TypeError):
            ShotEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=(106, 68),
                team_side="home",
                _xt=0.1,
                team_id=123,
                z_target=15.0,
                y_target=3.5,
                player_id=45849,
                shot_outcome="own_goal",
                body_part=3,
                type_of_play="corner_kick",
                first_touch=False,
                created_oppertunity="regular_play",
                related_event_id=123,
            )
        with self.assertRaises(ValueError):
            ShotEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=(106, 68),
                team_side="home",
                _xt=0.1,
                team_id=123,
                z_target=15.0,
                y_target=3.5,
                player_id=45849,
                shot_outcome="own_goal",
                body_part="wrong_body_part",
                type_of_play="corner_kick",
                first_touch=False,
                created_oppertunity="regular_play",
                related_event_id=123,
            )

        # type_of_play
        with self.assertRaises(TypeError):
            ShotEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=(106, 68),
                team_side="home",
                _xt=0.1,
                team_id=123,
                z_target=15.0,
                y_target=3.5,
                player_id=45849,
                shot_outcome="own_goal",
                body_part="head",
                type_of_play=3,
                first_touch=False,
                created_oppertunity="regular_play",
                related_event_id=123,
            )
        with self.assertRaises(ValueError):
            ShotEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=(106, 68),
                team_side="home",
                _xt=0.1,
                team_id=123,
                z_target=15.0,
                y_target=3.5,
                player_id=45849,
                shot_outcome="own_goal",
                body_part="head",
                type_of_play="wrong_type_of_play",
                first_touch=False,
                created_oppertunity="regular_play",
                related_event_id=123,
            )

        # first_touch
        with self.assertRaises(TypeError):
            ShotEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=(106, 68),
                team_side="home",
                _xt=0.1,
                team_id=123,
                z_target=15.0,
                y_target=3.5,
                player_id=45849,
                shot_outcome="own_goal",
                body_part="head",
                type_of_play="corner_kick",
                first_touch=3,
                created_oppertunity="regular_play",
                related_event_id=123,
            )

        # created_oppertunity
        with self.assertRaises(TypeError):
            ShotEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=(106, 68),
                team_side="home",
                _xt=0.1,
                team_id=123,
                z_target=15.0,
                y_target=3.5,
                player_id=45849,
                shot_outcome="own_goal",
                body_part="head",
                type_of_play="corner_kick",
                first_touch=False,
                created_oppertunity=3,
                related_event_id=123,
            )
        with self.assertRaises(ValueError):
            ShotEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=(106, 68),
                team_side="home",
                _xt=0.1,
                team_id=123,
                z_target=15.0,
                y_target=3.5,
                player_id=45849,
                shot_outcome="own_goal",
                body_part="head",
                type_of_play="corner_kick",
                first_touch=False,
                created_oppertunity="wrong_created_oppertunity",
                related_event_id=123,
            )

        # related_event_id
        with self.assertRaises(TypeError):
            ShotEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=(106, 68),
                team_side="home",
                _xt=0.1,
                team_id=123,
                z_target=15.0,
                y_target=3.5,
                player_id=45849,
                shot_outcome="own_goal",
                body_part="head",
                type_of_play="corner_kick",
                first_touch=False,
                created_oppertunity="regular_play",
                related_event_id="123",
            )
        # td_vars float
        with self.assertRaises(TypeError):
            ShotEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=(106, 68),
                team_side="home",
                _xt=0.1,
                team_id=123,
                z_target=15.0,
                y_target=3.5,
                player_id=45849,
                shot_outcome="own_goal",
                body_part="head",
                type_of_play="corner_kick",
                first_touch=False,
                created_oppertunity="regular_play",
                related_event_id=123,
                ball_goal_distance=10,
            )
        # td_vars int
        with self.assertRaises(TypeError):
            ShotEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                pitch_size=(106, 68),
                team_side="home",
                _xt=0.1,
                team_id=123,
                z_target=15.0,
                y_target=3.5,
                player_id=45849,
                shot_outcome="own_goal",
                body_part="head",
                type_of_play="corner_kick",
                first_touch=False,
                created_oppertunity="regular_play",
                related_event_id=123,
                n_obstructive_defenders=1.5,
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

    def test_shot_event_add_tracking_data_features(self):
        shot_event = self.shot_event.copy()
        shot_event.add_tracking_data_features(
            tracking_data_frame=self.tracking_data_frame,
            column_id="home_1",
            gk_column_id="away_1",
        )
        ball_left_post_vec = [53, 7.32 / 2]
        ball_right_post_vec = [53, -7.32 / 2]

        self.assertAlmostEqual(
            shot_event.ball_goal_distance, np.sqrt(53**2), places=4
        )
        self.assertAlmostEqual(
            shot_event.ball_gk_distance, np.sqrt(10**2 + 10**2), places=4
        )
        self.assertAlmostEqual(
            shot_event.shot_angle,
            get_smallest_angle(
                ball_left_post_vec, ball_right_post_vec, angle_format="degree"
            ),
            places=4,
        )

        self.assertAlmostEqual(
            shot_event.gk_optimal_loc_distance,
            10.0,
            places=4,
        )
        self.assertEqual(shot_event.n_obstructive_players, 1)
        self.assertEqual(shot_event.n_obstructive_defenders, 1)
        self.assertAlmostEqual(
            shot_event.goal_gk_distance, np.sqrt(43**2 + 10**2), places=4
        )

    def test_shot_event_copy(self):
        shot_event_copy = self.shot_event.copy()
        assert shot_event_copy == self.shot_event

    def test_get_xG_valid(self):
        shot_event = self.shot_event.copy()
        shot_event.ball_goal_distance = 18.0
        shot_event.shot_angle = 20
        shot_event.body_part = "left_foot"
        shot_event.type_of_play = "penalty"
        res_penalty = shot_event.get_xG()

        shot_event.type_of_play = "free_kick"
        res_free_kick = shot_event.get_xG()

        shot_event.type_of_play = "corner_kick"
        res_reg_foot = shot_event.get_xG()

        shot_event.body_part = "head"
        res_head = shot_event.get_xG()

        assert res_penalty == 0.79
        assert 0.3 > res_free_kick > res_reg_foot > res_head > 0.0

    def test_get_xG_invalid(self):
        shot_event = self.shot_event.copy()
        shot_event.ball_goal_distance = np.nan
        shot_event.shot_angle = 20
        shot_event.body_part = "left_foot"
        shot_event.type_of_play = "penalty"
        assert pd.isnull(shot_event.get_xG())

        shot_event.ball_goal_distance = 18.0
        shot_event.type_of_play = "unknown_type_of_play"
        assert round(shot_event.get_xG(), 4) == 0.0805
