import random
import unittest

import numpy as np
import pandas as pd

from databallpy.events import ShotEvent


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
            team_id=123,
            team_side="home",
            pitch_size=(106, 68),
            player_id=45849,
            jersey=10,
            outcome=False,
            related_event_id=123,
            _xt=0.1,
            body_part="head",
            possession_type="free_kick",
            set_piece="no_set_piece",
            z_target=15.0,
            y_target=3.5,
            outcome_str="own_goal",
            first_touch=False,
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

    def test_shot_event_post_init(self):
        # super call
        with self.assertRaises(TypeError):
            ShotEvent(
                event_id="2512690515",
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                team_id=123,
                team_side="home",
                pitch_size=(106, 68),
                player_id=45849,
                jersey=10,
                outcome=False,
                related_event_id=123,
                _xt=0.1,
                body_part="head",
                possession_type="free_kick",
                set_piece="no_set_piece",
                z_target=15.0,
                y_target=3.5,
                outcome_str="own_goal",
                first_touch=False,
            )

        # outcome_str
        with self.assertRaises(ValueError):
            ShotEvent(
                event_id=2512690515,
                period_id=1,
                minutes=9,
                seconds=17,
                datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                start_x=50.0,
                start_y=20.0,
                team_id=123,
                team_side="home",
                pitch_size=(106, 68),
                player_id=45849,
                jersey=10,
                outcome=False,
                related_event_id=123,
                _xt=0.1,
                body_part="head",
                possession_type="free_kick",
                set_piece="no_set_piece",
                z_target=15.0,
                y_target=3.5,
                outcome_str="unknown_outcome",
                first_touch=False,
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
                team_id=123,
                team_side="home",
                pitch_size=(106, 68),
                player_id=45849,
                jersey=10,
                outcome=False,
                related_event_id=123,
                _xt=0.1,
                body_part="head",
                possession_type="free_kick",
                set_piece="no_set_piece",
                z_target=15.0,
                y_target=3.5,
                outcome_str=1,
                first_touch="False",
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
                team_id=123,
                team_side="home",
                pitch_size=(106, 68),
                player_id=45849,
                jersey=10,
                outcome=False,
                related_event_id=123,
                _xt=0.1,
                body_part="head",
                possession_type="free_kick",
                set_piece="no_set_piece",
                z_target=15.0,
                y_target=3.5,
                outcome_str="own_goal",
                first_touch="False",
            )

        float_like_kwargs = {
            "y_target": 3.5,
            "z_target": 15.0,
            "ball_goal_distance": 18.0,
            "shot_angle": 20.0,
        }
        # assert runs without errors
        ShotEvent(
            event_id=2512690515,
            period_id=1,
            minutes=9,
            seconds=17,
            datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
            start_x=50.0,
            start_y=20.0,
            team_id=123,
            team_side="home",
            pitch_size=(106, 68),
            player_id=45849,
            jersey=10,
            outcome=False,
            related_event_id=123,
            _xt=0.1,
            body_part="head",
            possession_type="free_kick",
            set_piece="no_set_piece",
            outcome_str="own_goal",
            first_touch=False,
            **float_like_kwargs,
        )

        invalid_values = [1, "1", [1], {1}, {"val": 1}]
        for key in float_like_kwargs.keys():
            current_kwargs = float_like_kwargs.copy()
            current_kwargs[key] = random.choice(invalid_values)
            with self.assertRaises(TypeError):
                ShotEvent(
                    event_id=2512690515,
                    period_id=1,
                    minutes=9,
                    seconds=17,
                    datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True),
                    start_x=50.0,
                    start_y=20.0,
                    team_id=123,
                    team_side="home",
                    pitch_size=(106, 68),
                    player_id=45849,
                    jersey=10,
                    outcome=False,
                    related_event_id=123,
                    _xt=0.1,
                    body_part="head",
                    possession_type="free_kick",
                    set_piece="no_set_piece",
                    outcome_str="own_goal",
                    first_touch=False,
                    **current_kwargs,
                )

    def test_shot_event_df_attributes(self):
        assert self.shot_event.df_attributes == [
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
            "outcome_str",
            "y_target",
            "z_target",
            "first_touch",
            "xg",
        ]

    def test_shot_event_eq__copy(self):
        assert self.shot_event == self.shot_event

        shot_event_changed_event_attr = self.shot_event.copy()
        shot_event_changed_event_attr.event_id = 123
        assert self.shot_event != shot_event_changed_event_attr

        shot_event_changed_shot_attr = self.shot_event.copy()
        shot_event_changed_shot_attr.outcome_str = "goal"
        assert self.shot_event != shot_event_changed_shot_attr

        assert self.shot_event != 1

    def test_get_xg_valid(self):
        shot_event = self.shot_event.copy()
        shot_event.ball_goal_distance = 18.0
        shot_event.shot_angle = 20
        shot_event.body_part = "left_foot"
        shot_event.set_piece = "penalty"

        res_own_goal = shot_event.get_xg()

        shot_event.outcome_str = "miss_off_target"
        res_penalty = shot_event.get_xg()

        shot_event.set_piece = "free_kick"
        res_free_kick = shot_event.get_xg()

        shot_event.set_piece = "corner_kick"
        res_reg_foot = shot_event.get_xg()

        shot_event.body_part = "head"
        res_head = shot_event.get_xg()
        assert res_own_goal == 0.0
        assert res_penalty == 0.79
        assert 0.3 > res_free_kick > res_reg_foot > res_head > 0.0

    def test_get_xg_invalid(self):
        shot_event = self.shot_event.copy()
        shot_event.ball_goal_distance = np.nan
        shot_event.shot_angle = 20
        shot_event.body_part = "left_foot"
        shot_event.set_piece = "penalty"
        shot_event.outcome_str = "miss_off_target"
        assert pd.isnull(shot_event.get_xg())

        shot_event.ball_goal_distance = 18.0
        shot_event.set_piece = "unknown_type_of_play"
        assert round(shot_event.get_xg(), 4) == 0.0805
