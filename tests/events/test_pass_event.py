import random
import unittest

import numpy as np
import pandas as pd

from databallpy.events.pass_event import PassEvent, get_opponents_in_passing_lane
from databallpy.features.pressure import get_pressure_on_player
from databallpy.utils.constants import MISSING_INT


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
            team_side="home",
            pitch_size=[105.0, 68.0],
            player_id=1,
            jersey=10,
            outcome=True,
            related_event_id=MISSING_INT,
            body_part="unspecified",
            possession_type="counter_attack",
            set_piece="no_set_piece",
            _xt=0.02,
            outcome_str="successful",
            end_x=1.0,
            end_y=1.0,
            pass_type="pull_back",
        )

    def test_pass_event_copy__eq__(self):
        self.assertEqual(self.pass_event, self.pass_event.copy())
        not_eq_pass_event = self.pass_event.copy()
        not_eq_pass_event.event_id = 2
        self.assertNotEqual(self.pass_event, not_eq_pass_event)
        self.assertNotEqual(self.pass_event, None)

    def test_pass_event_post_init_(self):
        # super call
        with self.assertRaises(TypeError):
            PassEvent(
                event_id=1,
                period_id=1,
                minutes=1,
                seconds=1,
                datetime=pd.to_datetime("2021-01-01 00:00:00", utc=True),
                start_x="1.0",
                start_y=1.0,
                team_id=1,
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
                body_part="unspecified",
                possession_type="counter_attack",
                set_piece="no_set_piece",
                _xt=0.02,
                outcome_str="successful",
                end_x=1.0,
                end_y=1.0,
                pass_type="pull_back",
            )

        # outcome_str
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
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
                body_part="unspecified",
                possession_type="counter_attack",
                set_piece="no_set_piece",
                _xt=0.02,
                outcome_str="failed",
                end_x=1.0,
                end_y=1.0,
                pass_type="pull_back",
            )

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
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
                body_part="unspecified",
                possession_type="counter_attack",
                set_piece="no_set_piece",
                _xt=0.02,
                outcome_str=1,
                end_x=1.0,
                end_y=1.0,
                pass_type="pull_back",
            )

        # pass_type
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
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
                body_part="unspecified",
                possession_type="counter_attack",
                set_piece="no_set_piece",
                _xt=0.02,
                outcome_str="successful",
                end_x=1.0,
                end_y=1.0,
                pass_type="invalid",
            )

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
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
                body_part="unspecified",
                possession_type="counter_attack",
                set_piece="no_set_piece",
                _xt=0.02,
                outcome_str="successful",
                end_x=1.0,
                end_y=1.0,
                pass_type=1,
            )

        float_like_kwargs = {
            "end_x": 1.0,
            "end_y": 1.0,
            "pass_length": 1.0,
            "forward_distance": 1.0,
            "passer_goal_distance": 1.0,
            "pass_end_loc_goal_distance": 1.0,
            "pressure_on_passer": 1.0,
            "pressure_on_receiver": 1.0,
            "pass_goal_angle": 1.0,
        }
        invalid_values = [1, "1.", [1], {1}, (1), {"vkey": 1}]
        for key in float_like_kwargs.keys():
            current_kwargs = float_like_kwargs.copy()
            current_kwargs[key] = random.choice(invalid_values)
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
                    team_side="home",
                    pitch_size=[105.0, 68.0],
                    player_id=1,
                    jersey=10,
                    outcome=True,
                    related_event_id=MISSING_INT,
                    body_part="unspecified",
                    possession_type="counter_attack",
                    set_piece="no_set_piece",
                    _xt=0.02,
                    outcome_str="successful",
                    pass_type="pull_back",
                    **current_kwargs,
                )

        # opponents_in_passing_lane
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
                team_side="home",
                pitch_size=[105.0, 68.0],
                player_id=1,
                jersey=10,
                outcome=True,
                related_event_id=MISSING_INT,
                body_part="unspecified",
                possession_type="counter_attack",
                set_piece="no_set_piece",
                _xt=0.02,
                outcome_str="successful",
                end_x=1.0,
                end_y=1.0,
                pass_type="pull_back",
                opponents_in_passing_lane="1",
            )

    def test_pass_event_df_attributes(self):
        assert self.pass_event.df_attributes == [
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
            "end_x",
            "end_y",
            "pass_type",
            "receiver_player_id",
            "pass_length",
            "forward_distance",
            "passer_goal_distance",
            "pass_end_loc_goal_distance",
            "opponents_in_passing_lane",
            "pressure_on_passer",
            "pressure_on_receiver",
            "pass_goal_angle",
        ]

    def test_pass_add_tracking_data_features(self):
        pass_ = self.pass_event.copy()
        tracking_data_frame = pd.Series(
            {
                "away_1_x": 0.0,
                "away_1_y": 0.0,
                "ball_x": 0.0,
                "ball_y": 0.0,
                "away_2_x": 5.0,
                "away_2_y": 0.0,
                "home_1_x": 2.0,
                "home_1_y": 0.0,
                "home_2_x": 5.0,
                "home_2_y": 0.85,
                "home_3_x": 11.0,
                "home_3_y": 0.0,
            }
        )

        passer_column_id = "away_1"
        receiver_column_id = "away_2"
        pass_end_location = np.array([10.0, 1.0])
        pitch_dimensions = [105.0, 68.0]
        opponent_column_ids = ["home_1", "home_2", "home_3"]
        pass_manual = pass_.copy()
        pass_manual.pass_length = np.sqrt(101)
        pass_manual.forward_distance = -10.0
        pass_manual.passer_goal_distance = 52.5
        pass_manual.pass_end_loc_goal_distance = np.sqrt(62.5**2 + 1)
        pass_manual.opponents_in_passing_lane = 2
        pass_manual.pressure_on_passer = get_pressure_on_player(
            tracking_data_frame, passer_column_id, pitch_size=pitch_dimensions
        )
        pass_manual.pressure_on_receiver = get_pressure_on_player(
            tracking_data_frame, receiver_column_id, pitch_size=pitch_dimensions
        )
        pass_manual.pass_goal_angle = np.arctan2(1, -10) * 180 / np.pi

        pass_.add_tracking_data_features(
            tracking_data_frame,
            passer_column_id,
            receiver_column_id,
            pass_end_location,
            pitch_dimensions,
            opponent_column_ids,
        )
        self.assertEqual(pass_, pass_manual)

    def test_get_opponents_in_passing_lane(self):
        frame = pd.Series(
            {
                "home_1_x": 0.0,
                "home_1_y": 0.0,
                "ball_x": 0.0,
                "ball_y": 0.0,
                "home_2_x": 5.0,  # team mate in passing lane
                "home_2_y": 0.0,
                "away_1_x": 2.0,  # opponent in passing lane
                "away_1_y": 0.0,
                "away_2_x": 5.0,  # only in passing lane when angle > 5 deg.
                "away_2_y": 0.85,
                "away_3_x": 11.0,  # not in passing lane
                "away_3_y": 0.0,
            }
        )
        start_loc = [0.0, 0.0]
        end_loc = [10.0, 0.0]
        opponent_column_ids = ["away_1", "away_2", "away_3"]
        passing_lane_width = 0.8
        passing_lane_angle = 10.0

        res1 = get_opponents_in_passing_lane(
            frame,
            start_loc,
            end_loc,
            opponent_column_ids,
            lane_size=passing_lane_width,
            angle=0.0,
        )
        assert res1 == 1
        res2 = get_opponents_in_passing_lane(
            frame,
            start_loc,
            end_loc,
            opponent_column_ids,
            lane_size=passing_lane_width,
            angle=passing_lane_angle,
        )
        assert res2 == 2

        res3 = get_opponents_in_passing_lane(
            frame,
            [np.nan, np.nan],
            end_loc,
            opponent_column_ids,
            lane_size=passing_lane_width,
            angle=passing_lane_angle,
        )
        assert pd.isnull(res3)
