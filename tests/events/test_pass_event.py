import random
import unittest

import numpy as np
import pandas as pd

from databallpy.events.pass_event import PassEvent
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
        ]

