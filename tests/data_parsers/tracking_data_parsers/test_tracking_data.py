import unittest
import numpy as np
import pandas as pd
import math

from databallpy.features.pressure import (
    calculate_variable_dfront,
    calculate_l,
    calculate_z,
)
from databallpy.utils.utils import MISSING_INT

from databallpy.data_parsers.tracking_data_parsers.tracking_data import TrackingData


class TestTrackingData(unittest.TestCase):
    def setUp(self):
        self.vel_intervals = ((-1, 13), (30, 17))
        self.acc_intervals = ((37, 15), (0, 0.75))
        self.td_covered_distance = TrackingData(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "home_1_vx": [10, -20, 10, np.nan, 10, np.nan],
                "home_1_vy": [7, -12.5, 9, np.nan, 15, np.nan],
                "home_1_velocity": [
                    np.sqrt(149),
                    np.sqrt(556.25),
                    np.sqrt(181),
                    np.nan,
                    np.sqrt(325),
                    np.nan,
                ],
                "home_1_ax": [-30, 0, np.nan, 0, np.nan, np.nan],
                "home_1_ay": [-19.5, 1, np.nan, 3, np.nan, np.nan],
                "home_1_acceleration": [np.sqrt(1280.25), 1, np.nan, 3, np.nan, np.nan],
                "away_2_x": [2, 3, 1, 1, 3, 2],
                "away_2_y": [1, 2, 3, 4, 5, 6],
                "away_2_vx": [2, 3, 1, 1, 3, 2],
                "away_2_vy": [1, 1, 1, 1, 1, 1],
                "away_2_velocity": [
                    np.sqrt(5),
                    np.sqrt(10),
                    np.sqrt(2),
                    np.sqrt(2),
                    np.sqrt(10),
                    np.sqrt(5),
                ],
                "away_2_ax": [1, -0.5, -1, 1, 0.5, -1],
                "away_2_ay": [0, 0, 0, 0, 0, 0],
                "away_2_acceleration": [1, 0.5, 1, 1, 0.5, 1],
            },
            provider="test",
            frame_rate=1,
        )

        self.expected_total_distance = pd.DataFrame(
            index=["away_2", "home_1"],
            data={
                "total_distance": [
                    np.sqrt(5)
                    + np.sqrt(10)
                    + np.sqrt(2)
                    + np.sqrt(2)
                    + np.sqrt(10)
                    + np.sqrt(5),
                    np.sqrt(149) + np.sqrt(556.25) + np.sqrt(181) + np.sqrt(325),
                ],
                "total_distance_velocity_-1_13": [
                    np.sqrt(5)
                    + np.sqrt(10)
                    + np.sqrt(2)
                    + np.sqrt(2)
                    + np.sqrt(10)
                    + np.sqrt(5),
                    np.sqrt(149),
                ],
                "total_distance_velocity_17_30": [
                    0,
                    np.sqrt(556.25) + np.sqrt(325),
                ],
                "total_distance_acceleration_15_37": [
                    0,
                    np.sqrt(149),
                ],
                "total_distance_acceleration_0_0.75": [np.sqrt(10) + np.sqrt(10), 0],
            },
        )
        self.td_diff = TrackingData(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "home_1_vx": [1, 2, 5, 1, np.nan, -3.0],
                "home_1_vy": [1, 2, -5, 1, np.nan, 1.0],
                "home_1_velocity": [
                    np.sqrt(2),
                    np.sqrt(8),
                    np.sqrt(25),
                    np.sqrt(2),
                    np.nan,
                    np.sqrt(10),
                ],
            },
            provider="test",
            frame_rate=1,
        )
        self.expected_output_vel = TrackingData(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "home_1_vx": [10.0, -20.0, 10.0, np.nan, 10.0, np.nan],
                "home_1_vy": [7.0, -12.5, 9.0, np.nan, 15.0, np.nan],
                "home_1_velocity": [
                    np.sqrt(149),
                    np.sqrt(400 + 12.5**2),
                    np.sqrt(181),
                    np.nan,
                    np.sqrt(325),
                    np.nan,
                ],
            },
            provider="test",
            frame_rate=1,
        )

        self.expected_output_acc = TrackingData(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "home_1_vx": [1, 2, 5, 1, np.nan, -3.0],
                "home_1_vy": [1, 2, -5, 1, np.nan, 1.0],
                "home_1_velocity": [
                    np.sqrt(2),
                    np.sqrt(8),
                    np.sqrt(25),
                    np.sqrt(2),
                    np.nan,
                    np.sqrt(10),
                ],
                "home_1_ax": [1.0, 2.0, -0.5, np.nan, -2.0, np.nan],
                "home_1_ay": [1.0, -3, -0.5, np.nan, 0.0, np.nan],
                "home_1_acceleration": [
                    np.sqrt(2),
                    np.sqrt(13),
                    np.sqrt(0.5),
                    np.nan,
                    np.sqrt(4),
                    np.nan,
                ],
            },
            provider="test",
            frame_rate=1,
        )
        self.td_pitch_control = TrackingData(
            {
                "home_1_x": [1.0, 1.0],
                "home_1_y": [2.0, 2.0],
                "home_1_vx": [0.5, 0.5],
                "home_1_vy": [0.5, 2.0],
                "away_1_x": [3.0, 4.0],
                "away_1_y": [4.0, 3.0],
                "away_1_vx": [0.5, 6.0],
                "away_1_vy": [0.5, 5.0],
                "ball_x": [5.0, 22],
                "ball_y": [6.0, 22],
                "home_2_x": [23.0, 19.0],
                "home_2_y": [42.0, 12.0],
                "home_2_vx": [np.nan, 12.0],
                "home_2_vy": [np.nan, -8.0],
                "home_3_x": [-23.0, 21],
                "home_3_y": [-42.0, 1],
                "home_3_vx": [11, -6.5],
                "home_3_vy": [2, 2.22],
            },
            index=[0, 1],
            provider="test",
            frame_rate=1,
        )

        self.td_pressure = TrackingData(
            {
                "home_1_x": [1.0],
                "home_1_y": [1.0],
                "away_1_x": [2.0],
                "away_1_y": [2.0],
                "away_2_x": [3.0],
                "away_2_y": [3.0],
                "away_3_x": [40.0],
                "away_3_y": [30.0],
            },
            provider="test",
            frame_rate=1,
        )

        self.frame_pressure = pd.Series(
            {
                "home_1_x": 1.0,
                "home_1_y": 1.0,
                "away_1_x": 2.0,
                "away_1_y": 2.0,
                "away_2_x": 3.0,
                "away_2_y": 3.0,
                "away_3_x": 40.0,
                "away_3_y": 30.0,
            }
        )

        self.td_team_possession = TrackingData(
            {
                "event_id": [MISSING_INT, 1, 6, MISSING_INT, 8, MISSING_INT],
                "ball_possession": [None, None, None, None, None, None],
            },
            provider="test",
            frame_rate=1,
        )

        self.event_data = pd.DataFrame(
            {
                "event_id": [1, 3, 6, 7, 8],
                "databallpy_event": ["pass", "tackle", "pass", "interception", "pass"],
                "team_id": [1, 2, 1, 2, 2],
                "outcome": [1, 1, 1, 1, 1],
            }
        )

    def test_get_total_distance(self):
        tracking_data = self.td_covered_distance.copy().drop(
            columns=[
                col
                for col in self.td_covered_distance.columns
                if "_ax" in col or "_ay" in col or "acceleration" in col
            ]
        )
        expected_output_df = self.expected_total_distance.copy()
        expected_output_df.drop(
            columns=[
                col
                for col in expected_output_df.columns
                if "velocity" in col or "acceleration" in col
            ],
            inplace=True,
        )
        output = tracking_data.get_covered_distance(["home_1", "away_2"])
        pd.testing.assert_frame_equal(output, expected_output_df)

        output2 = tracking_data.get_covered_distance(["home_1"], start_idx=5)
        expected_output_df2 = pd.DataFrame(
            index=["home_1"],
            data={
                "total_distance": [0.0],
            },
        )
        pd.testing.assert_frame_equal(output2, expected_output_df2)

    def test_get_total_and_vel_distance(self):
        tracking_data = self.td_covered_distance.copy()
        expected_output_df = self.expected_total_distance.copy()
        expected_output_df.drop(
            columns=[col for col in expected_output_df.columns if "acceleration" in col],
            inplace=True,
        )
        output = tracking_data.get_covered_distance(
            ["home_1", "away_2"],
            velocity_intervals=self.vel_intervals,
        )
        pd.testing.assert_frame_equal(output, expected_output_df)

    def test_get_total_and_vel_and_acc_distance(self):
        tracking_data = self.td_covered_distance.copy()
        output = tracking_data.get_covered_distance(
            ["home_1", "away_2"],
            velocity_intervals=self.vel_intervals,
            acceleration_intervals=self.acc_intervals,
        )
        pd.testing.assert_frame_equal(output, self.expected_total_distance)

    def test_get_covered_distance_wrong_input(self):
        tracking_data = self.td_covered_distance.copy()
        tracking_data.drop(columns=["home_1_velocity"], inplace=True)
        with self.assertRaises(ValueError):
            tracking_data.get_covered_distance(["home_1", "away_2"])

        tracking_data = self.td_covered_distance.copy()
        tracking_data.drop(columns=["home_1_ax"], inplace=True)
        with self.assertRaises(ValueError):
            tracking_data.get_covered_distance(
                ["home_1", "away_2"],
                velocity_intervals=self.vel_intervals,
                acceleration_intervals=self.acc_intervals,
            )

        tracking_data = self.td_covered_distance.copy()
        with self.assertRaises(TypeError):
            players = "home_1"
            tracking_data.get_covered_distance(players, 1)

        with self.assertRaises(TypeError):
            players = ["home_1", 123]
            tracking_data.get_covered_distance([players], 1)

        with self.assertRaises(ValueError):
            tracking_data.get_covered_distance(
                ["home_1", "away_2"],
                start_idx=5,
                end_idx=3,
            )

        with self.assertRaises(ValueError):
            tracking_data.get_covered_distance(
                ["home_1", "away_2"],
                start_idx=5,
                end_idx=6,
            )

        with self.assertRaises(TypeError):
            tracking_data.get_covered_distance(
                ["home_1", "away_2"],
                start_idx=5.0,
                end_idx=6,
            )

    def test_get_velocity(self):
        tracking_data = self.td_diff.copy()
        tracking_data.add_velocity(["home_1"])
        pd.testing.assert_frame_equal(tracking_data, self.expected_output_vel)

        with self.assertRaises(ValueError):
            tracking_data.add_velocity(["home_1"], filter_type="test")

    def test_get_acceleration(self):
        tracking_data = self.td_diff.copy()
        tracking_data.drop(columns=["home_1_vx"], inplace=True)
        with self.assertRaises(ValueError):
            tracking_data.add_acceleration(["home_1"])

        tracking_data = self.td_diff.copy()
        tracking_data.add_acceleration("home_1")
        pd.testing.assert_frame_equal(tracking_data, self.expected_output_acc)

        with self.assertRaises(ValueError):
            tracking_data.add_acceleration(["home_1"], filter_type="wrong")

    def test_filter_tracking_data_input_types(self):
        tracking_data = TrackingData(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "ball_x": [10, 20, -30, 40, np.nan, 60],
                "ball_y": [5, 12, -20, 30, np.nan, 60],
                "away_13_x": [10, 20, -30, 40, np.nan, 60],
                "away_13_y": [5, 12, -20, 30, np.nan, 60],
            },
            provider="test",
            frame_rate=1,
        )

        with self.assertRaises(ValueError):
            tracking_data.filter_tracking_data(column_ids=[])
        with self.assertRaises(TypeError):
            tracking_data.filter_tracking_data(column_ids="home_1", inplace="True")
        with self.assertRaises(TypeError):
            tracking_data.filter_tracking_data(column_ids="home_1", window_length=3.5)
        with self.assertRaises(TypeError):
            tracking_data.filter_tracking_data(column_ids="home_1", polyorder=2.5)
        with self.assertRaises(ValueError):
            tracking_data.filter_tracking_data(
                column_ids="home_1", filter_type="invalid"
            )

    def test_filter_tracking_data_ma(self):
        tracking_data = TrackingData(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "ball_x": [10, 20, -30, 40, np.nan, 60],
                "ball_y": [5, 12, -20, 30, np.nan, 60],
                "away_13_x": [10, 20, -30, 40, np.nan, 60],
                "away_13_y": [5, 12, -20, 30, np.nan, 60],
            },
            provider="test",
            frame_rate=1,
        )

        tracking_data.filter_tracking_data(
            column_ids=["ball"],
            filter_type="moving_average",
            window_length=2,
        )

        expected_output = TrackingData(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "ball_x": [5, 15, -5, 5, np.nan, np.nan],
                "ball_y": [2.5, 8.5, -4, 5, np.nan, np.nan],
                "away_13_x": [10, 20, -30, 40, np.nan, 60],
                "away_13_y": [5, 12, -20, 30, np.nan, 60],
            },
            provider="test",
            frame_rate=1,
        )
        pd.testing.assert_frame_equal(tracking_data, expected_output)

    def test_filter_tracking_data_sg(self):
        tracking_data = TrackingData(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "ball_x": [10, 20, -30, 40, np.nan, 60],
                "ball_y": [5, 12, -20, 30, np.nan, 60],
                "away_13_x": [10, 20, -30, 40, np.nan, 60],
                "away_13_y": [5, 12, -20, 30, np.nan, 60],
            },
            provider="test",
            frame_rate=1,
        )

        tracking_data.filter_tracking_data(
            column_ids=["ball"],
            filter_type="savitzky_golay",
            window_length=3,
            polyorder=1,
        )

        expected_output = TrackingData(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "ball_x": [20, 0, 10, np.nan, np.nan, np.nan],
                "ball_y": [11.5, -1, 7.333333, np.nan, np.nan, np.nan],
                "away_13_x": [10, 20, -30, 40, np.nan, 60],
                "away_13_y": [5, 12, -20, 30, np.nan, 60],
            },
            provider="test",
            frame_rate=1,
        )

        pd.testing.assert_frame_equal(tracking_data, expected_output)

    def test_approximate_voronoi(self):
        tracking_data = self.td_pitch_control.copy()

        output_dists, output_col_ids = tracking_data.get_approximate_voronoi(
            [20, 20], 5, 3
        )

        assert output_dists.shape == output_col_ids.shape == (2, 3, 5)

        expected_dists = np.array(
            [
                [
                    [12.494443, 10.005554, 8.724168, 9.171211, 11.140516],
                    [9.219544, 5.3851647, 2.236068, 3.6055512, 6.4031243],
                    [10.137938, 6.839428, 4.013865, 2.8480012, 5.6666665],
                ],
                [
                    [12.494443, 10.005554, 8.724168, 9.171211, 10.46157],
                    [9.219544, 5.3851647, 2.236068, 3.0, 5.0],
                    [10.137938, 6.839428, 4.772607, 3.6666667, 5.4262733],
                ],
            ]
        )
        expected_col_ids = np.array(
            [
                [
                    ["home_1", "home_1", "home_1", "home_1", "home_1"],
                    ["home_1", "home_1", "home_1", "home_1", "away_1"],
                    ["home_1", "home_1", "away_1", "away_1", "away_1"],
                ],
                [
                    ["home_1", "home_1", "home_1", "home_1", "away_1"],
                    ["home_1", "home_1", "home_1", "away_1", "away_1"],
                    ["home_1", "home_1", "home_1", "away_1", "away_1"],
                ],
            ],
            dtype="U7",
        )
        np.testing.assert_array_equal(output_col_ids, expected_col_ids)
        np.testing.assert_array_almost_equal(output_dists, expected_dists)

        dists, col_ids = tracking_data.get_approximate_voronoi(
            [20, 20], 5, 3, start_idx=0, end_idx=0
        )

        assert dists.shape == col_ids.shape == (3, 5)
        np.testing.assert_array_equal(col_ids, expected_col_ids[0])
        np.testing.assert_array_almost_equal(dists, expected_dists[0])

    def test_get_pitch_control_period(self):
        tracking_data = self.td_pitch_control.copy()
        output = tracking_data.get_pitch_control([106, 68], 100, 50)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, (2, 50, 100))

    def test_get_individual_player_possession(self):
        tracking_data = TrackingData(
            {
                "ball_x": [0, 0, 5, 10, 10, 11, 22, 24, 26, 26],
                "ball_y": [0, 0, 5, 10, 10, 10, 10, 10, 10, 15],
                "home_1_x": [0, 0, 5, 0, 0, 0, 0, 0, 26, 0],
                "home_1_y": [0, 0, 5, 0, 0, 0, 0, 0, 10.5, 0],
                "away_1_x": [0.5, 0.5, 0.5, 0.5, 10, 11, 18, 24, 26, 26],
                "away_1_y": [0.5, 0.5, 0.5, 0.5, 10, 10, 10, 10, 10, 15],
                "ball_status": [
                    "dead",
                    "alive",
                    "alive",
                    "alive",
                    "alive",
                    "alive",
                    "alive",
                    "alive",
                    "alive",
                    "alive",
                ],
            },
            provider="test",
            frame_rate=1,
        )
        tracking_data.add_velocity("ball")

        expected_possession = np.array(
            [
                None,
                "home_1",
                None,
                None,
                "away_1",
                "away_1",
                "away_1",
                "away_1",
                "away_1",
                None,
            ]
        )

        assert "player_possession" not in tracking_data.columns
        tracking_data.get_individual_player_possession()
        assert "player_possession" in tracking_data.columns
        np.testing.assert_array_equal(
            tracking_data["player_possession"], expected_possession
        )

        with self.assertRaises(ValueError):
            tracking_data.get_individual_player_possession(
                tracking_data.drop(columns=["ball_velocity"], inplace=True)
            )

    def test_get_pressure_on_player_specified_d_front(self):
        tracking_data = self.td_pressure.copy()
        d_front = calculate_variable_dfront(
            tracking_data, "home_1", max_d_front=9, pitch_length=100.0
        )
        z1 = calculate_z(self.frame_pressure, "home_1", "away_1", pitch_length=100.0)
        z2 = calculate_z(self.frame_pressure, "home_1", "away_2", pitch_length=100.0)
        l1 = calculate_l(d_back=3.0, d_front=d_front, z=z1)
        l2 = calculate_l(d_back=3.0, d_front=d_front, z=z2)
        dist1 = math.dist([1, 1], [2, 2])
        dist2 = math.dist([1, 1], [3, 3])

        pres1 = (1 - dist1 / l1) ** 1.75 * 100
        pres2 = (1 - dist2 / l2) ** 1.75 * 100
        expected_pressure = pres1 + pres2

        result = tracking_data.get_pressure_on_player(
            index=0,
            column_id="home_1",
            pitch_size=[100.0, 50.0],
            d_front="variable",
            d_back=3.0,
            q=1.75,
        )
        self.assertAlmostEqual(result, expected_pressure, places=4)

    def test_get_pressure_on_player_variable_d_front(self):
        tracking_data = self.td_pressure.copy()
        z1 = calculate_z(self.frame_pressure, "home_1", "away_1", pitch_length=100.0)
        z2 = calculate_z(self.frame_pressure, "home_1", "away_2", pitch_length=100.0)
        l1 = calculate_l(d_back=3.0, d_front=9.0, z=z1)
        l2 = calculate_l(d_back=3.0, d_front=9.0, z=z2)
        dist1 = math.dist([1, 1], [2, 2])
        dist2 = math.dist([1, 1], [3, 3])

        pres1 = (1 - dist1 / l1) ** 1.75 * 100
        pres2 = (1 - dist2 / l2) ** 1.75 * 100
        expected_pressure = pres1 + pres2

        result = tracking_data.get_pressure_on_player(
            index=0,
            column_id="home_1",
            pitch_size=[100.0, 50.0],
            d_front=9.0,
            d_back=3.0,
            q=1.75,
        )
        self.assertAlmostEqual(result, expected_pressure, places=4)

    def get_pressure_on_player_wrong_input(self):
        tracking_data: TrackingData = self.td_pressure.copy()
        with self.assertRaises(ValueError):
            tracking_data.get_pressure_on_player(
                index=-100,
                column_id="home_1",
                pitch_size=[100.0, 50.0],
                d_front=9.0,
                d_back=3.0,
                q=1.75,
            )

    def test_add_team_possession(self):
        tracking_data = self.td_team_possession.copy()

        tracking_data.add_team_possession(self.event_data, 1)
        self.assertEqual(
            tracking_data["ball_possession"].tolist(),
            ["home", "home", "home", "home", "away", "away"],
        )
        assert not tracking_data.equals(self.td_team_possession)

    def test_add_team_possession_no_event_id(self):
        tracking_data = self.td_team_possession.copy()
        with self.assertRaises(ValueError):
            tracking_data.add_team_possession(self.event_data, 22)

        tracking_data.drop(columns=["event_id"], inplace=True)
        with self.assertRaises(ValueError):
            tracking_data.add_team_possession(self.event_data, 1)
