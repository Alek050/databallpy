import datetime as dt

import numpy as np
import pandas as pd

from databallpy.load_data.metadata import Metadata

TD_TRACAB = pd.DataFrame(
    {
        "timestamp": [1509993, 1509994, 1509995, 1509996, 1509997],
        "ball_x": [1.50, 1.81, 2.13, np.nan, 2.76],
        "ball_y": [-0.43, -0.49, -0.56, np.nan, -0.70],
        "ball_z": [0.07, 0.09, 0.11, np.nan, 0.15],
        "ball_status": ["alive", "dead", "alive", np.nan, "alive"],
        "ball_posession": ["away", "away", "away", np.nan, "home"],
        "home_34_x": [-13.50, -13.50, -13.50, np.nan, -13.49],
        "home_34_y": [-4.75, -4.74, -4.73, np.nan, -4.72],
        "away_17_x": [13.22, 13.21, 13.21, np.nan, 13.21],
        "away_17_y": [-13.16, -13.16, -13.17, np.nan, -13.18],
        "period": [1, 1, -999, 2, 2],
        "matchtime_td": [
            "00:00",
            "00:00",
            "Break",
            "45:00",
            "45:00",
        ],
    }
)


MD_TRACAB = Metadata(
    match_id=1908,
    pitch_dimensions=[100.0, 50.0],
    periods_frames=pd.DataFrame(
        {
            "period": [1, 2, 3, 4, 5],
            "start_frame": [1509993, 1509996, -999, -999, -999],
            "end_frame": [1509994, 1509997, -999, -999, -999],
            "start_time_td": [
                pd.to_datetime("2023-01-14")
                + dt.timedelta(milliseconds=int((1509993 / 25) * 1000)),
                pd.to_datetime("2023-01-14")
                + dt.timedelta(milliseconds=int((1509996 / 25) * 1000)),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
            ],
            "end_time_td": [
                pd.to_datetime("2023-01-14")
                + dt.timedelta(milliseconds=int((1509994 / 25) * 1000)),
                pd.to_datetime("2023-01-14")
                + dt.timedelta(milliseconds=int((1509997 / 25) * 1000)),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
            ],
        }
    ),
    frame_rate=25,
    home_team_id=3,
    home_team_name="TeamOne",
    home_formation=None,
    home_score=np.nan,
    home_players=pd.DataFrame(
        {
            "id": [19367, 45849],
            "full_name": ["Piet Schrijvers", "Jan Boskamp"],
            "shirt_num": [1, 2],
            "start_frame": [1509993, 1509993],
            "end_frame": [1509997, 1509995],
        }
    ),
    away_team_id=194,
    away_team_name="TeamTwo",
    away_formation=None,
    away_score=np.nan,
    away_players=pd.DataFrame(
        {
            "id": [184934, 450445],
            "full_name": ["Pepijn Blok", "TestSpeler"],
            "shirt_num": [1, 2],
            "start_frame": [1509993, 1509993],
            "end_frame": [1509997, 1509994],
        }
    ),
)
ED_OPTA = pd.DataFrame(
    {
        "event_id": [
            2499582269,
            2499594199,
            2499594195,
            2499594225,
            2499594243,
            2499594271,
            2499594279,
            2499594285,
            2499594291,
        ],
        "type_id": [34, 32, 32, 1, 1, 100, 43, 3, 7],
        "event": [
            "team set up",
            "start",
            "start",
            "pass",
            "pass",
            None,
            "deleted event",
            "take on",
            "tackle",
        ],
        "period_id": [16, 1, 1, 1, 1, 1, 2, 2, 2],
        "minutes": [0, 0, 0, 0, 0, 0, 30, 30, 31],
        "seconds": [0, 0, 0, 1, 4, 6, 9, 10, 10],
        "player_id": [
            np.nan,
            np.nan,
            np.nan,
            19367,
            45849,
            45849,
            184934,
            45849,
            184934,
        ],
        "team_id": [194, 3, 194, 3, 3, 3, 194, 3, 194],
        "outcome": [1, 1, 1, 1, 0, 0, 1, 0, 1],
        # field dimensions should be [10, 10] to scale down all values by
        # a factor of 10
        "start_x": [5.0, -5.0, 5.0, -0.03, -1.84, -1.9, 5.0, 1.57, 1.57],
        "start_y": [5.0, -5.0, 5.0, 0.01, -0.93, -0.57, 5.0, -2.68, -2.68],
        "datetime": np.array(
            [
                "2023-01-22T11:28:32.117",
                "2023-01-22T12:18:32.152",
                "2023-01-22T12:18:32.152",
                "2023-01-22T12:18:33.637",
                "2023-01-22T12:18:36.207",
                "2023-01-22T12:18:39.109",
                "2023-01-22T12:18:41.615",
                "2023-01-22T12:18:43.119",
                "2023-01-22T12:18:43.120",
            ],
            dtype="datetime64",
        ),
        "player_name": [
            None,
            None,
            None,
            "Piet Schrijvers",
            "Jan Boskamp",
            "Jan Boskamp",
            "Pepijn Blok",
            "Jan Boskamp",
            "Pepijn Blok",
        ],
    }
)
MD_OPTA = Metadata(
    match_id=1908,
    pitch_dimensions=[10.0, 10.0],
    periods_frames=pd.DataFrame(
        {
            "period": [1, 2, 3, 4, 5],
            "start_datetime_opta": [
                pd.to_datetime("2023-01-22T12:18:32.000"),
                pd.to_datetime("2023-01-22T13:21:13.000"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
            ],
            "end_datetime_opta": [
                pd.to_datetime("2023-01-22T13:04:32.000"),
                pd.to_datetime("2023-01-22T14:09:58.000"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
            ],
        }
    ),
    frame_rate=np.nan,
    home_team_id=3,
    home_team_name="TeamOne",
    home_formation="4231",
    home_score=3,
    home_players=pd.DataFrame(
        {
            "id": [19367, 45849],
            "full_name": ["Piet Schrijvers", "Jan Boskamp"],
            "formation_place": [4, 0],
            "position": ["midfielder", "midfielder"],
            "starter": [True, False],
            "shirt_num": [1, 2],
        }
    ),
    away_team_id=194,
    away_team_name="TeamTwo",
    away_formation="3412",
    away_score=1,
    away_players=pd.DataFrame(
        {
            "id": [184934, 450445],
            "full_name": ["Pepijn Blok", "TestSpeler"],
            "formation_place": [8, 0],
            "position": ["midfielder", "midfielder"],
            "starter": [True, False],
            "shirt_num": [1, 2],
        }
    ),
)
TD_METRICA = pd.DataFrame(
    {
        "timestamp": [1, 2, 3, 4, 5, 6],
        "ball_x": [np.nan, 0, 40, np.nan, np.nan, -40],
        "ball_y": [np.nan, 0, -20, np.nan, np.nan, 20],
        "ball_z": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "ball_status": ["dead", "alive", "alive", np.nan, "dead", "alive"],
        "ball_posession": [None, None, None, None, None, None],
        "home_11_x": [0, 30, 20, np.nan, np.nan, np.nan],
        "home_11_y": [0, 0, -5, np.nan, np.nan, np.nan],
        "home_1_x": [-40, -30, -20, np.nan, -10, 0],
        "home_1_y": [5, 0, -5, np.nan, 0, -20],
        "away_34_x": [20, -10, 0, np.nan, 0, 20],
        "away_34_y": [-5, 20, 10, np.nan, 5, 0],
        "away_35_x": [np.nan, np.nan, np.nan, np.nan, 10, 20],
        "away_35_y": [np.nan, np.nan, np.nan, np.nan, 5, 0],
        "period": [1, 1, 1, 2, 2, 2],
        "matchtime_td": ["00:00", "00:00", "00:01", "45:00", "45:00", "45:01"],
    }
)

ED_METRICA = pd.DataFrame(
    {
        "event_id": [3, 4, 5],
        "type_id": [5, 1, 10],
        "event": ["set piece", "pass", "carry"],
        "period_id": [1, 1, 2],
        "minutes": [0, 1, 1],
        "seconds": [14.44, 4.22, 15.08],
        "player_id": [3578, 3699, 3568],
        "player_name": ["Player 11", "Player 34", "Player 1"],
        "team_id": ["FIFATMA", "FIFATMB", "FIFATMA"],
        "outcome": [np.nan, np.nan, np.nan],
        "start_x": [np.nan, 0.0, 20.0],
        "start_y": [np.nan, -5.0, 5],
        "to_player_id": [np.nan, 3700, np.nan],
        "to_player_name": [None, "Player 35", None],
        "end_x": [np.nan, -20.0, -40.0],
        "end_y": [np.nan, -15.0, 0.0],
        "td_frame": [1, 3, 5],
        "datetime": [
            pd.to_datetime("2019-02-21T03:30:07.000"),
            pd.to_datetime("2019-02-21T03:30:08.000"),
            pd.to_datetime("2019-02-21T03:30:09.000"),
        ],
    }
)
MD_METRICA = Metadata(
    match_id=9999,
    pitch_dimensions=[100.0, 50.0],
    periods_frames=pd.DataFrame(
        {
            "period": [1, 2, 3, 4, 5],
            "start_frame": [1, 4, -999, -999, -999],
            "end_frame": [3, 6, -999, -999, -999],
            "start_time_td": [
                pd.to_datetime("2019-02-21T03:30:07.000"),
                pd.to_datetime("2019-02-21T03:30:08.500"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
            ],
            "end_time_td": [
                pd.to_datetime("2019-02-21T03:30:08.000"),
                pd.to_datetime("2019-02-21T03:30:09.500"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
            ],
        }
    ),
    frame_rate=2,
    home_team_id="FIFATMA",
    home_team_name="Team A",
    home_players=pd.DataFrame(
        {
            "id": [3578, 3568],
            "full_name": ["Player 11", "Player 1"],
            "formation_place": [0, 1],
            "position": ["Goalkeeper", "Right Back"],
            "starter": [True, True],
            "shirt_num": [11, 1],
        }
    ),
    home_formation="1100",
    home_score=0,
    away_team_id="FIFATMB",
    away_team_name="Team B",
    away_players=pd.DataFrame(
        {
            "id": [3699, 3700],
            "full_name": ["Player 34", "Player 35"],
            "formation_place": [3, 1],
            "position": ["Left Forward (2)", "Left Back"],
            "starter": [True, False],
            "shirt_num": [34, 35],
        }
    ),
    away_formation="0001",
    away_score=2,
)
TD_CHANNELS_METRICA = pd.DataFrame(
    {
        "start": [1, 5],
        "end": [3, 6],
        "ids": [
            ["home_11", "home_1", "away_34"],
            ["home_1", "away_34", "away_35"],
        ],
    }
)
