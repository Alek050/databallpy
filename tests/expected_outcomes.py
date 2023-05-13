import datetime as dt

import numpy as np
import pandas as pd

from databallpy.load_data.metadata import Metadata

TD_TRACAB = pd.DataFrame(
    {
        "frame": [1509993, 1509994, 1509995, 1509996, 1509997],
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
            "start_datetime_td": [
                pd.to_datetime(
                    "2023-01-14",
                ).tz_localize("Europe/Amsterdam")
                + dt.timedelta(milliseconds=int((1509993 / 25) * 1000)),
                pd.to_datetime("2023-01-14").tz_localize("Europe/Amsterdam")
                + dt.timedelta(milliseconds=int((1509996 / 25) * 1000)),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
            ],
            "end_datetime_td": [
                pd.to_datetime("2023-01-14").tz_localize("Europe/Amsterdam")
                + dt.timedelta(milliseconds=int((1509994 / 25) * 1000)),
                pd.to_datetime("2023-01-14").tz_localize("Europe/Amsterdam")
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
    home_formation="",
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
    away_formation="",
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
    country="",
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
            2512690515,
        ],
        "type_id": [34, 32, 32, 1, 1, 100, 43, 3, 7, 16],
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
            "own goal",
        ],
        "period_id": [16, 1, 1, 1, 1, 1, 2, 2, 2, 1],
        "minutes": [0, 0, 0, 0, 0, 0, 30, 30, 31, 9],
        "seconds": [0, 0, 0, 1, 4, 6, 9, 10, 10, 17],
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
            45849,
        ],
        "team_id": [194, 3, 194, 3, 3, 3, 194, 3, 194, 3],
        "outcome": [1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
        # field dimensions are [10, 10], for opta its standard [100, 100].
        # So all vallues should be divided by 10 and minus  5 to get the
        # standard databallpy values.
        "start_x": [5.0, -5.0, 5.0, -0.03, -1.84, -1.9, 5.0, 1.57, 1.57, -4.05],
        "start_y": [5.0, -5.0, 5.0, 0.01, -0.93, -0.57, 5.0, -2.68, -2.68, 0.28],
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
                "2023-01-22T12:18:44.120",
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
            "Jan Boskamp",
        ],
    }
)

ED_OPTA["datetime"] = pd.to_datetime(ED_OPTA["datetime"]).dt.tz_localize(
    "Europe/Amsterdam"
)

MD_OPTA = Metadata(
    match_id=1908,
    pitch_dimensions=[10.0, 10.0],
    periods_frames=pd.DataFrame(
        {
            "period": [1, 2, 3, 4, 5],
            "start_datetime_ed": [
                pd.to_datetime("2023-01-22T12:18:32.000").tz_localize(
                    "Europe/Amsterdam"
                ),
                pd.to_datetime("2023-01-22T13:21:13.000").tz_localize(
                    "Europe/Amsterdam"
                ),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
            ],
            "end_datetime_ed": [
                pd.to_datetime("2023-01-22T13:04:32.000").tz_localize(
                    "Europe/Amsterdam"
                ),
                pd.to_datetime("2023-01-22T14:09:58.000").tz_localize(
                    "Europe/Amsterdam"
                ),
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
    country="Netherlands",
)

TD_METRICA = pd.DataFrame(
    {
        "frame": [1, 2, 3, 4, 5, 6],
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
            pd.to_datetime("2019-02-21T03:30:07.000", utc=True),
            pd.to_datetime("2019-02-21T03:30:08.000", utc=True),
            pd.to_datetime("2019-02-21T03:30:09.000", utc=True),
        ],
    }
)
MD_METRICA_TD = Metadata(
    match_id=9999,
    pitch_dimensions=[100.0, 50.0],
    periods_frames=pd.DataFrame(
        {
            "period": [1, 2, 3, 4, 5],
            "start_frame": [1, 4, -999, -999, -999],
            "end_frame": [3, 6, -999, -999, -999],
            "start_datetime_td": [
                pd.to_datetime("2019-02-21T03:30:07.000", utc=True),
                pd.to_datetime("2019-02-21T03:30:08.500", utc=True),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
            ],
            "end_datetime_td": [
                pd.to_datetime("2019-02-21T03:30:08.000", utc=True),
                pd.to_datetime("2019-02-21T03:30:09.500", utc=True),
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
    country="",
)


MD_METRICA_ED = Metadata(
    match_id=9999,
    pitch_dimensions=[100.0, 50.0],
    periods_frames=pd.DataFrame(
        {
            "period": [1, 2, 3, 4, 5],
            "start_frame": [1, 4, -999, -999, -999],
            "end_frame": [3, 6, -999, -999, -999],
            "start_datetime_ed": [
                pd.to_datetime("2019-02-21T03:30:07.000", utc=True),
                pd.to_datetime("2019-02-21T03:30:08.500", utc=True),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
            ],
            "end_datetime_ed": [
                pd.to_datetime("2019-02-21T03:30:08.000", utc=True),
                pd.to_datetime("2019-02-21T03:30:09.500", utc=True),
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
    country="",
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

TD_INMOTIO = pd.DataFrame(
    {
        "frame": [1, 2, 3, 4, 5, 6],
        "ball_x": [0.0, 0.0, -1.0, np.nan, 0.0, 0.0],
        "ball_y": [0.0, 0.0, 1.0, np.nan, 0.0, 0.0],
        "ball_z": [0.0, 0.0, 0.0, np.nan, 0.0, 0.0],
        "ball_status": ["dead", "alive", "alive", np.nan, "alive", "alive"],
        "ball_posession": [None, None, None, None, None, None],
        "home_1_x": [-46.9, -45.9, -44.9, np.nan, 39.0, 39.0],
        "home_1_y": [0.8, -0.2, -1.2, np.nan, 1.5, 2.5],
        "home_2_x": [-19.0, -20.0, -21.0, np.nan, 23.3, 23.3],
        "home_2_y": [-6.0, -5.0, -6.0, np.nan, 6.9, 6.9],
        "away_1_x": [40.0, 39.0, 38.0, np.nan, -45.9, -44.9],
        "away_1_y": [0.5, 1.5, 2.5, np.nan, -0.2, -1.2],
        "away_2_x": [23.3, 23.3, 25.3, np.nan, -20.0, np.nan],
        "away_2_y": [5.9, 6.9, 5.9, np.nan, -5.0, np.nan],
        "period": [-999, 1, 1, -999, 2, 2],
        "matchtime_td": ["", "00:00", "00:00", "Break", "45:00", "45:00"],
    }
)

MD_INMOTIO = Metadata(
    match_id=9999,
    pitch_dimensions=[100.0, 50.0],
    periods_frames=pd.DataFrame(
        {
            "period": [1, 2, 3, 4, 5],
            "start_frame": [2, 5, -999, -999, -999],
            "end_frame": [3, 6, -999, -999, -999],
            "start_datetime_td": [
                pd.to_datetime("2023-01-01T20:00:00.000").tz_localize(
                    "Europe/Amsterdam"
                ),
                pd.to_datetime("2023-01-01T21:00:00.000").tz_localize(
                    "Europe/Amsterdam"
                ),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
            ],
            "end_datetime_td": [
                pd.to_datetime("2023-01-01T20:45:00.000").tz_localize(
                    "Europe/Amsterdam"
                ),
                pd.to_datetime("2023-01-01T21:45:00.0000").tz_localize(
                    "Europe/Amsterdam"
                ),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
            ],
        }
    ),
    frame_rate=25,
    home_team_id="T-0001",
    home_team_name="Team A",
    home_players=pd.DataFrame(
        {
            "id": [1, 2],
            "full_name": ["Player 1", "Player 2"],
            "shirt_num": [1, 2],
            "player_type": ["Goalkeeper", "Field player"],
            "start_frame": [2, 2],
            "end_frame": [6, 6],
        }
    ),
    home_formation="",
    home_score=1,
    away_team_id="T-0002",
    away_team_name="Team B",
    away_players=pd.DataFrame(
        {
            "id": [3, 4],
            "full_name": ["Player 11", "Player 12"],
            "shirt_num": [1, 2],
            "player_type": ["Goalkeeper", "Field player"],
            "start_frame": [2, 2],
            "end_frame": [6, 6],
        }
    ),
    away_formation="",
    away_score=1,
    country="",
)

MD_INSTAT = Metadata(
    match_id=9999,
    pitch_dimensions=[np.nan, np.nan],
    periods_frames=pd.DataFrame(
        {
            "period": [1, 2, 3, 4, 5],
            "start_datetime_ed": [
                pd.to_datetime("2023-01-01 20:00:00").tz_localize("Europe/Amsterdam"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
            ],
            "end_datetime_ed": [
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
            ],
        }
    ),
    frame_rate=np.nan,
    home_team_id=1,
    home_team_name="Team A",
    home_players=pd.DataFrame(
        {
            "id": [1, 2],
            "full_name": ["Player 1", "Player 2"],
            "position": ["Goalkeeper", "Defender"],
            "starter": [True, True],
            "shirt_num": [1, 2],
        }
    ),
    home_formation="352",
    home_score=0,
    away_team_id=2,
    away_team_name="Team B",
    away_players=pd.DataFrame(
        {
            "id": [3, 4],
            "full_name": ["Player 11", "Player 12"],
            "position": ["Goalkeeper", "Substitute player"],
            "starter": [True, False],
            "shirt_num": [1, 3],
        }
    ),
    away_formation="442",
    away_score=2,
    country="Netherlands",
)

ED_INSTAT = pd.DataFrame(
    {
        "event_id": [10107, 10110, 10111, 10112],
        "type_id": [1012, 1011, 2010, 2011],
        "event": ["pass", "pass", "Challenge", "Challenge"],
        "period_id": [1, 1, 1, 1],
        "minutes": [0.0, 0.0, 0.0, 0.0],
        "seconds": [5.37, 20.93, 23.64, 28.64],
        "player_id": [2, 1, 3, -999],
        "team_id": [1, 1, 2, -999],
        "outcome": [0, 1, np.nan, np.nan],
        "start_x": [0, 35.4, 35.5, np.nan],
        "start_y": [0, -18.0, -40.5, np.nan],
        "end_x": [-20.1, 40.5, np.nan, np.nan],
        "end_y": [43.0, -22.5, np.nan, np.nan],
        "datetime": [
            pd.to_datetime("2023-01-01 20:00:05.370").tz_localize("Europe/Amsterdam"),
            pd.to_datetime("2023-01-01 20:00:20.930").tz_localize("Europe/Amsterdam"),
            pd.to_datetime("2023-01-01 20:00:23.640").tz_localize("Europe/Amsterdam"),
            pd.to_datetime("2023-01-01 20:00:28.640").tz_localize("Europe/Amsterdam"),
        ],
        "instat_event": [
            "Attacking pass inaccurate",
            "Attacking pass accurate",
            "Challenge",
            "Challenge",
        ],
        "player_name": ["Player 2", "Player 1", "Player 11", np.nan],
    }
)

RES_SIM_MAT = np.array(
    [
        0.40006852,
        0.40006852,
        0.39676846,
        0.42604787,
        0.39410369,
        0.39410369,
        0.3922753,
        0.42664872,
        0.38802166,
        0.38802166,
        0.38767596,
        0.42703609,
        0.36787944,
        0.36787944,
        0.36787944,
        0.36787944,
        0.37342119,
        0.37342119,
        0.37878323,
        0.42755615,
        0.39987463,
        0.39987463,
        0.40133888,
        0.42945239,
        0.39263378,
        0.39263378,
        0.39708269,
        0.42795639,
        0.38521479,
        0.38521479,
        0.39260596,
        0.41395895,
        0.37126589,
        0.37126589,
        0.38263939,
        0.40387744,
        0.39703003,
        0.39703003,
        0.40487744,
        0.4017941,
        0.38921197,
        0.38921197,
        0.39914273,
        0.41462662,
        0.38150169,
        0.38150169,
        0.39270907,
        0.4194602,
        0.36787944,
        0.36787944,
        0.38141911,
        0.41999494,
    ]
)

RES_SIM_MAT_NO_PLAYER = np.array(
    [
        0.40006852,
        0.40006852,
        0.86247631,
        0.42604787,
        0.39410369,
        0.39410369,
        0.8646367,
        0.42664872,
        0.38802166,
        0.38802166,
        0.86681802,
        0.42703609,
        0.36787944,
        0.36787944,
        0.36787944,
        0.36787944,
        0.37342119,
        0.37342119,
        0.87099448,
        0.42755615,
        0.39987463,
        0.39987463,
        0.87109937,
        0.42945239,
        0.39263378,
        0.39263378,
        0.87328136,
        0.42795639,
        0.38521479,
        0.38521479,
        0.87548448,
        0.41395895,
        0.37126589,
        0.37126589,
        0.87795412,
        0.40387744,
        0.39703003,
        0.39703003,
        0.87805984,
        0.4017941,
        0.38921197,
        0.38921197,
        0.87850958,
        0.41462662,
        0.38150169,
        0.38150169,
        0.87722815,
        0.4194602,
        0.36787944,
        0.36787944,
        0.87620901,
        0.41999494,
    ]
)
