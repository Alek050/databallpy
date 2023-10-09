import datetime as dt

import numpy as np
import pandas as pd

from databallpy.load_data.event_data.dribble_event import DribbleEvent
from databallpy.load_data.event_data.pass_event import PassEvent
from databallpy.load_data.event_data.shot_event import ShotEvent
from databallpy.load_data.metadata import Metadata
from databallpy.utils.utils import MISSING_INT

TD_TRACAB = pd.DataFrame(
    {
        "frame": [1509993, 1509994, 1509995, 1509996, 1509997],
        "ball_x": [1.50, 1.81, 2.13, np.nan, 2.76],
        "ball_y": [
            -0.43,
            -0.49,
            -0.56,
            np.nan,
            -0.70,
        ],
        "ball_z": [0.07, 0.09, 0.11, np.nan, 0.15],
        "ball_status": [
            "alive",
            "dead",
            "alive",
            None,
            "alive",
        ],
        "ball_possession": [
            "away",
            "away",
            "away",
            None,
            "home",
        ],
        "home_34_x": [
            -13.50,
            -13.50,
            -13.50,
            np.nan,
            -13.49,
        ],
        "home_34_y": [
            -4.75,
            -4.74,
            -4.73,
            np.nan,
            -4.72,
        ],
        "away_17_x": [
            13.22,
            13.21,
            13.21,
            np.nan,
            13.21,
        ],
        "away_17_y": [
            -13.16,
            -13.16,
            -13.17,
            np.nan,
            -13.18,
        ],
        "period": [1, 1, MISSING_INT, 2, 2],
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
            "start_frame": [1509993, 1509996, MISSING_INT, MISSING_INT, MISSING_INT],
            "end_frame": [1509994, 1509997, MISSING_INT, MISSING_INT, MISSING_INT],
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
    home_score=MISSING_INT,
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
    away_score=MISSING_INT,
    away_players=pd.DataFrame(
        {
            "id": [184934, 450445],
            "full_name": ["Pepijn Blok", "TestSpeler"],
            "shirt_num": [1, 2],
            "start_frame": [1509993, 1509993],
            "end_frame": [1509997, 1509995],
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
            2512690516,
            2512690517,
        ],
        "type_id": [34, 32, 32, 1, 1, 100, 43, 3, 7, 16, 16, 15],
        "databallpy_event": [
            None,
            None,
            None,
            "pass",
            "pass",
            None,
            None,
            "dribble",
            None,
            "own_goal",
            "shot",
            "shot",
        ],
        "period_id": [16, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1],
        "minutes": [0, 0, 0, 0, 0, 0, 30, 30, 31, 9, 9, 9],
        "seconds": [0, 0, 0, 1, 4, 6, 9, 10, 10, 17, 17, 17],
        "player_id": [
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            19367,
            184934,
            45849,
            184934,
            45849,
            184934,
            45849,
            184934,
            184934,
        ],
        "team_id": [194, 3, 194, 3, 194, 3, 194, 3, 194, 3, 194, 194],
        "outcome": [
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            1,
            0,
            MISSING_INT,
            MISSING_INT,
            0,
            MISSING_INT,
            1,
            1,
            0,
        ],
        # field dimensions are [10, 10], for opta its standard [100, 100].
        # So all values should be divided by 10 and minus  5 to get the
        # standard databallpy values.
        "start_x": [
            5.0,
            -5.0,
            5.0,
            -0.03,
            1.84,
            -1.9,
            5.0,
            1.57,
            1.57,
            -4.05,
            4.05,
            4.05,
        ],
        "start_y": [
            5.0,
            -5.0,
            5.0,
            0.01,
            0.93,
            -0.57,
            5.0,
            -2.68,
            -2.68,
            0.28,
            -0.28,
            -0.28,
        ],
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
                "2023-01-22T12:18:44.120",
                "2023-01-22T12:18:44.120",
            ],
            dtype="datetime64[ns]",
        ),
        "opta_event": [
            "team set up",
            "start",
            "start",
            "pass",
            "pass",
            "unknown event",
            "deleted event",
            "take on",
            "tackle",
            "own goal",
            "goal",
            "attempt saved",
        ],
        "opta_id": [1, 2, 2, 3, 22, 5, 6, 7, 8, 9, 10, 11],
        "player_name": [
            None,
            None,
            None,
            "Piet Schrijvers",
            "Pepijn Blok",
            "Jan Boskamp",
            "Pepijn Blok",
            "Jan Boskamp",
            "Pepijn Blok",
            "Jan Boskamp",
            "Pepijn Blok",
            "Pepijn Blok",
        ],
    }
)

ED_OPTA["datetime"] = pd.to_datetime(ED_OPTA["datetime"]).dt.tz_localize(
    "Europe/Amsterdam"
)

SHOT_EVENTS_OPTA = {
    2512690515: ShotEvent(
        event_id=2512690515,
        period_id=1,
        minutes=9,
        seconds=17,
        datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True).tz_convert(
            "Europe/Amsterdam"
        ),
        start_x=9.5 / 100 * 106 - 53,  # standard opta pitch dimensions = [106, 68]
        start_y=52.8 / 100 * 68 - 34,
        team_id=3,
        z_target=18.4 / 100 * 2.44,
        y_target=54.3 / 100 * 7.32 - (7.32 / 2),
        player_id=45849,
        shot_outcome="own_goal",
        body_part="head",
        type_of_play="corner_kick",
        first_touch=False,
        created_oppertunity="regular_play",
        related_event_id=MISSING_INT,
        ball_goal_distance=np.nan,
        ball_gk_distance=np.nan,
        shot_angle=np.nan,
        gk_angle=np.nan,
        pressure_on_ball=np.nan,
        n_obstructive_players=MISSING_INT,
    ),
    2512690516: ShotEvent(
        event_id=2512690516,
        period_id=1,
        minutes=9,
        seconds=17,
        datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True).tz_convert(
            "Europe/Amsterdam"
        ),
        start_x=(9.5 / 100 * 106 - 53)
        * -1,  # standard opta pitch dimensions = [106, 68]
        start_y=(52.8 / 100 * 68 - 34) * -1,  # times -1 because its away team
        z_target=18.4 / 100 * 2.44,
        y_target=54.3 / 100 * 7.32 - (7.32 / 2),
        team_id=194,
        player_id=184934,
        shot_outcome="goal",
        body_part="head",
        type_of_play="regular_play",
        first_touch=False,
        created_oppertunity="regular_play",
        related_event_id=22,
        ball_goal_distance=np.nan,
        ball_gk_distance=np.nan,
        shot_angle=np.nan,
        gk_angle=np.nan,
        pressure_on_ball=np.nan,
        n_obstructive_players=MISSING_INT,
    ),
    2512690517: ShotEvent(
        event_id=2512690517,
        period_id=1,
        minutes=9,
        seconds=17,
        datetime=pd.to_datetime("2023-01-22T11:18:44.120", utc=True).tz_convert(
            "Europe/Amsterdam"
        ),
        start_x=(9.5 / 100 * 106 - 53)
        * -1,  # standard opta pitch dimensions = [106, 68]
        start_y=(52.8 / 100 * 68 - 34) * -1,  # times -1 because its away team
        z_target=np.nan,
        y_target=np.nan,
        team_id=194,
        player_id=184934,
        shot_outcome="blocked",
        body_part="head",
        type_of_play="regular_play",
        first_touch=False,
        created_oppertunity="regular_play",
        related_event_id=MISSING_INT,
        ball_goal_distance=np.nan,
        ball_gk_distance=np.nan,
        shot_angle=np.nan,
        gk_angle=np.nan,
        pressure_on_ball=np.nan,
        n_obstructive_players=MISSING_INT,
    ),
}

SHOT_EVENTS_OPTA_TRACAB = {}
for key, shot_event in SHOT_EVENTS_OPTA.items():
    SHOT_EVENTS_OPTA_TRACAB[key] = shot_event.copy()
    SHOT_EVENTS_OPTA_TRACAB[key].start_x = (
        SHOT_EVENTS_OPTA_TRACAB[key].start_x / 106 * 100
    )
    SHOT_EVENTS_OPTA_TRACAB[key].start_y = (
        SHOT_EVENTS_OPTA_TRACAB[key].start_y / 68 * 50
    )

DRIBBLE_EVENTS_OPTA = {
    2499594285: DribbleEvent(
        event_id=2499594285,
        period_id=2,
        minutes=30,
        seconds=10,
        datetime=pd.to_datetime("2023-01-22T12:18:43.119").tz_localize(
            "Europe/Amsterdam"
        ),
        start_x=65.7 / 100 * 106 - 53,  # standard opta pitch dimensions = [106, 68]
        start_y=23.2 / 100 * 68 - 34,
        team_id=3,
        player_id=45849,
        related_event_id=4,
        duel_type="offensive",
        outcome=True,
        has_opponent=True,
    )
}

DRIBBLE_EVENTS_OPTA_TRACAB = {}
for key, dribble_event in DRIBBLE_EVENTS_OPTA.items():
    DRIBBLE_EVENTS_OPTA_TRACAB[key] = dribble_event.copy()
    DRIBBLE_EVENTS_OPTA_TRACAB[key].start_x = (
        DRIBBLE_EVENTS_OPTA_TRACAB[key].start_x / 106 * 100
    )
    DRIBBLE_EVENTS_OPTA_TRACAB[key].start_y = (
        DRIBBLE_EVENTS_OPTA_TRACAB[key].start_y / 68 * 50
    )

PASS_EVENTS_OPTA = {
    2499594225: PassEvent(
        event_id=2499594225,
        period_id=1,
        minutes=0,
        seconds=1,
        datetime=pd.to_datetime("2023-01-22T12:18:33.637").tz_localize(
            "Europe/Amsterdam"
        ),
        start_x=(49.7 / 100 * 106 - 53),  # standard opta pitch dimensions = [106, 68]
        start_y=(50.1 / 100 * 68 - 34),
        team_id=3,
        outcome="successful",
        player_id=19367,
        end_x=np.nan,
        end_y=np.nan,
        pass_type="not_specified",
        set_piece="kick_off",
    ),
    2499594243: PassEvent(
        event_id=2499594243,
        period_id=1,
        minutes=0,
        seconds=4,
        datetime=pd.to_datetime("2023-01-22T12:18:36.207").tz_localize(
            "Europe/Amsterdam"
        ),
        start_x=(31.6 / 100 * 106 - 53)
        * -1,  # standard opta pitch dimensions = [106, 68]
        start_y=(40.7 / 100 * 68 - 34) * -1,  # times -1 because its away team
        team_id=194,
        outcome="assist",
        player_id=184934,
        end_x=(70.6 / 100 * 106 - 53) * -1,
        end_y=(57.5 / 100 * 68 - 34) * -1,
        pass_type="long_ball",
        set_piece="no_set_piece",
    ),
}

PASS_EVENTS_OPTA_TRACAB = {}
for key, pass_event in PASS_EVENTS_OPTA.items():
    PASS_EVENTS_OPTA_TRACAB[key] = pass_event.copy()
    PASS_EVENTS_OPTA_TRACAB[key].start_x = (
        PASS_EVENTS_OPTA_TRACAB[key].start_x / 106 * 100
    )
    PASS_EVENTS_OPTA_TRACAB[key].start_y = (
        PASS_EVENTS_OPTA_TRACAB[key].start_y / 68 * 50
    )
    PASS_EVENTS_OPTA_TRACAB[key].end_x = PASS_EVENTS_OPTA_TRACAB[key].end_x / 106 * 100
    PASS_EVENTS_OPTA_TRACAB[key].end_y = PASS_EVENTS_OPTA_TRACAB[key].end_y / 68 * 50

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
    frame_rate=MISSING_INT,
    home_team_id=3,
    home_team_name="TeamOne",
    home_formation="4231",
    home_score=3,
    home_players=pd.DataFrame(
        {
            "id": [19367, 45849],
            "full_name": ["Piet Schrijvers", "Jan Boskamp"],
            "formation_place": [4, 0],
            "position": ["goalkeeper", "midfielder"],
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
            "position": ["midfielder", "goalkeeper"],
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
        "ball_status": ["dead", "alive", "alive", None, "dead", "alive"],
        "ball_possession": [None, None, None, None, None, None],
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
        "event_id": [3, 4, 5, 6, 7, 8],
        "type_id": [5, 1, 10, 2, 2, 2],
        "databallpy_event": [None, "pass", "dribble", "shot", "shot", "shot"],
        "period_id": [1, 1, 2, 2, 2, 2],
        "minutes": [0, 1, 1, 1, 1, 1],
        "seconds": [14.44, 4.22, 15.08, 16.08, 16.08, 16.08],
        "player_id": [3578, 3699, 3568, 3568, 3568, 3568],
        "player_name": [
            "Player 11",
            "Player 34",
            "Player 1",
            "Player 1",
            "Player 1",
            "Player 1",
        ],
        "team_id": ["FIFATMA", "FIFATMB", "FIFATMA", "FIFATMA", "FIFATMA", "FIFATMA"],
        "outcome": [MISSING_INT, 0, 1, 0, 1, 1],
        "start_x": [np.nan, 0.0, 20.0, 20.0, 20.0, 20.0],
        "start_y": [np.nan, -5.0, 5, 5, 5, 5],
        "to_player_id": [
            MISSING_INT,
            3700,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
        ],
        "to_player_name": [None, "Player 35", None, None, None, None],
        "end_x": [np.nan, -20.0, -40.0, -40.0, -40.0, -40.0],
        "end_y": [np.nan, -15.0, 0.0, 0.0, 0.0, 0.0],
        "td_frame": [1, 3, 5, 7, 7, 7],
        "metrica_event": ["set piece", "pass", "carry", "shot", "shot", "shot"],
        "datetime": [
            pd.to_datetime("2019-02-21T03:30:07.000", utc=True),
            pd.to_datetime("2019-02-21T03:30:08.000", utc=True),
            pd.to_datetime("2019-02-21T03:30:09.000", utc=True),
            pd.to_datetime("2019-02-21T03:30:10.000", utc=True),
            pd.to_datetime("2019-02-21T03:30:10.000", utc=True),
            pd.to_datetime("2019-02-21T03:30:10.000", utc=True),
        ],
    }
)

SHOT_EVENTS_METRICA = {
    6: ShotEvent(
        event_id=6,
        period_id=2,
        minutes=1,
        seconds=16.08,
        datetime=pd.to_datetime("2019-02-21T03:30:10.000", utc=True),
        start_x=20.0,
        start_y=5.0,
        team_id="FIFATMA",
        player_id=3568,
        shot_outcome="miss",
        body_part=None,
        type_of_play=None,
        first_touch=None,
        created_oppertunity=None,
        related_event_id=MISSING_INT,
    ),
    7: ShotEvent(
        event_id=7,
        period_id=2,
        minutes=1,
        seconds=16.08,
        datetime=pd.to_datetime("2019-02-21T03:30:10.000", utc=True),
        start_x=20.0,
        start_y=5.0,
        team_id="FIFATMA",
        player_id=3568,
        shot_outcome="goal",
        body_part=None,
        type_of_play=None,
        first_touch=None,
        created_oppertunity=None,
        related_event_id=MISSING_INT,
    ),
    8: ShotEvent(
        event_id=8,
        period_id=2,
        minutes=1,
        seconds=16.08,
        datetime=pd.to_datetime("2019-02-21T03:30:10.000", utc=True),
        start_x=20.0,
        start_y=5.0,
        team_id="FIFATMA",
        player_id=3568,
        shot_outcome="goal",
        body_part=None,
        type_of_play=None,
        first_touch=None,
        created_oppertunity=None,
        related_event_id=MISSING_INT,
    ),
}

DRIBBLE_EVENTS_METRICA = {
    5: DribbleEvent(
        event_id=5,
        period_id=2,
        minutes=1,
        seconds=15.08,
        datetime=pd.to_datetime("2019-02-21T03:30:09.000", utc=True),
        start_x=20.0,
        start_y=5.0,
        team_id="FIFATMA",
        player_id=3568,
        related_event_id=MISSING_INT,
        duel_type=None,
        outcome=True,
        has_opponent=False,
    )
}

PASS_EVENTS_METRICA = {
    4: PassEvent(
        event_id=4,
        period_id=1,
        minutes=1,
        seconds=4.22,
        datetime=pd.to_datetime("2019-02-21T03:30:08.000", utc=True),
        start_x=0.0,
        start_y=-5.0,
        team_id="FIFATMB",
        outcome="unsuccessful",
        player_id=3699,
        end_x=-20.0,
        end_y=-15.0,
        pass_type="not_specified",
        set_piece="unspecified_set_piece",
    )
}


MD_METRICA_TD = Metadata(
    match_id=9999,
    pitch_dimensions=[100.0, 50.0],
    periods_frames=pd.DataFrame(
        {
            "period": [1, 2, 3, 4, 5],
            "start_frame": [1, 4, MISSING_INT, MISSING_INT, MISSING_INT],
            "end_frame": [3, 6, MISSING_INT, MISSING_INT, MISSING_INT],
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
            "position": ["goalkeeper", "right back"],
            "starter": [True, True],
            "shirt_num": [11, 1],
            "start_frame": [MISSING_INT, MISSING_INT],
            "end_frame": [MISSING_INT, MISSING_INT],
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
            "position": ["left forward (2)", "left back"],
            "starter": [True, False],
            "shirt_num": [34, 35],
            "start_frame": [MISSING_INT, MISSING_INT],
            "end_frame": [MISSING_INT, MISSING_INT],
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
            "start_frame": [1, 4, MISSING_INT, MISSING_INT, MISSING_INT],
            "end_frame": [3, 6, MISSING_INT, MISSING_INT, MISSING_INT],
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
            "position": ["goalkeeper", "right back"],
            "starter": [True, True],
            "shirt_num": [11, 1],
            "start_frame": [MISSING_INT, MISSING_INT],
            "end_frame": [MISSING_INT, MISSING_INT],
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
            "position": ["left forward (2)", "left back"],
            "starter": [True, False],
            "shirt_num": [34, 35],
            "start_frame": [MISSING_INT, MISSING_INT],
            "end_frame": [MISSING_INT, MISSING_INT],
        }
    ),
    away_formation="0001",
    away_score=2,
    country="",
)

TD_CHANNELS_METRICA = pd.DataFrame(
    {
        "start": [1, 4],
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
        "ball_status": ["dead", "alive", "alive", None, "alive", "alive"],
        "ball_possession": [None, None, None, None, None, None],
        "home_1_x": [-46.9, -45.9, -44.9, np.nan, 39.0, 39.0],
        "home_1_y": [0.8, -0.2, -1.2, np.nan, 1.5, 2.5],
        "home_2_x": [-19.0, -20.0, -21.0, np.nan, 23.3, 23.3],
        "home_2_y": [-6.0, -5.0, -6.0, np.nan, 6.9, 6.9],
        "away_1_x": [40.0, 39.0, 38.0, np.nan, -45.9, -44.9],
        "away_1_y": [0.5, 1.5, 2.5, np.nan, -0.2, -1.2],
        "away_2_x": [23.3, 23.3, 25.3, np.nan, -20.0, np.nan],
        "away_2_y": [5.9, 6.9, 5.9, np.nan, -5.0, np.nan],
        "period": [MISSING_INT, 1, 1, MISSING_INT, 2, 2],
        "matchtime_td": ["", "00:00", "00:00", "Break", "45:00", "45:00"],
    }
)

MD_INMOTIO = Metadata(
    match_id=9999,
    pitch_dimensions=[100.0, 50.0],
    periods_frames=pd.DataFrame(
        {
            "period": [1, 2, 3, 4, 5],
            "start_frame": [2, 5, MISSING_INT, MISSING_INT, MISSING_INT],
            "end_frame": [3, 6, MISSING_INT, MISSING_INT, MISSING_INT],
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
        "databallpy_event": ["pass", "pass", None, None],
        "period_id": [1, 1, 1, 1],
        "minutes": [0.0, 0.0, 0.0, 0.0],
        "seconds": [5.37, 20.93, 23.64, 28.64],
        "player_id": [2, 1, 3, MISSING_INT],
        "team_id": [1, 1, 2, MISSING_INT],
        "outcome": [0, 1, MISSING_INT, MISSING_INT],
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
        "player_name": ["Player 2", "Player 1", "Player 11", None],
    }
)

RES_SIM_MAT = np.array(
    [
        0.28483747,
        0.28483747,
        0.18797667,
        0.39203142,
        0.27909375,
        0.27909375,
        0.1841453,
        0.39263127,
        0.27326898,
        0.27326898,
        0.18025994,
        0.39301802,
        0.94746644,
        0.94746644,
        0.0109247,
        0.97337888,
        0.25941903,
        0.25941903,
        0.17285291,
        0.3935373,
        0.28465028,
        0.28465028,
        0.19191006,
        0.39543116,
        0.27768306,
        0.27768306,
        0.18824594,
        0.39393699,
        0.27059174,
        0.27059174,
        0.18442605,
        0.37997822,
        0.25739063,
        0.25739063,
        0.17604776,
        0.36994929,
        0.28190777,
        0.28190777,
        0.19498039,
        0.36787944,
        0.2744064,
        0.2744064,
        0.19001546,
        0.38064315,
        0.26706079,
        0.26706079,
        0.18451364,
        0.38545957,
        0.25421203,
        0.25421203,
        0.17503393,
        0.3859927,
    ]
)

RES_SIM_MAT_MISSING_PLAYER = np.array(
    [
        0.01913249,
        0.01913249,
        0.00516593,
        0.37467631,
        0.01794321,
        0.01794321,
        0.00484143,
        0.37304383,
        0.01678971,
        0.01678971,
        0.00452686,
        0.37126927,
        0.84365927,
        0.84365927,
        0.00000066,
        0.91850927,
        0.01425216,
        0.01425216,
        0.00396633,
        0.36787944,
        0.01909291,
        0.01909291,
        0.0055142,
        0.39094392,
        0.01765903,
        0.01765903,
        0.00518928,
        0.38924056,
        0.01627694,
        0.01627694,
        0.00486472,
        0.38738895,
        0.01390403,
        0.01390403,
        0.00420191,
        0.38060291,
        0.01851936,
        0.01851936,
        0.00579694,
        0.40446509,
        0.01701086,
        0.01701086,
        0.00534451,
        0.40270282,
        0.01561715,
        0.01561715,
        0.00487201,
        0.40078718,
        0.01337024,
        0.01337024,
        0.00412614,
        0.38712872,
    ]
)
RES_SIM_MAT_NO_PLAYER = np.array(
    [
        0.04373023,
        0.04373023,
        0.36787944,
        0.09693938,
        0.04156558,
        0.04156558,
        0.37415304,
        0.09730945,
        0.03943723,
        0.03943723,
        0.38057966,
        0.09754851,
        0.87416595,
        0.87416595,
        0.36787944,
        0.93496842,
        0.03464278,
        0.03464278,
        0.39314713,
        0.09787003,
        0.04365864,
        0.04365864,
        0.39346724,
        0.09904802,
        0.04104396,
        0.04104396,
        0.40017719,
        0.09811793,
        0.03848137,
        0.03848137,
        0.40705082,
        0.08968121,
        0.03397166,
        0.03397166,
        0.41487527,
        0.08389799,
        0.04261789,
        0.04261789,
        0.41521307,
        0.08273305,
        0.03984758,
        0.03984758,
        0.41665267,
        0.09007282,
        0.03724212,
        0.03724212,
        0.41256202,
        0.09294004,
        0.03293576,
        0.03293576,
        0.40933317,
        0.09326073,
    ]
)
