import datetime as dt

import numpy as np
import pandas as pd

from databallpy.data_parsers.metadata import Metadata
from databallpy.events import DribbleEvent, PassEvent, ShotEvent, TackleEvent
from databallpy.utils.constants import MISSING_INT

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
        "period_id": [1, 1, MISSING_INT, 2, 2],
        "datetime": pd.to_datetime(
            [
                "2023-01-14 16:46:39.720000",
                "2023-01-14 16:46:39.760000",
                "2023-01-14 16:46:39.800000",
                "2023-01-14 16:46:39.840000",
                "2023-01-14 16:46:39.880000",
            ]
        ).tz_localize("Europe/Amsterdam"),
        "gametime_td": [
            "00:00",
            "00:00",
            "Break",
            "45:00",
            "45:00",
        ],
    }
)


MD_TRACAB = Metadata(
    game_id=1908,
    pitch_dimensions=[100.0, 50.0],
    periods_frames=pd.DataFrame(
        {
            "period_id": [1, 2, 3, 4, 5],
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
    periods_changed_playing_direction=[],
)


def opta_raw_to_scaled(val, new_dim, is_home):
    new_val = (val - 50) / 100 * new_dim
    if is_home:
        return new_val * -1
    return new_val


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
            "tackle",
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
            1,
            1,
            1,
            0,
        ],
        # field dimensions are [105, 50], for opta its standard [100, 100].
        # So all values should be divided by 10 (or 5) and minus 50 (or 25) to get the
        # standard databallpy values.
        "start_x": [
            opta_raw_to_scaled(0.0, 100, True),
            opta_raw_to_scaled(0.0, 100, False),
            opta_raw_to_scaled(0.0, 100, True),
            opta_raw_to_scaled(49.7, 100, False),
            opta_raw_to_scaled(31.6, 100, True),
            opta_raw_to_scaled(31.0, 100, False),
            opta_raw_to_scaled(0.0, 100, True),
            opta_raw_to_scaled(65.7, 100, False),
            opta_raw_to_scaled(34.3, 100, True),
            opta_raw_to_scaled(9.5, 100, False),
            opta_raw_to_scaled(9.5, 100, True),
            opta_raw_to_scaled(9.5, 100, True),
        ],
        "start_y": [
            opta_raw_to_scaled(0.0, 50, True),
            opta_raw_to_scaled(0.0, 50, False),
            opta_raw_to_scaled(0.0, 50, True),
            opta_raw_to_scaled(50.1, 50, False),
            opta_raw_to_scaled(40.7, 50, True),
            opta_raw_to_scaled(44.3, 50, False),
            opta_raw_to_scaled(0.0, 50, True),
            opta_raw_to_scaled(23.2, 50, False),
            opta_raw_to_scaled(76.8, 50, True),
            opta_raw_to_scaled(52.8, 50, False),
            opta_raw_to_scaled(52.8, 50, True),
            opta_raw_to_scaled(52.8, 50, True),
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
        team_side="home",
        pitch_size=[106.0, 68.0],
        player_id=45849,
        jersey=2,
        outcome=False,
        related_event_id=MISSING_INT,
        body_part="head",
        possession_type="corner_kick",
        set_piece="no_set_piece",
        _xt=-1,
        outcome_str="own_goal",
        z_target=18.4 / 100 * 2.44,
        y_target=54.3 / 100 * 7.32 - (7.32 / 2),
        first_touch=False,
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
        pitch_size=[106.0, 68.0],
        team_side="away",
        team_id=194,
        player_id=184934,
        jersey=1,
        outcome=True,
        related_event_id=22,
        body_part="head",
        possession_type="open_play",
        set_piece="no_set_piece",
        _xt=-1,
        outcome_str="goal",
        z_target=18.4 / 100 * 2.44,
        y_target=54.3 / 100 * 7.32 - (7.32 / 2),
        first_touch=False,
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
        pitch_size=[106.0, 68.0],
        team_side="away",
        team_id=194,
        player_id=184934,
        jersey=1,
        outcome=False,
        related_event_id=MISSING_INT,
        body_part="head",
        possession_type="open_play",
        set_piece="no_set_piece",
        _xt=-1,
        outcome_str="blocked",
        z_target=np.nan,
        y_target=np.nan,
        first_touch=False,
    ),
}

SHOT_EVENTS_OPTA_TRACAB = {}
for key, shot_event in SHOT_EVENTS_OPTA.items():
    SHOT_EVENTS_OPTA_TRACAB[key] = shot_event.copy()
    SHOT_EVENTS_OPTA_TRACAB[key].start_x = (
        SHOT_EVENTS_OPTA_TRACAB[key].start_x / 106 * 100
    )
    SHOT_EVENTS_OPTA_TRACAB[key].start_y = SHOT_EVENTS_OPTA_TRACAB[key].start_y / 68 * 50

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
        pitch_size=[106.0, 68.0],
        team_side="home",
        team_id=3,
        player_id=45849,
        jersey=2,
        outcome=False,
        related_event_id=4,
        body_part="foot",
        possession_type="open_play",
        set_piece="no_set_piece",
        _xt=-1,
        duel_type="offensive",
        with_opponent=True,
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

TACKLE_EVENTS_OPTA = {
    2499594291: TackleEvent(
        event_id=2499594291,
        period_id=2,
        minutes=31,
        seconds=10,
        datetime=pd.to_datetime("2023-01-22T12:18:43.120").tz_localize(
            "Europe/Amsterdam"
        ),
        start_x=(34.3 / 100 * 106 - 53)
        * -1,  # standard opta pitch dimensions = [106, 68]
        start_y=(76.8 / 100 * 68 - 34) * -1,
        pitch_size=[106.0, 68.0],
        team_side="away",
        team_id=194,
        player_id=184934,
        jersey=1,
        outcome=True,
        related_event_id=6,
    )
}
TACKLE_EVENTS_OPTA_TRACAB = {}
for key, tackle_event in TACKLE_EVENTS_OPTA.items():
    TACKLE_EVENTS_OPTA_TRACAB[key] = tackle_event.copy()
    TACKLE_EVENTS_OPTA_TRACAB[key].start_x = (
        TACKLE_EVENTS_OPTA_TRACAB[key].start_x / 106 * 100
    )
    TACKLE_EVENTS_OPTA_TRACAB[key].start_y = (
        TACKLE_EVENTS_OPTA_TRACAB[key].start_y / 68 * 50
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
        pitch_size=[106.0, 68.0],
        team_side="home",
        team_id=3,
        player_id=19367,
        jersey=1,
        outcome=True,
        related_event_id=MISSING_INT,
        body_part="unspecified",
        possession_type="open_play",
        set_piece="kick_off",
        _xt=-1,
        outcome_str="successful",
        end_x=np.nan,
        end_y=np.nan,
        pass_type="unspecified",
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
        pitch_size=[106.0, 68.0],
        team_side="away",
        player_id=184934,
        jersey=1,
        team_id=194,
        outcome=False,
        related_event_id=MISSING_INT,
        body_part="unspecified",
        possession_type="open_play",
        set_piece="no_set_piece",
        _xt=-1,
        outcome_str="assist",
        end_x=(70.6 / 100 * 106 - 53) * -1,
        end_y=(57.5 / 100 * 68 - 34) * -1,
        pass_type="long_ball",
    ),
}

PASS_EVENTS_OPTA_TRACAB = {}
for key, pass_event in PASS_EVENTS_OPTA.items():
    PASS_EVENTS_OPTA_TRACAB[key] = pass_event.copy()
    PASS_EVENTS_OPTA_TRACAB[key].start_x = (
        PASS_EVENTS_OPTA_TRACAB[key].start_x / 106 * 100
    )
    PASS_EVENTS_OPTA_TRACAB[key].start_y = PASS_EVENTS_OPTA_TRACAB[key].start_y / 68 * 50
    PASS_EVENTS_OPTA_TRACAB[key].end_x = PASS_EVENTS_OPTA_TRACAB[key].end_x / 106 * 100
    PASS_EVENTS_OPTA_TRACAB[key].end_y = PASS_EVENTS_OPTA_TRACAB[key].end_y / 68 * 50

MD_OPTA = Metadata(
    game_id=1908,
    pitch_dimensions=[100.0, 50.0],
    periods_frames=pd.DataFrame(
        {
            "period_id": [1, 2, 3, 4, 5],
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
        "datetime": pd.to_datetime(
            [
                "2019-02-21T03:30:07.000",
                "2019-02-21T03:30:07.500",
                "2019-02-21T03:30:08.000",
                "2019-02-21T03:30:08.500",
                "2019-02-21T03:30:09.000",
                "2019-02-21T03:30:09.500",
            ],
            utc=True,
        ),
        "period_id": [1, 1, 1, 2, 2, 2],
        "gametime_td": ["00:00", "00:00", "00:01", "45:00", "45:00", "45:01"],
    }
)

ED_METRICA = pd.DataFrame(
    {
        "event_id": [3, 4, 5, 6, 7, 8, 9, 9],
        "type_id": [
            5,
            1,
            10,
            2,
            2,
            2,
            9,
            9,
        ],
        "databallpy_event": [
            None,
            "pass",
            "dribble",
            "shot",
            "shot",
            "shot",
            "tackle",
            None,
        ],
        "period_id": [1, 1, 2, 2, 2, 2, 2, 2],
        "minutes": [0, 1, 1, 1, 1, 1, 1, 1],
        "seconds": [14.44, 4.22, 15.08, 16.08, 16.08, 16.08, 20.00, 20.00],
        "player_id": [3578, 3699, 3568, 3568, 3568, 3568, 3568, 3568],
        "player_name": [
            "Player 11",
            "Player 34",
            "Player 1",
            "Player 1",
            "Player 1",
            "Player 1",
            "Player 1",
            "Player 1",
        ],
        "team_id": [
            "FIFATMA",
            "FIFATMB",
            "FIFATMA",
            "FIFATMA",
            "FIFATMA",
            "FIFATMA",
            "FIFATMA",
            "FIFATMA",
        ],
        "outcome": [MISSING_INT, 0, 1, 0, 1, 1, 0, MISSING_INT],
        "start_x": [np.nan, 0.0, 20.0, 20.0, 20.0, 20.0, -20.0, -20.0],
        "start_y": [np.nan, -5.0, 5, 5, 5, 5, -20.0, -20.0],
        "to_player_id": [
            MISSING_INT,
            3700,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
        ],
        "to_player_name": [None, "Player 35", None, None, None, None, None, None],
        "end_x": [np.nan, -20.0, -40.0, -40.0, -40.0, -40.0, np.nan, np.nan],
        "end_y": [np.nan, -15.0, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan],
        "td_frame": [1, 3, 5, 7, 7, 7, 10, 10],
        "metrica_event": [
            "set piece",
            "pass",
            "carry",
            "shot",
            "shot",
            "shot",
            "tackle",
            "challenge",
        ],
        "datetime": [
            pd.to_datetime("2019-02-21T03:30:07.000", utc=True),
            pd.to_datetime("2019-02-21T03:30:08.000", utc=True),
            pd.to_datetime("2019-02-21T03:30:09.000", utc=True),
            pd.to_datetime("2019-02-21T03:30:10.000", utc=True),
            pd.to_datetime("2019-02-21T03:30:10.000", utc=True),
            pd.to_datetime("2019-02-21T03:30:10.000", utc=True),
            pd.to_datetime("2019-02-21T03:30:11.500", utc=True),
            pd.to_datetime("2019-02-21T03:30:11.500", utc=True),
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
        team_side="home",
        pitch_size=[100.0, 50.0],
        player_id=3568,
        jersey=1,
        outcome=False,
        related_event_id=MISSING_INT,
        _xt=-1.0,
        body_part="unspecified",
        possession_type="unspecified",
        set_piece="unspecified",
        outcome_str="miss",
        first_touch=False,
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
        team_side="home",
        pitch_size=[100.0, 50.0],
        player_id=3568,
        jersey=1,
        outcome=True,
        related_event_id=MISSING_INT,
        _xt=-1.0,
        body_part="unspecified",
        possession_type="unspecified",
        set_piece="unspecified",
        outcome_str="goal",
        first_touch=False,
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
        team_side="home",
        pitch_size=[100.0, 50.0],
        player_id=3568,
        jersey=1,
        outcome=True,
        related_event_id=MISSING_INT,
        _xt=-1.0,
        body_part="unspecified",
        possession_type="unspecified",
        set_piece="unspecified",
        outcome_str="goal",
        first_touch=False,
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
        team_side="home",
        pitch_size=[100.0, 50.0],
        player_id=3568,
        jersey=1,
        outcome=True,
        body_part="unspecified",
        possession_type="unspecified",
        set_piece="unspecified",
        related_event_id=MISSING_INT,
        _xt=-1.0,
        duel_type="unspecified",
        with_opponent=False,
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
        team_side="away",
        pitch_size=[100.0, 50.0],
        player_id=3699,
        jersey=34,
        outcome=False,
        related_event_id=MISSING_INT,
        body_part="unspecified",
        possession_type="unspecified",
        set_piece="unspecified",
        _xt=-1.0,
        outcome_str="unsuccessful",
        end_x=-20.0,
        end_y=-15.0,
        pass_type="unspecified",
    )
}

TACKLE_EVENTS_METRICA = {
    9: TackleEvent(
        event_id=9,
        period_id=2,
        minutes=1,
        seconds=20.0,
        datetime=pd.to_datetime("2019-02-21T03:30:11.500", utc=True),
        start_x=-20.0,
        start_y=-20.0,
        team_id="FIFATMA",
        team_side="home",
        pitch_size=[100.0, 50.0],
        player_id=3568,
        jersey=1,
        outcome=False,
        related_event_id=MISSING_INT,
    )
}


MD_METRICA_TD = Metadata(
    game_id=9999,
    pitch_dimensions=[100.0, 50.0],
    periods_frames=pd.DataFrame(
        {
            "period_id": [1, 2, 3, 4, 5],
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
            "position": ["goalkeeper", "defender"],
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
            "position": ["forward", "defender"],
            "starter": [True, False],
            "shirt_num": [34, 35],
            "start_frame": [MISSING_INT, MISSING_INT],
            "end_frame": [MISSING_INT, MISSING_INT],
        }
    ),
    away_formation="0001",
    away_score=2,
    country="",
    periods_changed_playing_direction=[],
)


MD_METRICA_ED = Metadata(
    game_id=9999,
    pitch_dimensions=[100.0, 50.0],
    periods_frames=pd.DataFrame(
        {
            "period_id": [1, 2, 3, 4, 5],
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
            "position": ["goalkeeper", "defender"],
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
            "position": ["forward", "defender"],
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
        "datetime": pd.to_datetime(
            [
                "2023-01-01T20:00:00.000",
                "2023-01-01T20:00:00.000",
                "2023-01-01T20:00:00.040",
                "2023-01-01T20:00:00.080",
                "2023-01-01T20:00:00.120",
                "2023-01-01T20:00:00.160",
            ]
        ).tz_localize("Europe/Amsterdam"),
        "period_id": [MISSING_INT, 1, 1, MISSING_INT, 2, 2],
        "gametime_td": ["", "00:00", "00:00", "Break", "45:00", "45:00"],
    }
)

MD_INMOTIO = Metadata(
    game_id=9999,
    pitch_dimensions=[100.0, 50.0],
    periods_frames=pd.DataFrame(
        {
            "period_id": [1, 2, 3, 4, 5],
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
    periods_changed_playing_direction=[2],
)

MD_INSTAT = Metadata(
    game_id=9999,
    pitch_dimensions=[np.nan, np.nan],
    periods_frames=pd.DataFrame(
        {
            "period_id": [1, 2, 3, 4, 5],
            "start_datetime_ed": [
                pd.to_datetime("2023-01-01 20:00:00").tz_localize("Europe/Amsterdam"),
                pd.to_datetime("2023-01-01 21:00:00").tz_localize("Europe/Amsterdam"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
                pd.to_datetime("NaT"),
            ],
            "end_datetime_ed": [
                pd.to_datetime("2023-01-01 20:45:00").tz_localize("Europe/Amsterdam"),
                pd.to_datetime("2023-01-01 21:45:00").tz_localize("Europe/Amsterdam"),
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
            "position": ["goalkeeper", "defender"],
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
            "position": ["goalkeeper", ""],
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
        [0.57920396, 0.57920396, 0.33008111, 0.56176306],
        [0.57984666, 0.57984666, 0.33020744, 0.56314942],
        [0.25, 0.25, 0.3303289, 0.39491859],
        [0.63486825, 0.63486825, 0.99133706, 0.60193588],
        [0.24986165, 0.24986165, 0.33055797, 0.39506437],
        [0.20523789, 0.20523789, 0.33066592, 0.45069861],
        [0.37932751, 0.37932751, 0.33076971, 0.4862895],
        [0.39330079, 0.39330079, 0.33086949, 0.49367201],
        [0.21890386, 0.21890386, 0.33096541, 0.48964095],
        [0.20638475, 0.20638475, 0.33105762, 0.37269406],
        [0.38252189, 0.38252189, 0.33105762, 0.33382869],
        [0.59188298, 0.59188298, 0.33096541, 0.51924347],
        [0.39716059, 0.39716059, 0.33086949, 0.35968392],
    ]
)

RES_SIM_MAT_MISSING_PLAYER = np.array(
    [
        [0.57920396, 0.57920396, 0.33008111, 0.74177792],
        [0.57984666, 0.57984666, 0.33020744, 0.7418784],
        [0.25, 0.25, 0.3303289, 0.49364823],
        [0.63486825, 0.63486825, 0.99133706, 0.65832152],
        [0.24986165, 0.24986165, 0.33055797, 0.49383046],
        [0.20523789, 0.20523789, 0.33066592, 0.49391635],
        [0.37932751, 0.37932751, 0.33076971, 0.49399892],
        [0.39330079, 0.39330079, 0.33086949, 0.49407831],
        [0.21890386, 0.21890386, 0.33096541, 0.49415463],
        [0.20638475, 0.20638475, 0.33105762, 0.494228],
        [0.38252189, 0.38252189, 0.33105762, 0.49429854],
        [0.59188298, 0.59188298, 0.33096541, 0.74269314],
        [0.39716059, 0.39716059, 0.33086949, 0.49429854],
    ]
)

RES_SIM_MAT_NO_PLAYER = np.array(
    [
        [0.57920396, 0.57920396, 0.33008111, 0.56176306],
        [0.57984666, 0.57984666, 0.33020744, 0.56314942],
        [0.25, 0.25, 0.3303289, 0.39491859],
        [0.63486825, 0.63486825, 0.99133706, 0.60193588],
        [0.24986165, 0.24986165, 0.33055797, 0.39506437],
        [0.20523789, 0.20523789, 0.33066592, 0.45069861],
        [0.37932751, 0.37932751, 0.33076971, 0.4862895],
        [0.39330079, 0.39330079, 0.33086949, 0.49367201],
        [0.21890386, 0.21890386, 0.33096541, 0.48964095],
        [0.20638475, 0.20638475, 0.33105762, 0.37269406],
        [0.38252189, 0.38252189, 0.33105762, 0.33382869],
        [0.59188298, 0.59188298, 0.33096541, 0.51924347],
        [0.39716059, 0.39716059, 0.33086949, 0.35968392],
    ]
)

MD_SCISPORTS = Metadata(
    game_id="some_id",
    pitch_dimensions=(106.0, 68.0),
    periods_frames=pd.DataFrame(
        {
            "period_id": [1, 2, 3, 4, 5],
            "start_datetime_ed": pd.to_datetime(
                [
                    "2023-01-01 00:00:00.7",
                    "2023-01-01 00:0:12.0",
                    "NaT",
                    "NaT",
                    "NaT",
                ]
            ).tz_localize("Europe/Amsterdam"),
            "end_datetime_ed": pd.to_datetime(
                [
                    "2023-01-01 00:0:09.3",
                    "2023-01-01 00:0:15.0",
                    "NaT",
                    "NaT",
                    "NaT",
                ]
            ).tz_localize("Europe/Amsterdam"),
        }
    ),
    frame_rate=MISSING_INT,
    home_team_id=100,
    home_team_name="Team 1",
    home_players=pd.DataFrame(
        {
            "id": [101, 102],
            "full_name": ["home player 1", "home player 2"],
            "formation_place": [3, 4],
            "position": ["defender", "defender"],
            "starter": [True, True],
            "shirt_num": [22, 4],
        }
    ),
    home_score=0,
    home_formation=None,
    away_team_id=200,
    away_team_name="Team 2",
    away_players=pd.DataFrame(
        {
            "id": [201, 203],
            "full_name": ["away player 1", "away player 2"],
            "formation_place": [7, 9],
            "position": ["defender", "forward"],
            "starter": [True, False],
            "shirt_num": [4, 17],
        }
    ),
    away_score=1,
    away_formation=None,
    country="Netherlands",
    periods_changed_playing_direction=None,
)

ED_SCISPORTS = pd.DataFrame(
    {
        "event_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "databallpy_event": [
            None,
            None,
            None,
            None,
            None,
            "pass",
            "dribble",
            "pass",
            None,
            "pass",
            "shot",
            "shot",
            None,
            "tackle",
            "tackle",
        ],
        "period_id": [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        "minutes": [0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 45, 45, 45, 45, 45],
        "seconds": [0, 0, 0, 0, 0, 0.7, 3.4, 6.6, 9.3, 0.0, 0.1, 0.7, 1.5, 2.5, 3.0],
        "player_id": [
            MISSING_INT,
            MISSING_INT,
            101,
            102,
            201,
            101,
            102,
            102,
            MISSING_INT,
            203,
            201,
            201,
            MISSING_INT,
            101,
            101,
        ],
        "team_id": [
            100,
            200,
            100,
            100,
            200,
            100,
            100,
            100,
            200,
            200,
            200,
            200,
            200,
            100,
            100,
        ],
        "outcome": [
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            MISSING_INT,
            1,
            1,
            0,
            MISSING_INT,
            1,
            0,
            1,
            MISSING_INT,
            1,
            0,
        ],
        "start_x": [
            0.0,
            -0.0,
            0.0,
            0.0,
            -0.0,
            0.01,
            -17.68,
            24.01,
            0.0,
            4.03,
            -37.7,
            -42.5,
            -0.0,
            12.125,
            -6.3,
        ],
        "start_y": [
            0.0,
            -0.0,
            0.0,
            0.0,
            -0.0,
            -0.6,
            2.01,
            25.97,
            0.0,
            -1.9,
            32.5,
            -0.0,
            -0.0,
            28.83,
            14.28,
        ],
        "datetime": pd.to_datetime(
            ["2023-01-01 00:00:00.0"] * 5
            + [
                "2023-01-01 00:00:00.7",
                "2023-01-01 00:00:03.4",
                "2023-01-01 00:00:06.6",
                "2023-01-01 00:00:09.3",
                "2023-01-01 00:00:12.0",
                "2023-01-01 00:00:12.1",
                "2023-01-01 00:00:12.7",
                "2023-01-01 00:00:13.5",
                "2023-01-01 00:00:14.5",
                "2023-01-01 00:00:15.0",
            ]
        ).tz_localize("Europe/Amsterdam"),
        "scisports_event": ["formation"] * 2
        + ["position"] * 3
        + [
            "pass",
            "dribble",
            "cross",
            "period",
            "pass",
            "shot",
            "shot",
            "period",
            "defensive_duel",
            "foul",
        ],
        "player_name": [
            "not_applicable",
            "not_applicable",
            "home player 1",
            "home player 2",
            "away player 1",
            "home player 1",
            "home player 2",
            "home player 2",
            "not_applicable",
            "away player 2",
            "away player 1",
            "away player 1",
            "not_applicable",
            "home player 1",
            "home player 1",
        ],
        "team_name": [
            "Team 1",
            "Team 2",
            "Team 1",
            "Team 1",
            "Team 2",
            "Team 1",
            "Team 1",
            "Team 1",
            "Team 2",
            "Team 2",
            "Team 2",
            "Team 2",
            "Team 2",
            "Team 1",
            "Team 1",
        ],
    }
)


SPORTEC_METADATA_TD = Metadata(
    game_id="Match-1",
    pitch_dimensions=[105.0, 68.0],
    periods_frames=pd.DataFrame(
        {
            "period_id": [1, 2, 3, 4, 5],
            "start_frame": [10000, 100000] + [MISSING_INT] * 3,
            "end_frame": [10002, 100002] + [MISSING_INT] * 3,
            "start_datetime_td": pd.to_datetime(
                ["2022-11-11 17:31:12.360000+0000", "2022-11-11 18:33:12.360000+0000"]
                + ["NaT"] * 3,
                utc=True,
            ),
            "end_datetime_td": pd.to_datetime(
                ["2022-11-11 17:31:14.360000+0000", "2022-11-11 18:33:14.360000+0000"]
                + ["NaT"] * 3,
                utc=True,
            ),
        }
    ),
    home_team_id="Team1",
    home_team_name="TeamA",
    home_players=pd.DataFrame(
        {
            "id": ["A-1", "A-3", "A-5"],
            "full_name": ["Adam Bodzek", "Rouwen Hennings", "Raphael Wolf"],
            "shirt_num": [13, 28, 1],
            "position": ["", "forward", ""],
            "start_frame": [MISSING_INT] * 3,
            "end_frame": [MISSING_INT] * 3,
            "starter": [False, True, False],
        }
    ),
    home_score=1,
    home_formation="442",
    away_team_id="Team2",
    away_team_name="TeamB",
    away_players=pd.DataFrame(
        {
            "id": ["B-1", "B-2", "B-3"],
            "full_name": ["Mike Wunderlich", "Andreas Luthe", "Kevin Kraus"],
            "shirt_num": [28, 1, 5],
            "position": ["midfielder", "goalkeeper", "defender"],
            "start_frame": [MISSING_INT] * 3,
            "end_frame": [MISSING_INT] * 3,
            "starter": [True, True, False],
        }
    ),
    away_score=2,
    away_formation="4231",
    country="Germany",
    frame_rate=1,
    periods_changed_playing_direction=[1],
)
SPORTEC_METADATA_TD.periods_frames["start_datetime_td"] = pd.to_datetime(
    SPORTEC_METADATA_TD.periods_frames["start_datetime_td"]
).dt.tz_convert("Europe/Berlin")
SPORTEC_METADATA_TD.periods_frames["end_datetime_td"] = pd.to_datetime(
    SPORTEC_METADATA_TD.periods_frames["end_datetime_td"]
).dt.tz_convert("Europe/Berlin")


SPORTEC_METADATA_ED = SPORTEC_METADATA_TD.copy()
SPORTEC_METADATA_ED.periods_frames = pd.DataFrame(
    {
        "period_id": [1, 2, 3, 4, 5],
        "start_datetime_ed": pd.to_datetime(
            ["2022-11-11T18:31:12.000+01:00", "2022-11-11T19:31:09.000+01:00"]
            + ["NaT"] * 3,
            utc=True,
        ),
        "end_datetime_ed": pd.to_datetime(
            ["2022-11-11T18:37:36.200+01:00", "2022-11-11T20:10:18.500+01:00"]
            + ["NaT"] * 3,
            utc=True,
        ),
    }
)
SPORTEC_METADATA_ED.periods_frames["start_datetime_ed"] = pd.to_datetime(
    SPORTEC_METADATA_ED.periods_frames["start_datetime_ed"]
).dt.tz_convert("Europe/Berlin")
SPORTEC_METADATA_ED.periods_frames["end_datetime_ed"] = pd.to_datetime(
    SPORTEC_METADATA_ED.periods_frames["end_datetime_ed"]
).dt.tz_convert("Europe/Berlin")
SPORTEC_METADATA_ED.frame_rate = MISSING_INT

SPORTEC_EVENT_DATA = pd.DataFrame(
    {
        "event_id": [12, 13, 14, 15, 17, 18],
        "databallpy_event": ["pass", "shot", "pass", None, "dribble", None],
        "period_id": [1, 1, 2, 2, 2, 2],
        "minutes": [0, 6, 45, 72, 83, 84],
        "seconds": [0, 24.2, 0, 26.5, 19.6, 9.5],
        "player_id": ["B-1", "B-3", "A-1", "A-3", "A-5", "B-2"],
        "team_id": ["Team2", "Team2", "Team1", "Team1", "Team1", "Team2"],
        "outcome": [True, False, True, None, True, None],
        "start_x": np.array([52.5, 98.41, 52.5, 63.44, 15.19, 44.28]) - 52.5,
        "start_y": np.array([34.00, 36.55, 34.00, 40.54, 4.39, 12.24]) - 34.0,
        "datetime": [
            "2022-11-11T18:31:12.000+01:00",
            "2022-11-11T18:37:36.200+01:00",
            "2022-11-11T19:31:09.000+01:00",
            "2022-11-11T19:58:35.500+01:00",
            "2022-11-11T20:09:28.600+01:00",
            "2022-11-11T20:10:18.500+01:00",
        ],
        "sportec_event": [
            "Pass",
            "SavedShot",
            "Pass",
            "ballcontactSucceeded",
            "dribbledAround",
            "OtherBallAction",
        ],
        "player_name": [
            "Mike Wunderlich",
            "Kevin Kraus",
            "Adam Bodzek",
            "Rouwen Hennings",
            "Raphael Wolf",
            "Andreas Luthe",
        ],
    }
)
SPORTEC_EVENT_DATA["datetime"] = pd.to_datetime(
    SPORTEC_EVENT_DATA["datetime"]
).dt.tz_convert("Europe/Berlin")
SPORTEC_EVENT_DATA.loc[
    SPORTEC_EVENT_DATA["period_id"] == 1, ["start_x", "start_y"]
] *= -1

SPORTEC_DATABALLPY_EVENTS = {
    "shot_events": {
        13: ShotEvent(
            event_id=13,
            period_id=1,
            minutes=6,
            seconds=24.2,
            datetime=pd.to_datetime("2022-11-11T18:37:36.200+01:00").tz_convert(
                "Europe/Berlin"
            ),
            start_x=-45.91,
            start_y=-2.55,
            team_id="Team2",
            pitch_size=[105.0, 68.0],
            player_id="B-3",
            jersey=5,
            outcome=False,
            related_event_id=None,
            body_part="head",
            possession_type="free_kick",
            set_piece="no_set_piece",
            _xt=-1,
            outcome_str="miss_on_target",
            team_side="away",
        )
    },
    "pass_events": {
        12: PassEvent(
            event_id=12,
            period_id=1,
            minutes=0,
            seconds=0,
            datetime=pd.to_datetime("2022-11-11T18:31:12.000+01:00").tz_convert(
                "Europe/Berlin"
            ),
            start_x=0.0,
            start_y=0.0,
            team_id="Team2",
            team_side="away",
            pitch_size=[
                105.0,
                68.0,
            ],
            player_id="B-1",
            jersey=28,
            related_event_id=None,
            body_part="unspecified",
            possession_type="unspecified",
            set_piece="kick_off",
            _xt=-1,
            outcome=True,
            outcome_str="unspecified",
            end_x=np.nan,
            end_y=np.nan,
            pass_type="unspecified",
            receiver_player_id="B-3",
        ),
        14: PassEvent(
            event_id=14,
            period_id=2,
            minutes=45,
            seconds=0,
            datetime=pd.to_datetime("2022-11-11T19:31:09.000+01:00").tz_convert(
                "Europe/Berlin"
            ),
            start_x=0.0,
            start_y=0.0,
            team_id="Team1",
            team_side="home",
            pitch_size=[
                105.0,
                68.0,
            ],
            player_id="A-1",
            jersey=13,
            related_event_id=None,
            body_part="unspecified",
            possession_type="unspecified",
            set_piece="kick_off",
            _xt=-1,
            outcome=True,
            outcome_str="unspecified",
            end_x=np.nan,
            end_y=np.nan,
            pass_type="unspecified",
            receiver_player_id="A-5",
        ),
    },
    "dribble_events": {
        17: DribbleEvent(
            event_id=17,
            period_id=2,
            minutes=83,
            seconds=19.6,
            datetime=pd.to_datetime("2022-11-11T20:09:28.600+01:00").tz_convert(
                "Europe/Berlin"
            ),
            start_x=15.19 - 52.5,
            start_y=4.39 - 34.0,
            team_id="Team1",
            team_side="home",
            pitch_size=[
                105.0,
                68.0,
            ],
            player_id="A-5",
            jersey=1,
            outcome=True,
            related_event_id=None,
            body_part="foot",
            possession_type="open_play",
            set_piece="no_set_piece",
            _xt=-1,
            duel_type="unspecified",
            with_opponent=True,
        )
    },
    "other_events": {},
}

TRACAB_SPORTEC_XML_TD = pd.DataFrame(
    {
        "frame": [10000, 10001, 10002, 100000, 100001, 100002],
        "ball_x": [0.8, 1.8, 2.8, -0.8, 1.8, -13.82],
        "ball_y": [-0.1, -0.3, 2.1, 1.0, 1.3, 2.5],
        "ball_z": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        "ball_status": ["alive", "alive", "dead", "alive", "dead", "dead"],
        "ball_possession": ["home", "away", "home", "away", "away", "home"],
        "datetime": [
            "2022-11-11 17:31:12.360000+0000",
            "2022-11-11 17:31:13.360000+0000",
            "2022-11-11 17:31:14.360000+0000",
            "2022-11-11 18:33:12.360000+0000",
            "2022-11-11 18:33:13.360000+0000",
            "2022-11-11 18:33:14.360000+0000",
        ],
        "home_13_x": [-5.1, -5.2, -6.3, -5.4, -12.2, -8.3],
        "home_13_y": [8.2, -3.3, 8.1, -8.8, -22.2, -0.8],
        "away_5_x": [5.4, 15.2, 15.1, np.nan, np.nan, np.nan],
        "away_5_y": [-12.1, -18.1, -20.3, np.nan, np.nan, np.nan],
        "away_1_x": [np.nan, np.nan, np.nan, 52.0, 50.1, 33.2],
        "period_id": [1, 1, 1, 2, 2, 2],
        "away_1_y": [np.nan, np.nan, np.nan, -8.1, -4.8, -18.8],
        "gametime_td": ["00:00", "00:01", "00:02", "45:00", "45:01", "45:02"],
    }
)
TRACAB_SPORTEC_XML_TD["datetime"] = pd.to_datetime(
    TRACAB_SPORTEC_XML_TD["datetime"]
).dt.tz_convert("Europe/Berlin")

MD_STATSBOMB = Metadata(
    game_id=15946,
    pitch_dimensions=(105.0, 68.0),
    periods_frames=pd.DataFrame(
        {
            "period_id": [1, 2, 3, 4, 5],
            "start_datetime_ed": [
                pd.to_datetime("2018-08-18 22:15:00+00:00"),
                pd.to_datetime("2018-08-18 23:15:00+00:00"),
                pd.NaT,
                pd.NaT,
                pd.NaT,
            ],
            "end_datetime_ed": [
                pd.to_datetime("2018-08-18 23:00:00+00:00"),
                pd.to_datetime("2018-08-19 00:00:00+00:00"),
                pd.NaT,
                pd.NaT,
                pd.NaT,
            ],
        }
    ),
    frame_rate=MISSING_INT,
    home_team_id=217,
    home_team_name="Barcelona",
    home_players=pd.DataFrame(
        {
            "id": [5211, 8206, 8652],
            "full_name": [
                "Jordi Alba Ramos",
                "Arturo Erasmo Vidal Pardo",
                "Jasper Cillessen",
            ],
            "formation_place": [6, 10, MISSING_INT],
            "position": ["defender", "midfielder", "unspecified"],
            "starter": [True, False, False],
            "shirt_num": [18, 22, 13],
        }
    ),
    home_score=3,
    home_formation="442",
    away_team_id=206,
    away_team_name="Deportivo Alavés",
    away_players=pd.DataFrame(
        {
            "id": [6566, 6581, 6923],
            "full_name": [
                "Borja González Tomás",
                "Jonathan Rodríguez Menéndez",
                "Joaquín Navarro Jiménez",
            ],
            "formation_place": [16, 16, MISSING_INT],
            "position": ["midfielder", "midfielder", "unspecified"],
            "starter": [False, True, False],
            "shirt_num": [18, 23, 15],
        }
    ),
    away_score=0,
    away_formation="451",
    country="Spain",
)

ED_STATSBOMB = pd.DataFrame(
    {
        "event_id": [0, 1, 2, 3, 4],
        "databallpy_event": ["pass", None, "shot", "dribble", "pass"],
        "period_id": [1, 1, 1, 1, 1],
        "minutes": [0, 0, 5, 7, 0],
        "seconds": [11, 45, 39, 7, 11],
        "player_id": [5211, 5211, 5211, 6581, 5211],
        "team_id": [217, 217, 217, 206, 206],
        "outcome": [False, None, False, False, False],
        "start_x": [-30.0125, -4.7250, 47.1625, -27.4750, 30.0125],
        "start_y": [32.64, 30.260, 11.560, -13.515, -32.64],
        "datetime": [
            pd.to_datetime("2018-08-18 22:15:11+00:00"),
            pd.to_datetime("2018-08-18 22:15:45+00:00"),
            pd.to_datetime("2018-08-18 22:20:39+00:00"),
            pd.to_datetime("2018-08-18 22:22:07+00:00"),
            pd.to_datetime("2018-08-18 22:15:11+00:00"),
        ],
        "statsbomb_event": ["pass", "carry", "shot", "dribble", "pass"],
        "statsbomb_event_id": [
            "c723053c-b956-494b-bb5d-352e1833203a",
            "4f2502b5-2014-4265-afa1-f011aa4fd32e",
            "9107d374-2942-4876-a14f-1b9f86901c15",
            "d9cbb43c-e1a4-45d1-a4b9-2151657bb62a",
            "c723053c-b956-494b-bb5d-352e1833203a",
        ],
        "player_name": [
            "Jordi Alba Ramos",
            "Jordi Alba Ramos",
            "Jordi Alba Ramos",
            "Jonathan Rodríguez Menéndez",
            "Jordi Alba Ramos",
        ],
        "team_name": [
            "Barcelona",
            "Barcelona",
            "Barcelona",
            "Deportivo Alavés",
            "Deportivo Alavés",
        ],
        "statsbomb_outcome": ["Incomplete", "", "Off T", "Incomplete", "Incomplete"],
    }
)

SHOT_EVENT_STATSBOMB = {
    2: ShotEvent(
        event_id=2,
        period_id=1,
        minutes=5,
        seconds=39,
        datetime=pd.to_datetime("2018-08-18 22:20:39+0000", utc=True),
        start_x=47.1625,
        start_y=11.560,
        team_id=217,
        team_side="home",
        pitch_size=[105.0, 68.0],
        player_id=5211,
        jersey=18,
        outcome=False,
        related_event_id=["7fb36c67-4b6c-4c3d-bc52-4e1cd712e790"],
        body_part="left_foot",
        possession_type="open_play",
        set_piece="no_set_piece",
        _xt=0.07851,
        outcome_str="miss_off_target",
    )
}

PASS_EVENT_STATSBOMB = {
    0: PassEvent(
        event_id=0,
        period_id=1,
        minutes=0,
        seconds=11,
        datetime=pd.to_datetime("2018-08-18 22:15:11+0000", utc=True),
        start_x=-30.0125,
        start_y=32.64,
        team_id=217,
        team_side="home",
        pitch_size=[105.0, 68.0],
        player_id=5211,
        jersey=18,
        outcome=False,
        related_event_id=["994ee6b7-0e8b-4168-ac79-81c68ee580f7"],
        body_part="left_foot",
        possession_type="corner_kick",
        set_piece="corner_kick",
        _xt=0.049,
        outcome_str="unspecified",
        end_x=26.3375,
        end_y=24.225,
        pass_type="through_ball",
        receiver_player_id=5246,
    ),
    4: PassEvent(
        event_id=4,
        period_id=1,
        minutes=0,
        seconds=11,
        datetime=pd.to_datetime("2018-08-18 22:15:11+0000", utc=True),
        start_x=30.0125,
        start_y=-32.64,
        team_id=206,
        team_side="away",
        pitch_size=[105.0, 68.0],
        player_id=5211,
        jersey=18,
        outcome=False,
        related_event_id=["994ee6b7-0e8b-4168-ac79-81c68ee580f7"],
        body_part="left_foot",
        possession_type="corner_kick",
        set_piece="corner_kick",
        _xt=0.049,
        outcome_str="unspecified",
        end_x=-26.3375,
        end_y=-24.225,
        pass_type="through_ball",
        receiver_player_id=5246,
    ),
}

DRIBBLE_EVENT_STATSBOMB = {
    3: DribbleEvent(
        event_id=3,
        period_id=1,
        minutes=7,
        seconds=7,
        datetime=pd.to_datetime("2018-08-18 22:22:07+0000", utc=True),
        start_x=-27.4750,
        start_y=-13.515,
        team_id=206,
        team_side="away",
        pitch_size=[105.0, 68.0],
        player_id=6581,
        jersey=23,
        outcome=False,
        related_event_id=["4c601a79-df4f-4001-902b-6fc9788718eb"],
        body_part="unspecified",
        possession_type="unspecified",
        set_piece="unspecified",
        _xt=0.001879,
        duel_type="unspecified",
        with_opponent=None,
    )
}
