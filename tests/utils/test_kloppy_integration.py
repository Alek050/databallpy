import datetime as dt
import unittest
import warnings

from kloppy import signality, sportec, statsbomb, statsperform, tracab
from kloppy.domain import (
    BodyPart,
    BodyPartQualifier,
    CardQualifier,
    CardType,
    DuelQualifier,
    DuelResult,
    DuelType,
    EventFactory,
    EventType,
    GoalkeeperActionType,
    GoalkeeperQualifier,
    InterceptionResult,
    PassQualifier,
    PassResult,
    PassType,
    Period,
    SecondSpectrumCoordinateSystem,
    SetPieceQualifier,
    SetPieceType,
    ShotResult,
    TakeOnResult,
)
from lxml import objectify

from databallpy import get_game_from_kloppy
from databallpy.data_parsers.kloppy_parsers.helpers import (
    _get_kloppy_minutes,
    _get_kloppy_spadl_action,
    _get_kloppy_spadl_bodypart,
    _get_kloppy_spadl_card,
)
from databallpy.game import EventData, Game, TrackingData
from databallpy.utils.warnings import DataBallPyWarning


class TestKloppySportec(unittest.TestCase):
    """Tests for Sportec data provider integration via Kloppy"""

    def setUp(self):
        """Load Sportec datasets for testing"""
        self.event_dataset = sportec.load_event(
            meta_data="tests/test_data/from_kloppy/sportec_meta.xml",
            event_data="tests/test_data/from_kloppy/sportec_events.xml",
        )
        self.tracking_dataset = sportec.load_tracking(
            meta_data="tests/test_data/from_kloppy/sportec_meta.xml",
            raw_data="tests/test_data/from_kloppy/sportec_positional.xml",
        )

    def test_sportec_both_datasets(self):
        """Test loading Sportec with both tracking and event data"""
        game = get_game_from_kloppy(
            tracking_dataset=self.tracking_dataset, event_dataset=self.event_dataset
        )

        assert isinstance(game, Game)
        assert isinstance(game.event_data, EventData)
        assert isinstance(game.tracking_data, TrackingData)
        assert len(game.event_data) == 29
        assert len(game.tracking_data) == 90100

    def test_sportec_tracking_only(self):
        """Test loading Sportec with only tracking data"""
        game = get_game_from_kloppy(tracking_dataset=self.tracking_dataset)

        assert isinstance(game, Game)
        assert isinstance(game.event_data, EventData)
        assert isinstance(game.tracking_data, TrackingData)
        assert game.event_data.empty
        assert len(game.tracking_data) == 90100

    def test_sportec_event_only(self):
        """Test loading Sportec with only event data"""
        game = get_game_from_kloppy(event_dataset=self.event_dataset)

        assert isinstance(game, Game)
        assert isinstance(game.event_data, EventData)
        assert isinstance(game.tracking_data, TrackingData)
        assert len(game.event_data) == 29
        assert game.tracking_data.empty

    def test_sportec_event_only_lxml_score(self):
        """Test that lxml score elements are converted to integers"""
        score = self.event_dataset.metadata.score
        self.event_dataset.metadata.score = type(score)(
            home=objectify.fromstring("<score>2</score>"),
            away=objectify.fromstring("<score>1</score>"),
        )
        game = get_game_from_kloppy(event_dataset=self.event_dataset)

        assert game.home_score == 2 and type(game.home_score) is int
        assert game.away_score == 1 and type(game.away_score) is int


class TestKloppyTracabStatsPerform(unittest.TestCase):
    """Tests for Tracab/StatsPerform data provider integration via Kloppy"""

    def setUp(self):
        self.event_dataset = statsperform.load_event(
            ma1_data="tests/test_data/from_kloppy/statsperform_event_ma1.json",
            ma3_data="tests/test_data/from_kloppy/statsperform_event_ma3.json",
        )
        self.tracking_dataset = tracab.load(
            meta_data="tests/test_data/from_kloppy/tracab_meta.xml",
            raw_data="tests/test_data/from_kloppy/tracab_raw.json",
        )

    def test_tracab_tracking_only_with_warning(self):
        """Test loading Tracab tracking data and verify DataBallPyWarning is raised"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            game = get_game_from_kloppy(tracking_dataset=self.tracking_dataset)

            assert isinstance(game, Game)
            assert isinstance(game.event_data, EventData)
            assert isinstance(game.tracking_data, TrackingData)
            assert game.event_data.empty
            assert len(game.tracking_data) == 2

            # Check that DataBallPyWarning was raised
            databallpy_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DataBallPyWarning)
            ]
            assert len(databallpy_warnings) > 0

            # Verify warning message about pitch middle point
            warning_messages = [str(warning.message) for warning in databallpy_warnings]
            assert any("middle point of the pitch" in msg for msg in warning_messages)
            assert any("kick-off" in msg for msg in warning_messages)

    def test_statsperform_event_only(self):
        """Test loading StatsPerform event data - no date warning expected"""
        # Note: The warning about game dates not being equal is NOT raised when loading
        # event data only, since there's no tracking dataset to compare dates with
        game = get_game_from_kloppy(event_dataset=self.event_dataset)

        assert isinstance(game, Game)
        assert isinstance(game.event_data, EventData)
        assert isinstance(game.tracking_data, TrackingData)
        assert game.tracking_data.empty
        assert len(game.event_data) == 1643


class TestKloppyStatsBombSignality(unittest.TestCase):
    """Tests for StatsBomb/Signality data provider integration via Kloppy"""

    def setUp(self):
        self.event_dataset = statsbomb.load(
            event_data="tests/test_data/from_kloppy/statsbomb_15986_event.json",
            lineup_data="tests/test_data/from_kloppy/statsbomb_15986_lineup.json",
        )
        self.tracking_dataset = signality.load(
            meta_data="tests/test_data/from_kloppy/signality_meta_data.json",
            venue_information="tests/test_data/from_kloppy/signality_venue_information.json",
            raw_data_feeds=[
                "tests/test_data/from_kloppy/signality_p1_raw_data_subset.json",
                "tests/test_data/from_kloppy/signality_p2_raw_data_subset.json",
            ],
        )

    def test_pitch_dimensions_mismatch_error(self):
        """Test that mismatched pitch dimensions raise ValueError"""
        with self.assertRaises(ValueError) as context:
            get_game_from_kloppy(
                event_dataset=self.event_dataset, tracking_dataset=self.tracking_dataset
            )

        # Verify error message mentions pitch dimensions
        error_message = str(context.exception)
        assert "dimensions" in error_message.lower() or "pitch" in error_message.lower()

    def test_coordinate_system_transformation(self):
        """Test loading with coordinate system transformation to match dimensions"""
        # Transform tracking dataset to match event dataset dimensions
        transformed_tracking = self.tracking_dataset.transform(
            to_coordinate_system=SecondSpectrumCoordinateSystem(
                pitch_length=105, pitch_width=68
            )
        )

        game = get_game_from_kloppy(
            tracking_dataset=transformed_tracking, event_dataset=self.event_dataset
        )

        assert isinstance(game, Game)
        assert isinstance(game.event_data, EventData)
        assert isinstance(game.tracking_data, TrackingData)
        assert not game.event_data.empty
        assert not game.tracking_data.empty

    def test_own_goal_event(self):
        """Test that a kloppy shot with result OWN_GOAL becomes an own_goal event"""
        shot = next(
            event
            for event in self.event_dataset.events
            if event.event_type == EventType.SHOT
        )
        shot.result = ShotResult.OWN_GOAL

        game = get_game_from_kloppy(event_dataset=self.event_dataset)

        event_data = game.event_data
        own_goals = event_data[event_data["databallpy_event"] == "own_goal"]
        assert own_goals["original_event_id"].tolist() == [shot.event_id]


class TestKloppyEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling in Kloppy integration"""

    def setUp(self):
        """Load Sportec datasets for testing"""
        self.event_dataset = sportec.load_event(
            meta_data="tests/test_data/from_kloppy/sportec_meta.xml",
            event_data="tests/test_data/from_kloppy/sportec_events.xml",
        )
        self.tracking_dataset = sportec.load_tracking(
            meta_data="tests/test_data/from_kloppy/sportec_meta.xml",
            raw_data="tests/test_data/from_kloppy/sportec_positional.xml",
        )

    def test_no_datasets_provided(self):
        """Test that calling get_game_from_kloppy with no datasets raises appropriate error"""
        with self.assertRaises((ValueError, TypeError)):
            get_game_from_kloppy()

    def test_none_datasets(self):
        """Test that calling get_game_from_kloppy with None datasets raises appropriate error"""
        with self.assertRaises((ValueError, TypeError)):
            get_game_from_kloppy(tracking_dataset=None, event_dataset=None)

    def test_wrong_type_dataset(self):
        """Test that calling get_game_from_kloppy with wrong type datasets raises appropriate error"""
        with self.assertRaises(TypeError):
            get_game_from_kloppy(tracking_dataset=["my_dataset"])

        with self.assertRaises(TypeError):
            get_game_from_kloppy(event_dataset={"my_dataset"})

    def test_non_equal_date(self):
        """Test that event and tracking dataset with non equal date raises userwarning and overwrites both dates"""

        self.event_dataset.metadata.date = dt.datetime(year=2025, month=1, day=1)
        self.tracking_dataset.metadata.date = dt.datetime(year=2024, month=1, day=1)

        with self.assertWarns(UserWarning):
            game = get_game_from_kloppy(
                event_dataset=self.event_dataset, tracking_dataset=self.tracking_dataset
            )

        assert game.tracking_data["datetime"].iloc[0].year == 1975  # fallback year


def _build_kloppy_event(builder: str, **kwargs):
    event_kwargs = {
        "event_id": "1",
        "period": None,
        "timestamp": dt.timedelta(0),
        "ball_owning_team": None,
        "ball_state": None,
        "team": None,
        "player": None,
        "coordinates": None,
        "raw_event": {},
        "qualifiers": None,
        "result": None,
    }
    event_kwargs.update(kwargs)
    return getattr(EventFactory(), builder)(**event_kwargs)


class TestKloppySpadl(unittest.TestCase):
    """Tests for the SPADL actions of kloppy event data"""

    def test_get_kloppy_spadl_bodypart(self):
        for body_part, expected in [
            (BodyPart.HEAD, "head"),
            (BodyPart.HEAD_OTHER, "head/other"),
            (BodyPart.LEFT_FOOT, "foot_left"),
            (BodyPart.RIGHT_FOOT, "foot_right"),
            (BodyPart.DROP_KICK, "foot"),
            (BodyPart.BOTH_HANDS, "other"),
        ]:
            event = _build_kloppy_event(
                "build_shot", qualifiers=[BodyPartQualifier(body_part)]
            )
            self.assertEqual(_get_kloppy_spadl_bodypart(event, "foot"), expected)
        self.assertEqual(
            _get_kloppy_spadl_bodypart(_build_kloppy_event("build_shot"), "other"),
            "other",
        )

    def test_get_kloppy_minutes(self):
        for period_id, expected in [(1, 10), (2, 55), (3, 100), (4, 115), (5, 130)]:
            period = Period(
                id=period_id,
                start_timestamp=dt.timedelta(0),
                end_timestamp=dt.timedelta(minutes=50),
            )
            event = _build_kloppy_event(
                "build_generic",
                period=period,
                timestamp=dt.timedelta(minutes=10, seconds=30),
            )
            with self.subTest(period_id=period_id):
                self.assertEqual(_get_kloppy_minutes(event), expected)

    def test_get_kloppy_spadl_card(self):
        self.assertEqual(_get_kloppy_spadl_card(CardType.FIRST_YELLOW), "yellow_card")
        self.assertEqual(_get_kloppy_spadl_card(CardType.SECOND_YELLOW), "red_card")
        self.assertEqual(_get_kloppy_spadl_card(CardType.RED), "red_card")
        self.assertIsNone(_get_kloppy_spadl_card(None))

    def test_get_kloppy_spadl_action(self):
        cases = [
            ("build_pass", PassResult.COMPLETE, [], ("pass", "success", "foot")),
            ("build_pass", PassResult.OUT, [], ("pass", "fail", "foot")),
            ("build_pass", PassResult.OFFSIDE, [], ("pass", "offside", "foot")),
            ("build_pass", None, [], (None, None, None)),
            (
                "build_pass",
                PassResult.COMPLETE,
                [PassQualifier(PassType.CROSS)],
                ("cross", "success", "foot"),
            ),
            (
                "build_pass",
                PassResult.INCOMPLETE,
                [
                    SetPieceQualifier(SetPieceType.FREE_KICK),
                    PassQualifier(PassType.HIGH_PASS),
                ],
                ("freekick_crossed", "fail", "foot"),
            ),
            (
                "build_pass",
                PassResult.COMPLETE,
                [SetPieceQualifier(SetPieceType.FREE_KICK)],
                ("freekick_short", "success", "foot"),
            ),
            (
                "build_pass",
                PassResult.COMPLETE,
                [
                    SetPieceQualifier(SetPieceType.CORNER_KICK),
                    PassQualifier(PassType.CROSS),
                ],
                ("corner_crossed", "success", "foot"),
            ),
            (
                "build_pass",
                PassResult.COMPLETE,
                [SetPieceQualifier(SetPieceType.CORNER_KICK)],
                ("corner_short", "success", "foot"),
            ),
            (
                "build_pass",
                PassResult.COMPLETE,
                [SetPieceQualifier(SetPieceType.GOAL_KICK)],
                ("goalkick", "success", "foot"),
            ),
            (
                "build_pass",
                PassResult.COMPLETE,
                [SetPieceQualifier(SetPieceType.THROW_IN)],
                ("throw_in", "success", "other"),
            ),
            (
                "build_shot",
                ShotResult.GOAL,
                [BodyPartQualifier(BodyPart.HEAD)],
                ("shot", "success", "head"),
            ),
            ("build_shot", ShotResult.SAVED, None, ("shot", "fail", "foot")),
            (
                "build_shot",
                ShotResult.GOAL,
                [SetPieceQualifier(SetPieceType.PENALTY)],
                ("shot_penalty", "success", "foot"),
            ),
            (
                "build_shot",
                ShotResult.OFF_TARGET,
                [SetPieceQualifier(SetPieceType.FREE_KICK)],
                ("shot_freekick", "fail", "foot"),
            ),
            ("build_shot", ShotResult.OWN_GOAL, [], ("bad_touch", "owngoal", "foot")),
            (
                "build_goalkeeper_event",
                None,
                [GoalkeeperQualifier(GoalkeeperActionType.SMOTHER)],
                ("keeper_save", "success", "other"),
            ),
            (
                "build_goalkeeper_event",
                None,
                [GoalkeeperQualifier(GoalkeeperActionType.CLAIM)],
                ("keeper_claim", "success", "other"),
            ),
            (
                "build_goalkeeper_event",
                None,
                [GoalkeeperQualifier(GoalkeeperActionType.PUNCH)],
                ("keeper_punch", "success", "other"),
            ),
            (
                "build_goalkeeper_event",
                None,
                [GoalkeeperQualifier(GoalkeeperActionType.PICK_UP)],
                ("keeper_pick_up", "success", "other"),
            ),
            (
                "build_goalkeeper_event",
                None,
                [GoalkeeperQualifier(GoalkeeperActionType.SAVE_ATTEMPT)],
                (None, None, None),
            ),
            ("build_foul_committed", None, None, ("foul", "fail", "foot")),
            (
                "build_foul_committed",
                None,
                [CardQualifier(CardType.SECOND_YELLOW)],
                ("foul", "red_card", "foot"),
            ),
            (
                "build_duel",
                DuelResult.LOST,
                [DuelQualifier(DuelType.GROUND)],
                ("tackle", "fail", "foot"),
            ),
            (
                "build_duel",
                DuelResult.WON,
                [DuelQualifier(DuelType.TACKLE)],
                ("tackle", "success", "foot"),
            ),
            (
                "build_duel",
                DuelResult.WON,
                [DuelQualifier(DuelType.GROUND), DuelQualifier(DuelType.LOOSE_BALL)],
                (None, None, None),
            ),
            (
                "build_duel",
                DuelResult.WON,
                [DuelQualifier(DuelType.AERIAL)],
                (None, None, None),
            ),
            (
                "build_interception",
                InterceptionResult.SUCCESS,
                None,
                ("interception", "success", "foot"),
            ),
            (
                "build_interception",
                InterceptionResult.LOST,
                [BodyPartQualifier(BodyPart.HEAD)],
                ("interception", "fail", "head"),
            ),
            (
                "build_take_on",
                TakeOnResult.COMPLETE,
                None,
                ("take_on", "success", "foot"),
            ),
            ("build_take_on", TakeOnResult.OUT, None, ("take_on", "fail", "foot")),
            ("build_clearance", None, None, ("clearance", "success", "foot")),
            ("build_miscontrol", None, None, ("bad_touch", "fail", "foot")),
            ("build_recovery", None, None, ("dribble", "success", "foot")),
            ("build_generic", None, None, (None, None, None)),
        ]
        for builder, result, qualifiers, expected in cases:
            with self.subTest(builder=builder, result=result, qualifiers=qualifiers):
                event = _build_kloppy_event(
                    builder, result=result, qualifiers=qualifiers
                )
                self.assertEqual(_get_kloppy_spadl_action(event), expected)

        carry = _build_kloppy_event(
            "build_carry", end_timestamp=dt.timedelta(1), end_coordinates=None
        )
        self.assertEqual(_get_kloppy_spadl_action(carry), ("dribble", "success", "foot"))
        card = _build_kloppy_event("build_card", card_type=CardType.RED)
        self.assertEqual(_get_kloppy_spadl_action(card), (None, None, None))

    def test_sportec_spadl(self):
        event_dataset = sportec.load_event(
            meta_data="tests/test_data/from_kloppy/sportec_meta.xml",
            event_data="tests/test_data/from_kloppy/sportec_events.xml",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataBallPyWarning)
            game = get_game_from_kloppy(event_dataset=event_dataset)
            actions = game.to_spadl()
        event_data = game.event_data

        assert len(actions) == 18
        # the away player scores an own goal in the right goal
        own_goal = event_data[event_data["action_result"] == "owngoal"].iloc[0]
        assert own_goal["original_event"] == "SHOT"
        assert own_goal[["end_x", "end_y"]].tolist() == [50.0, 0.0]
        # the card sets the result of the previous foul of the same player
        fouls = event_data[event_data["action_type"] == "foul"]
        assert fouls["action_result"].tolist() == ["yellow_card", "fail"]
        assert (
            event_data.loc[event_data["original_event"] == "CARD", "action_type"]
            .isna()
            .all()
        )
        # the recovery is a dribble to the start of the next action
        recovery_idx = event_data.index[event_data["original_event"] == "RECOVERY"][0]
        assert event_data.loc[recovery_idx, "action_type"] == "dribble"
        assert (
            event_data.loc[recovery_idx, ["end_x", "end_y"]].tolist()
            == event_data.loc[recovery_idx + 1, ["start_x", "start_y"]].tolist()
        )

    def test_statsbomb_spadl(self):
        event_dataset = statsbomb.load(
            event_data="tests/test_data/from_kloppy/statsbomb_15986_event.json",
            lineup_data="tests/test_data/from_kloppy/statsbomb_15986_lineup.json",
        )
        game = get_game_from_kloppy(event_dataset=event_dataset)
        actions = game.to_spadl()
        event_data = game.event_data

        assert len(actions) == 2388
        fouls = event_data[event_data["action_type"] == "foul"]
        assert (fouls["action_result"] == "yellow_card").sum() == 7
        assert (fouls["action_result"] == "red_card").sum() == 1
        recoveries = event_data[event_data["original_event"] == "RECOVERY"]
        dribbles = recoveries[recoveries["action_type"] == "dribble"]
        assert len(dribbles) > 0
        assert dribbles[["end_x", "end_y"]].notna().all().all()


if __name__ == "__main__":
    unittest.main()
