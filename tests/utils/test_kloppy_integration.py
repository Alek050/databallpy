import datetime as dt
import unittest
import warnings

from kloppy import signality, sportec, statsbomb, statsperform, tracab
from kloppy.domain import SecondSpectrumCoordinateSystem

from databallpy import get_game_from_kloppy
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


if __name__ == "__main__":
    unittest.main()
