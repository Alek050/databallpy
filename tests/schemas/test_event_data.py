import unittest
import warnings

try:
    import pandera.pandas as pa
except ModuleNotFoundError:
    import pandera as pa

from databallpy.schemas.event_data import EventData, EventDataSchema
from databallpy.schemas.tracking_data import TrackingData
from databallpy.utils.constants import (
    SPADL_ACTION_TYPES,
    SPADL_BODYPARTS,
    SPADL_COLUMNS,
    SPADL_RESULTS,
)
from databallpy.utils.warnings import DataBallPyWarning
from tests.expected_outcomes import ED_OPTA


class TestEventDataSchemaSpadl(unittest.TestCase):
    def setUp(self):
        self.event_data = ED_OPTA.copy()

    def test_spadl_vocabularies(self):
        # the index in the list is the SPADL id, so the order must match socceraction
        self.assertEqual(len(SPADL_ACTION_TYPES), 23)
        self.assertEqual(SPADL_ACTION_TYPES.index("non_action"), 20)
        self.assertEqual(SPADL_ACTION_TYPES.index("goalkick"), 22)
        self.assertEqual(
            SPADL_RESULTS,
            ["fail", "success", "offside", "owngoal", "yellow_card", "red_card"],
        )
        self.assertEqual(
            SPADL_BODYPARTS,
            ["foot", "head", "other", "head/other", "foot_left", "foot_right"],
        )

    def test_spadl_columns_null(self):
        EventDataSchema.validate(self.event_data)

    def test_spadl_columns_valid_values(self):
        self.event_data["action_type"] = "pass"
        self.event_data["action_result"] = "success"
        self.event_data["action_bodypart"] = "foot_left"
        EventDataSchema.validate(self.event_data)

    def test_spadl_columns_invalid_values(self):
        for col, value in zip(SPADL_COLUMNS, ["kick", "succesful", "left_foot"]):
            with self.subTest(col=col):
                event_data = self.event_data.copy()
                event_data.loc[0, col] = value
                with self.assertRaises(pa.errors.SchemaError):
                    EventDataSchema.validate(event_data)

    def test_spadl_columns_required(self):
        for col in SPADL_COLUMNS:
            with self.subTest(col=col):
                with self.assertRaises(pa.errors.SchemaError):
                    EventDataSchema.validate(self.event_data.drop(columns=col))


class TestEventData(unittest.TestCase):
    def test_event_data_provider_default(self):
        df = EventData()
        assert df.provider == "unspecified"

    def test_event_data_provider_custom(self):
        df = EventData(provider="custom_provider")
        assert df.provider == "custom_provider"

    def test_event_data_provider_setter(self):
        df = EventData()
        with self.assertRaises(AttributeError):
            df.provider = "new_provider"

    def test_state(self):
        state = EventData().__getstate__()
        state["_provider"] = "other_provider"
        new_ed = EventData()
        self.assertEqual(new_ed.provider, "unspecified")

        new_ed.__setstate__(state)
        self.assertEqual(new_ed.provider, "other_provider")

    def test_databallpy_event_deprecation(self):
        ed = EventData({"databallpy_event": ["pass"], "action_type": ["pass"]})
        for access in (
            lambda: ed["databallpy_event"],
            lambda: ed.databallpy_event,
            lambda: ed.loc[:, "databallpy_event"],
        ):
            with self.assertWarns(DataBallPyWarning) as cm:
                access()
            # the warning should point to the code of the user
            self.assertEqual(cm.filename, __file__)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ed["action_type"]
            ed.action_type
            EventDataSchema.validate(ED_OPTA)


class TestTrackingDataDeprecation(unittest.TestCase):
    def test_databallpy_event_deprecation(self):
        td = TrackingData({"databallpy_event": ["pass"], "action_type": ["pass"]})
        for access in (lambda: td["databallpy_event"], lambda: td.databallpy_event):
            with self.assertWarns(DataBallPyWarning):
                access()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            td["action_type"]
