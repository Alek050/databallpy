import unittest

from databallpy.schemas.event_data import EventData


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
