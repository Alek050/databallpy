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
