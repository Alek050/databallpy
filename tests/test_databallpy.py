import unittest

from databallpy import __version__


class TestDataballpy(unittest.TestCase):
    def test_version(self):
        assert __version__ == "0.5.3"
