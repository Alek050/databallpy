import unittest

import pandas as pd

from databallpy.utils.utils import _to_float, _to_int


class TestUtils(unittest.TestCase):
    def test_to_float(self):
        assert pd.isnull(_to_float("s2"))
        assert pd.isnull(_to_float(None))
        assert 2.0 == _to_float("2")
        assert 3.3 == _to_float("3.3")

    def test_to_int(self):
        assert -999 == _to_int("s2.3")
        assert -999 == _to_int(None)
        assert 2 == _to_int("2")
        assert 3 == _to_int("3.3")
