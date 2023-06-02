import unittest

import pandas as pd

from databallpy.utils.utils import (
    MISSING_INT,
    _to_float,
    _to_int,
    align_player_ids,
    get_matching_full_name,
)
from tests.expected_outcomes import MD_INMOTIO, MD_INSTAT


class TestUtils(unittest.TestCase):
    def test_to_float(self):
        assert pd.isnull(_to_float("s2"))
        assert pd.isnull(_to_float(None))
        assert 2.0 == _to_float("2")
        assert 3.3 == _to_float("3.3")

    def test_to_int(self):
        assert MISSING_INT == _to_int("s2.3")
        assert MISSING_INT == _to_int(None)
        assert 2 == _to_int("2")
        assert 3 == _to_int("3.3")

    def test_align_player_ids(self):
        unaligned_metadata = MD_INSTAT.copy()
        unaligned_metadata.away_players.loc[0, "id"] = 9
        aligned_metadata = align_player_ids(unaligned_metadata, MD_INMOTIO)
        assert aligned_metadata == MD_INSTAT

    def test_get_matching_full_name(self):
        input = "Bart Christaan Albert van den Boom"
        options = ["Bart Chris", "Bart van den Boom", "Piet Pieters"]
        output = get_matching_full_name(input, options)
        assert output == "Bart van den Boom"

    def test_missing_int(self):
        assert MISSING_INT == -999
