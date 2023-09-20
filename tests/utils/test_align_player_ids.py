import unittest

from databallpy.utils.align_player_ids import align_player_ids, get_matching_full_name
from tests.expected_outcomes import MD_INMOTIO, MD_INSTAT


class TestAlginPlayerIds(unittest.TestCase):
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
