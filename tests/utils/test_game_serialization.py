import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from databallpy.utils.constants import MISSING_INT
from databallpy.utils.get_game import get_game, get_saved_game


class TestGameSerialization(unittest.TestCase):
    def _get_ma_game(self, provider="opta"):
        return get_game(
            event_data_loc="tests/test_data/ma13_test.xml",
            event_metadata_loc="tests/test_data/ma2_test.xml",
            event_data_provider=provider,
        )

    def _assert_round_trip(self, game):
        original = game.copy()
        with tempfile.TemporaryDirectory() as path:
            game.save_game(name="game", path=path, verbose=False)
            saved_game = get_saved_game(name="game", path=path)

        self.assertEqual(game, original)
        self.assertEqual(saved_game, original)
        pd.testing.assert_frame_equal(saved_game.event_data, original.event_data)
        pd.testing.assert_frame_equal(saved_game.pass_events, original.pass_events)

    def test_opta_ma_round_trip(self):
        self._assert_round_trip(self._get_ma_game())

    def test_statsperform_ma_round_trip(self):
        self._assert_round_trip(self._get_ma_game("statsperform"))

    def test_mixed_identifiers_round_trip(self):
        game = self._get_ma_game()
        values = ["00123", "-999", MISSING_INT, None, np.int64(42), "42"]
        game.event_data.loc[game.event_data.index[:6], "player_id"] = values
        game.pass_events["receiver_player_id"] = game.pass_events[
            "receiver_player_id"
        ].astype(object)
        game.pass_events.loc[game.pass_events.index[:2], "receiver_player_id"] = (
            pd.Series(["hplayermaid01", MISSING_INT], dtype=object)
        )
        self._assert_round_trip(game)

    def test_saved_game_without_encoding_metadata(self):
        game = self._get_ma_game()
        values = ["00123", "-999", MISSING_INT, None, np.int64(42), "42"]
        game.event_data.loc[game.event_data.index[:6], "player_id"] = values

        with tempfile.TemporaryDirectory() as path:
            game.save_game(name="game", path=path, verbose=False)
            metadata_path = os.path.join(path, "game", "metadata.json")
            with open(metadata_path) as file:
                metadata = json.load(file)
            # the player_id column must actually have been json-encoded,
            # otherwise dropping the key below would not exercise the
            # fallback decode-skip path at all.
            self.assertEqual(
                metadata["json_encoded_columns"]["event_data"], ["player_id"]
            )
            metadata.pop("json_encoded_columns", None)
            with open(metadata_path, "w") as file:
                json.dump(metadata, file)

            saved_game = get_saved_game(name="game", path=path)

        # without the encoding metadata, get_saved_game has no way of knowing
        # player_id needs decoding, so it is left as raw json strings
        decoded = saved_game.event_data["player_id"].map(json.loads, na_action="ignore")
        pd.testing.assert_series_equal(
            decoded, game.event_data["player_id"], check_names=False
        )
