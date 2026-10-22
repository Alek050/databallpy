import json
import os
import tempfile
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from databallpy.game import _serialize_numpy_number
from databallpy.schemas import TrackingData
from databallpy.utils.constants import MISSING_INT
from databallpy.utils.get_game import _decode_json_column, get_game, get_saved_game
from tests.pandas_testing import assert_frame_equal, assert_series_equal


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
        assert_frame_equal(saved_game.event_data, original.event_data)
        assert_frame_equal(saved_game.pass_events, original.pass_events)
        return saved_game

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

    def test_pandas_score_round_trip(self):
        game = self._get_ma_game()
        # Scores selected from a pandas match summary are NumPy integers.
        home_score = pd.Series([game.home_score]).iloc[0]
        game = replace(game, home_score=home_score)
        self._assert_round_trip(game)

    def test_numpy_integer_metadata_round_trip(self):
        for dtype in (np.int32, np.int64, np.uint64):
            with self.subTest(dtype=dtype):
                game = get_game(
                    event_data_loc="tests/test_data/f24_test.xml",
                    event_metadata_loc="tests/test_data/f7_test.xml",
                    event_data_provider="opta",
                )
                game = replace(
                    game,
                    home_team_id=dtype(game.home_team_id),
                    away_team_id=dtype(game.away_team_id),
                    home_score=dtype(game.home_score),
                    away_score=dtype(game.away_score),
                )
                loaded = self._assert_round_trip(game)
                for name in ("home_team_id", "away_team_id", "home_score", "away_score"):
                    self.assertEqual(getattr(loaded, name), int(getattr(game, name)))
                    self.assertIs(type(getattr(loaded, name)), int)

    def test_numpy_floating_metadata_round_trip(self):
        for dtype in (np.float16, np.float32, np.float64):
            for container in (list, tuple):
                with self.subTest(dtype=dtype, container=container):
                    game = replace(
                        self._get_ma_game(),
                        pitch_dimensions=container([dtype(105.1), dtype(68.1)]),
                    )
                    loaded = self._assert_round_trip(game)
                    self.assertEqual(
                        loaded.pitch_dimensions,
                        [float(x) for x in game.pitch_dimensions],
                    )

    def test_numpy_frame_rate_round_trip(self):
        for frame_rate in (np.int64(25), np.float32(29.97)):
            with self.subTest(frame_rate=frame_rate):
                game = self._get_ma_game()
                game = replace(
                    game,
                    tracking_data=TrackingData(
                        game.tracking_data,
                        provider=game.tracking_data.provider,
                        frame_rate=frame_rate,
                    ),
                )
                loaded = self._assert_round_trip(game)
                self.assertEqual(loaded.frame_rate, frame_rate)

    def test_numpy_missing_score_round_trip(self):
        game = replace(
            self._get_ma_game(), home_score=np.float32(np.nan), away_score=None
        )
        self._assert_round_trip(game)

    def test_metadata_encoder_rejects_unsupported_values(self):
        for value in (object(), np.array([1, 2])):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaisesRegex(TypeError, "is not JSON serializable"):
                    json.dumps({"value": value}, default=_serialize_numpy_number)

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
        decoded = _decode_json_column(saved_game.event_data["player_id"])
        assert_series_equal(decoded, game.event_data["player_id"], check_names=False)
