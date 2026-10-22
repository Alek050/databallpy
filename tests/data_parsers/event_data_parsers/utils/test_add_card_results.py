import unittest

import pandas as pd

from databallpy.data_parsers.event_data_parsers.utils import _add_card_results


class TestAddCardResults(unittest.TestCase):
    def test_add_card_results(self):
        event_data = pd.DataFrame(
            {
                "period_id": [1, 1, 1, 1, 1, 2, 2, 1, 1],
                "player_id": [1, 1, 1, 2, 2, 3, 3, 4, 4],
                "datetime": pd.to_datetime(
                    [
                        "2023-01-01 00:00:00",
                        "2023-01-01 00:00:20",
                        "2023-01-01 00:00:30",
                        "2023-01-01 00:01:00",
                        "2023-01-01 00:02:01",
                        "2023-01-01 00:03:00",
                        "2023-01-01 00:03:10",
                        "2023-01-01 00:04:10",
                        "2023-01-01 00:04:00",
                    ]
                ),
                "action_type": [
                    "foul",
                    "foul",
                    None,
                    "foul",
                    None,
                    "pass",
                    None,
                    "foul",
                    None,
                ],
                "action_result": [
                    "fail",
                    "fail",
                    None,
                    "fail",
                    None,
                    "success",
                    None,
                    "fail",
                    None,
                ],
            }
        )
        cards = {2: "yellow_card", 4: "red_card", 6: "yellow_card", 8: "yellow_card"}
        res = _add_card_results(event_data, cards)

        # the last foul of the same player within 60 seconds before the card gets
        # the card, cards long after the foul, without a foul or before the foul
        # are ignored
        self.assertListEqual(
            res["action_result"].tolist(),
            ["fail", "yellow_card", None, "fail", None, "success", None, "fail", None],
        )
