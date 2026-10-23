import pandas as pd

MAX_CARD_DELAY = pd.Timedelta(seconds=60)


def _add_card_results(event_data: pd.DataFrame, cards: dict) -> pd.DataFrame:
    """Function to set the card results of the SPADL foul actions. Cards are no SPADL
    actions, but the result of the last foul of the same player in the same period,
    at most 60 seconds before the card. Cards without such a foul (e.g. for dissent)
    are ignored.

    Args:
        event_data (pd.DataFrame): The event data with the SPADL columns, the
            player_id, period_id and datetime columns
        cards (dict): The index of the card events with their SPADL result, either
            yellow_card or red_card

    Returns:
        pd.DataFrame: The event data with the card results of the fouls
    """
    for card_idx, card_result in cards.items():
        card = event_data.loc[card_idx]
        fouls = event_data.loc[
            (event_data["action_type"] == "foul")
            & (event_data["player_id"] == card["player_id"])
            & (event_data["period_id"] == card["period_id"])
            & (event_data["datetime"] <= card["datetime"])
            & (event_data["datetime"] >= card["datetime"] - MAX_CARD_DELAY)
        ]
        if len(fouls) > 0:
            event_data.loc[fouls["datetime"].idxmax(), "action_result"] = card_result
    return event_data
