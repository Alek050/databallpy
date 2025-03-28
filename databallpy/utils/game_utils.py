import pandas as pd

from databallpy.events import DribbleEvent, PassEvent, ShotEvent, TackleEvent


def player_column_id_to_full_name(
    home_players: pd.DataFrame, away_players: pd.DataFrame, column_id: str
) -> str:
    """Simple function to get the full name of a player from the column id

    Args:
        home_players (pd.DataFrame): DataFrame containing all the home players
        away_players (pd.DataFrame): DataFrame containing all the away players
        column_id (str): the column id of a player, for instance "home_1"

    Returns:
        str: full name of the player
    """
    shirt_num = int(column_id.split("_")[1])
    if column_id[:4] == "home":
        return home_players.loc[
            home_players["shirt_num"] == shirt_num, "full_name"
        ].iloc[0]
    else:
        return away_players.loc[
            away_players["shirt_num"] == shirt_num, "full_name"
        ].iloc[0]


def player_id_to_column_id(
    home_players: pd.DataFrame, away_players: pd.DataFrame, player_id: int
) -> str:
    """Simple function to get the column id based on player id

    Args:
        home_players (pd.DataFrame): DataFrame containing all the home players
        away_players (pd.DataFrame): DataFrame containing all the away players
        player_id (int): id of the player

    Returns:
        str: column id of the player, for instance "home_1"
    """
    if (home_players["id"].eq(player_id)).any():
        num = home_players[home_players["id"] == player_id]["shirt_num"].iloc[0]
        return f"home_{num}"
    elif (away_players["id"].eq(player_id)).any():
        num = away_players[away_players["id"] == player_id]["shirt_num"].iloc[0]
        return f"away_{num}"
    else:
        raise ValueError(f"{player_id} is not in either one of the teams")


def create_event_attributes_dataframe(
    events: dict[str | int, ShotEvent | PassEvent | DribbleEvent | TackleEvent],
) -> pd.DataFrame:
    """Function to create a DataFrame from a dictionary of events

    Args:
        events (dict[str | int, ShotEvent | PassEvent | DribbleEvent]):
            The dictionary of events

    Returns:
        pd.DataFrame: DataFrame with the attributes of the events
    """
    if len(events.values()) == 0:
        return pd.DataFrame()
    attributes = list(events.values())[0].df_attributes
    res_dict = {
        attr: [getattr(event, attr) for event in events.values()] for attr in attributes
    }
    return pd.DataFrame(res_dict)
