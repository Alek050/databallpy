import pandas as pd

from databallpy.data_parsers.metadata import Metadata
from databallpy.events import DribbleEvent, PassEvent, ShotEvent, TackleEvent


def _remove_offside_players(
    col_ids: list[str], tracking_frame: pd.Series, tolerance: float = 0.5
) -> list[str]:
    """Update the list of col_ids to remove the players that are offside.

    Args:
        col_ids (list[str]): the column names of player jersey combination (like 'home_1' and 'away_2')
        tracking_frame (pd.Series): frame of tracking data
        tolerance (float): The tolerance for the offside rule to correct for noise in the data

    Returns:
        list[str]: updated list with col ids.
    """

    if not tracking_frame["ball_status"] == "alive":
        return col_ids

    change_cols = [col for col in tracking_frame.index if col.endswith("_x")]

    if tracking_frame["team_possession"] == "away":
        tracking_frame = (tracking_frame[change_cols] * -1).copy()
        attacking_team = "away"
        defending_team = "home"
    else:
        attacking_team = "home"
        defending_team = "away"

    att_x = (
        tracking_frame[[x for x in change_cols if attacking_team in x]]
        .dropna()
        .sort_values()
    )
    def_x = (
        tracking_frame[[x for x in change_cols if defending_team in x]]
        .dropna()
        .sort_values()
    )

    midline_x = 0
    defending_line_x = def_x.iloc[-2]
    ball_line_x = tracking_frame["ball_x"]

    offside_line = max(midline_x, defending_line_x + tolerance, ball_line_x + tolerance)
    def_col_ids = [col[:-2] for col in def_x.index.to_list()]
    att_col_ids = [
        col_id[:-2] for col_id in att_x.index.to_list() if att_x[col_id] <= offside_line
    ]
    all_cols = att_col_ids + def_col_ids
    return [x for x in all_cols if x in col_ids]


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


def _add_starter_information(
    metadata: Metadata,
    tracking_data: pd.DataFrame | None = None,
    event_data: pd.DataFrame | None = None,
) -> Metadata:
    """Function to add starter information to metadata when not provided by the data source.

    This function will only add starter information when it's missing (all None or all False).
    It prioritizes tracking data if available, otherwise uses event data.

    Args:
        metadata (Metadata): The metadata object containing player information
        tracking_data (pd.DataFrame | None): Optional tracking data to determine starters
            from first 22 players (11 per team) with non-null data. Defaults to None.
        event_data (pd.DataFrame | None): Optional event data to determine starters
            from substitute events and player participation. Defaults to None.

    Returns:
        Metadata: Updated metadata with starter information added
    """
    # Check if starter information already exists and has meaningful values
    home_has_starters = (
        "starter" in metadata.home_players.columns
        and metadata.home_players["starter"].notna().any()
        and metadata.home_players["starter"].any()
    )
    away_has_starters = (
        "starter" in metadata.away_players.columns
        and metadata.away_players["starter"].notna().any()
        and metadata.away_players["starter"].any()
    )

    if home_has_starters and away_has_starters:
        # Starter information already exists for both teams, no need to add
        return metadata

    # Initialize starter column if it doesn't exist
    if "starter" not in metadata.home_players.columns:
        metadata.home_players["starter"] = False
    if "starter" not in metadata.away_players.columns:
        metadata.away_players["starter"] = False

    # Try to use tracking data first
    if tracking_data is not None and not tracking_data.empty:
        _add_starters_from_tracking_data(metadata, tracking_data)
    elif event_data is not None and not event_data.empty:
        _add_starters_from_event_data(metadata, event_data)

    return metadata


def _add_starters_from_tracking_data(
    metadata: Metadata, tracking_data: pd.DataFrame
) -> None:
    """Add starter information based on tracking data.

    Identifies the first 22 players (11 from each team) that have non-null tracking data
    in the first frames of the game.

    Args:
        metadata (Metadata): The metadata object to update
        tracking_data (pd.DataFrame): The tracking data
    """
    # Get the first frame of the first period
    first_period = metadata.periods_frames[metadata.periods_frames["period_id"] == 1]
    if first_period.empty or "start_frame" not in first_period.columns:
        return

    start_frame = first_period["start_frame"].iloc[0]

    # Find the first frame in tracking data
    first_frame_data = tracking_data[tracking_data["frame"] == start_frame]
    if first_frame_data.empty:
        # Use the very first frame available
        first_frame_data = tracking_data.iloc[[0]]

    # Get all player columns (those ending with _x)
    player_x_cols = [col for col in tracking_data.columns if col.endswith("_x")]

    # Separate home and away players
    home_cols = [col for col in player_x_cols if col.startswith("home_")]
    away_cols = [col for col in player_x_cols if col.startswith("away_")]

    # Find players with non-null data in the first frame
    home_starters = []
    away_starters = []

    for col in home_cols:
        if first_frame_data[col].notna().any():
            shirt_num = int(col.split("_")[1])
            home_starters.append(shirt_num)

    for col in away_cols:
        if first_frame_data[col].notna().any():
            shirt_num = int(col.split("_")[1])
            away_starters.append(shirt_num)

    # Update metadata with starter information
    metadata.home_players["starter"] = metadata.home_players["shirt_num"].isin(
        home_starters
    )
    metadata.away_players["starter"] = metadata.away_players["shirt_num"].isin(
        away_starters
    )


def _add_starters_from_event_data(metadata: Metadata, event_data: pd.DataFrame) -> None:
    """Add starter information based on event data.

    Uses substitute events to determine starters. A player is a starter if:
    1. They performed an event before the first substitute event, OR
    2. They were substituted in but performed an event somewhere during the game
       (meaning they must have started)

    Args:
        metadata (Metadata): The metadata object to update
        event_data (pd.DataFrame): The event data
    """
    if event_data.empty or "databallpy_event" not in event_data.columns:
        return

    # Find substitute events (assuming they are marked in some way)
    # Look for substitute/substitution related event types
    substitute_mask = (
        event_data["event_type"]
        .str.lower()
        .str.contains("substitut", case=False, na=False)
    )
    substitute_events = event_data[substitute_mask].sort_values("event_id")

    # Get all player IDs that participated in events
    participating_players = set(event_data["player_id"].dropna().unique())

    # If there are no substitute events, assume all participating players are starters
    if substitute_events.empty:
        metadata.home_players["starter"] = metadata.home_players["id"].isin(
            participating_players
        )
        metadata.away_players["starter"] = metadata.away_players["id"].isin(
            participating_players
        )
        return

    # Get the event_id of the first substitute
    first_sub_event_id = substitute_events["event_id"].iloc[0]

    # Players who performed events before the first substitute are starters
    events_before_first_sub = event_data[event_data["event_id"] < first_sub_event_id]
    starters_from_early_events = set(
        events_before_first_sub["player_id"].dropna().unique()
    )

    # Players who were substituted in
    # This is tricky without standardized substitute event structure
    # We'll identify them by looking for players in substitute events
    if "player_id" in substitute_events.columns:
        # Players who were subbed in but still appear in events must be starters
        # (this is a conservative approach)
        for _, sub_event in substitute_events.iterrows():
            player_id = sub_event.get("player_id")
            if pd.notna(player_id) and player_id in participating_players:
                # Check if this player appears in events after being "subbed"
                # If they do, they were likely actually a starter
                events_after_sub = event_data[
                    event_data["event_id"] > sub_event["event_id"]
                ]
                if player_id in events_after_sub["player_id"].values:
                    starters_from_early_events.add(player_id)

    # Combine starters
    all_starters = starters_from_early_events

    # Update metadata
    metadata.home_players["starter"] = metadata.home_players["id"].isin(all_starters)
    metadata.away_players["starter"] = metadata.away_players["id"].isin(all_starters)
