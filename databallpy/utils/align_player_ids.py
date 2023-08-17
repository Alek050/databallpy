from difflib import SequenceMatcher

from databallpy.load_data.metadata import Metadata


def get_matching_full_name(full_name: str, options: list) -> str:
    """Function that finds the best match between a name and a list of names,
    based on difflib.SequenceMatcher

    Args:
        full_name (str): name that has to be matched
        options (list): list of possible names

    Returns:
        str: the name from the option list that is the best match
    """
    similarity = []
    for option in options:
        s = SequenceMatcher(None, full_name, option)
        similarity.append(s.ratio())
    return options[similarity.index(max(similarity))]


def align_player_ids(event_metadata: Metadata, tracking_metadata: Metadata) -> Metadata:
    """Function to align player ids when the player ids between tracking and event
    data are different. The player ids in the metadata of the tracking data is leading.

    Args:
        event_metadata (Metadata): metadata of the event data
        tracking_metadata (Metadata): metadata of the tracking data

    Returns:
        Metadata: metadata of the event date with alignes player ids
    """
    for idx, row in event_metadata.home_players.iterrows():
        full_name_tracking_metadata = get_matching_full_name(
            row["full_name"], tracking_metadata.home_players["full_name"]
        )
        id_tracking_data = tracking_metadata.home_players.loc[
            tracking_metadata.home_players["full_name"] == full_name_tracking_metadata,
            "id",
        ].values[0]
        event_metadata.home_players.loc[idx, "id"] = id_tracking_data

    for idx, row in event_metadata.away_players.iterrows():
        full_name_tracking_metadata = get_matching_full_name(
            row["full_name"], tracking_metadata.away_players["full_name"]
        )
        id_tracking_data = tracking_metadata.away_players.loc[
            tracking_metadata.away_players["full_name"] == full_name_tracking_metadata,
            "id",
        ].values[0]
        event_metadata.away_players.loc[idx, "id"] = id_tracking_data

    return event_metadata
