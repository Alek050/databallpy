from difflib import SequenceMatcher

from databallpy.data_parsers.metadata import Metadata


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


def align_player_ids_jersey(metadata1: Metadata, metadata2: Metadata) -> Metadata:
    """Function to align player ids based on the jersey numbers. The player ids in
    metadata1 will be replaced by the player ids in metadata2.

    Args:
        metadata1 (Metadata): metadata
        metadata2 (Metadata): metadtaa

    Returns:
        Metadata: metadata1 with replaced player ids
    """

    metadata1.home_players.drop(columns=["id"], inplace=True)
    metadata1.away_players.drop(columns=["id"], inplace=True)

    metadata1.home_players.insert(
        0,
        "id",
        metadata1.home_players["shirt_num"].map(
            metadata2.home_players.set_index("shirt_num")["id"]
        ),
    )
    metadata1.away_players.insert(
        0,
        "id",
        metadata1.away_players["shirt_num"].map(
            metadata2.away_players.set_index("shirt_num")["id"]
        ),
    )
    return metadata1


def align_player_ids_name_similarity(
    metadata1: Metadata, metadata2: Metadata
) -> Metadata:
    """Function to align player ids when the player ids between tracking and event
    data are different. The player ids in the metadata of metadata1 will be replaced
    by the player ids in the metadata of metadata2.

    Args:
        metadata1 (Metadata): metadata
        metadata2 (Metadata): metadata

    Returns:
        Metadata: metadata1 with aligned player ids
    """
    for idx, row in metadata1.home_players.iterrows():
        full_name_tracking_metadata = get_matching_full_name(
            row["full_name"], metadata2.home_players["full_name"].to_list()
        )
        id_tracking_data = metadata2.home_players.loc[
            metadata2.home_players["full_name"] == full_name_tracking_metadata,
            "id",
        ].values[0]
        metadata1.home_players.loc[idx, "id"] = id_tracking_data

    for idx, row in metadata1.away_players.iterrows():
        full_name_tracking_metadata = get_matching_full_name(
            row["full_name"], metadata2.away_players["full_name"].to_list()
        )
        id_tracking_data = metadata2.away_players.loc[
            metadata2.away_players["full_name"] == full_name_tracking_metadata,
            "id",
        ].values[0]
        metadata1.away_players.loc[idx, "id"] = id_tracking_data

    return metadata1
