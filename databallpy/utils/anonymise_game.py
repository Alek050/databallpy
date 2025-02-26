import hashlib
import secrets

import numpy as np
import pandas as pd
from pandas._libs.tslibs.timestamps import Timestamp

from databallpy.game import Game
from databallpy.utils.constants import MISSING_INT
from databallpy.utils.errors import DataBallPyError


def anonymise_game(
    game: Game,
    keys_df: pd.DataFrame,
    base_time: Timestamp = pd.to_datetime("1980-1-1 15:00:00", utc=True),
) -> tuple[Game, pd.DataFrame]:
    """Function to anonymise a game. The function will replace all player names with a
    unique identifier as well as all teams. Furthermore, it will replace all player
    jersey numbers with a counter from 1 to n_players in that team. Finally, it will
    replace all datetime objects to datetime.time objects so that every game starts at
    1980 15:00:00.

    Since soccer data often has names and teams that are known, a hash is not enough.
    Therefore, we gather the name/team name and add 8 bytes of random data (salt). After
    that, we hash the result. This way, we can still check if two names are the same,
    but the hash is not the same as the name. For every name/ team name, the first 8
    characters of the hash are used as pseudonym. For players, the pseudonym is
    'P-' + pseudonym. For teams, the pseudonym is 'T-' + pseudonym.

    Args:
        game (Game): game to anonymise
        keys_df (pd.DataFrame): df containing the keys for known players and teams. If
            new player of team is found, a random key will be generated and added to
            this df.
        base_time (Timestamp, optional): The base timestamp to which to set the
            start of the game datetime.
            Defaults to pd.to_datetime("1980-1-1 15:00:00", utc=True).

    Returns:
        tuple[Game, pd.DataFrame]: tuple containing the game with anonymised players
            and teams and the potentially updated keys dataframe.

    Raises:
        ValueError: if base_time is not a timezone aware timestamp
        DataBallPyError: if keys_df does not have 1 of the obligated column name "name",
            "pseudonym", "salt", "original_id".
        DataBallPyError: if keys_df has more than 4 columns
    """

    if not base_time.tz:
        raise ValueError("base_time must be a timezone aware timestamp")

    for col in ["name", "pseudonym", "salt", "original_id"]:
        if col not in keys_df.columns:
            raise DataBallPyError(
                f"keys_df does not contain a {col} column, this is mandatory!"
            )

    if len(keys_df.columns) > 4:
        raise DataBallPyError("keys_df is not allowed to have more than 4 columns")

    game = game.copy()
    game, keys_df = anonymise_players(game, keys_df)
    game, keys_df = anonymise_teams(game, keys_df)
    game = anonymise_datetime(game, base_time)

    # Return game
    return game, keys_df


def add_new_pseudonym(
    keys: pd.DataFrame, key_type: str, name: str, old_id: int | str
) -> pd.DataFrame:
    """Function to create a new key for a specific key type. The function will create a
    new key that does not yet exist in the keys dataframe. It will also check if the
    key_type is valid. key_type can be one of the following: player, team. If not,
    the function will raise an error.

    Since soccer data often has names and teams that are known, a hash is not enough.
    Therefore, we gather the name/team name and add 8 bytes of random data (salt). After
    that, we hash the result. This way, we can still check if two names are the same,
    but the hash is not the same as the name. For every name/ team name, the first 8
    characters of the hash are used as pseudonym. For players, the pseudonym is
    'P-' + pseudonym. For teams, the pseudonym is 'T-' + pseudonym.

    Args:
        keys (pd.DataFrame): dataframe containing all keys of teams and players
        key_type (str): type of key to create. Can be one of the following: player,
            team. If not, the function will raise an error.
        name (str): name of the player/team to anonymise
        old_id union(int, str): original id of the player/team to anonymise

    Returns:
        pd.DataFrame: The potentially updated keys dataframe.

    Raises:
        ValueError: if key_type is not one of the following: player, team
    """

    # Check if key_type is valid
    if key_type not in ["player", "team"]:
        raise ValueError(
            "key_type must be one of the following: 'player'"
            f" or 'team', not {key_type}",
        )

    if old_id in keys["original_id"].values:
        return keys

    salt = secrets.token_hex(8)
    hash = hashlib.sha256((name + salt).encode()).hexdigest()
    pseudonym = key_type[0].upper() + "-" + hash[:8]

    if pseudonym in keys["pseudonym"].values:
        pseudonym = add_new_pseudonym(keys, key_type, name, old_id)

    # add name, pseudonym and salt to keys dataframe
    keys = pd.concat(
        [
            keys,
            pd.DataFrame(
                {
                    "name": name,
                    "pseudonym": pseudonym,
                    "salt": salt,
                    "original_id": old_id,
                },
                index=[0],
            ),
        ],
        ignore_index=True,
        sort=True,
    )
    return keys


def get_player_mappings(
    home_players: pd.DataFrame, away_players: pd.DataFrame, keys: pd.DataFrame
) -> tuple[dict, dict, dict, dict, pd.DataFrame]:
    """Function to get the player mappings. The function will create a mapping from
    original player id to pseudonym, a mapping from original player name to pseudonym,
    a mapping from original jersey number to anonymised jersey number for the home team,
    and a mapping from original jersey number to anonymised jersey number for the away
    team.

    Args:
        home_players (pd.DataFrame): dataframe containing all players of the home team
        away_players (pd.DataFrame): dataframe containing all players of the away team
        keys (pd.DataFrame): dataframe containing all keys of teams and players

    Returns:
        tuple[dict, dict, dict, dict, pd.DataFrame]: tuple containing the player id map,
            the player name map, the home players jersey map, the away players jersey
            map, and the potentially updated keys dataframe.
    """
    player_id_map = {}
    player_name_map = {}
    home_players_jersey_map = {}
    away_players_jersey_map = {}
    for side, players, jersey_map in zip(
        ["home", "away"],
        [home_players, away_players],
        [home_players_jersey_map, away_players_jersey_map],
    ):
        counter = 1
        for _, row in players.iterrows():
            original_player_name = row["full_name"]
            original_id = row["id"]
            if original_id not in keys["original_id"].values:
                keys = add_new_pseudonym(
                    keys, "player", original_player_name, original_id
                )
            pseudonym = keys.loc[keys["original_id"] == original_id, "pseudonym"].values[
                0
            ]

            player_id_map[original_id] = pseudonym
            player_name_map[original_player_name] = f"{side}_{counter}"

            jersey_map[row["shirt_num"]] = counter
            counter += 1

    return (
        player_id_map,
        player_name_map,
        home_players_jersey_map,
        away_players_jersey_map,
        keys,
    )


def anonymise_players(game: Game, keys: pd.DataFrame) -> tuple[Game, pd.DataFrame]:
    """Function to anonymise all players in a game. The function will replace all
    player ids with a unique identifier. It will first look if the player id is
    already in the keys dataframe. If not, it will create a new pseudonym for that
    player. After that, it will replace all player names with their pseudonym.
    The function will also replace all jersey numbers with a counter from 1 to n_players
    in that team. The player name will be replaced with the team_side + the counter.

    Args:
        game (Game): game to anonymise players for
        keys (pd.DataFrame): dataframe containing all keys of teams and players

    Returns:
        tuple[Game, pd.DataFrame]: tuple containing the game with anonymised players
            and the potentially updated keys dataframe.
    """

    (
        player_id_map,
        player_name_map,
        home_players_jersey_map,
        away_players_jersey_map,
        keys,
    ) = get_player_mappings(game.home_players, game.away_players, keys)

    game.home_players["id"] = game.home_players["id"].map(player_id_map)
    game.away_players["id"] = game.away_players["id"].map(player_id_map)
    game.home_players["full_name"] = game.home_players["full_name"].map(player_name_map)
    game.away_players["full_name"] = game.away_players["full_name"].map(player_name_map)
    game.home_players["shirt_num"] = game.home_players["shirt_num"].map(
        home_players_jersey_map
    )
    game.away_players["shirt_num"] = game.away_players["shirt_num"].map(
        away_players_jersey_map
    )

    # update tracking data
    if len(game.tracking_data) > 0:
        game.tracking_data = rename_tracking_data_columns(
            game.tracking_data, home_players_jersey_map, "home"
        )
        game.tracking_data = rename_tracking_data_columns(
            game.tracking_data, away_players_jersey_map, "away"
        )

    # update event data
    if len(game.event_data) > 0:
        game.event_data["player_name"] = game.event_data["player_name"].map(
            player_name_map
        )
        game.event_data["player_id"] = game.event_data["player_id"].map(player_id_map)
        if "to_player_name" in game.event_data.columns:
            game.event_data["to_player_name"] = game.event_data["to_player_name"].map(
                player_name_map
            )
        if "to_player_id" in game.event_data.columns:
            game.event_data["to_player_id"] = game.event_data["to_player_id"].map(
                player_id_map
            )
        if "player_name" in game.pass_events.columns:
            game.pass_events["player_name"] = game.pass_events["player_name"].map(
                player_name_map
            )
        game.pass_events["player_id"] = game.pass_events["player_id"].map(player_id_map)
        if "player_name" in game.dribble_events.columns:
            game.dribble_events["player_name"] = game.dribble_events["player_name"].map(
                player_name_map
            )
        game.dribble_events["player_id"] = game.dribble_events["player_id"].map(
            player_id_map
        )
        if "player_name" in game.shot_events.columns:
            game.shot_events["player_name"] = game.shot_events["player_name"].map(
                player_name_map
            )
        game.shot_events["player_id"] = game.shot_events["player_id"].map(player_id_map)

    return game, keys


def rename_tracking_data_columns(
    tracking_data: pd.DataFrame,
    jersey_map: dict,
    side: str,
) -> pd.DataFrame:
    """Function to rename the columns in the tracking data. The function will rename
    all player columns.

    Args:
        tracking_data (pd.DataFrame): tracking data to rename columns for
        jersey_map (dict): dictionary containing the mapping from original
            jersey numbers to new jersey numbers for one team.
        side (str): either "home" or "away", indicating which team the mapping is for

    Returns:
        pd.DataFrame: tracking data with renamed columns
    """

    col_rename_mappings = {}
    side_columns = [col for col in tracking_data.columns if col.startswith(side)]
    for col in side_columns:
        jersey_number = int(col.split("_")[1])

        if jersey_number not in jersey_map.keys():
            raise ValueError(
                "Something went wrong when anonymising the data."
                f"Jersey number {jersey_number} from the {side} team does not "
                "have a anonymised jersey number in the mapping.",
            )

        affix = col.split("_")[2]
        col_rename_mappings[col] = f"{side}_{jersey_map[jersey_number]}_{affix}"

    tracking_data = tracking_data.rename(columns=col_rename_mappings)
    return tracking_data


def get_team_mappings(
    home_team_name: str,
    away_team_name: str,
    home_team_id: int,
    away_team_id: int,
    keys: pd.DataFrame,
) -> tuple[dict, dict, pd.DataFrame]:
    """Function to get the team mappings. The function will create a mapping from
    original team id to pseudonym and a mapping from original team name to pseudonym.

    Args:
        home_team_name (str): home team name
        away_team_name (str): away team name
        home_team_id (int): home team id
        away_team_id (int): away team id
        keys (pd.DataFrame): dataframe containing all keys of teams and players

    Returns:
        tuple[dict, dict, pd.DataFrame]: tuple containing the team id map, the team
        name map, and the potentially updated keys dataframe.

    """
    team_id_map = {}
    team_name_map = {}
    for original_team_name, original_team_id in zip(
        [home_team_name, away_team_name], [home_team_id, away_team_id]
    ):
        if original_team_name not in keys["name"].values:
            keys = add_new_pseudonym(keys, "team", original_team_name, original_team_id)

        pseudonym = keys.loc[keys["name"] == original_team_name, "pseudonym"].values[0]

        team_id_map[original_team_id] = pseudonym
        team_name_map[original_team_name] = pseudonym

    return team_id_map, team_name_map, keys


def anonymise_teams(game: Game, keys: pd.DataFrame) -> tuple[Game, pd.DataFrame]:
    """Function to anonymise the teams in a game. The function will replace all
    team ids with a unique identifier. It will first look if the team name is
    already in the keys dataframe. If not, it will create a new pseudonym for that
    team. After that, it will replace all team names with their pseudonym. Both
    team names and team ids will be replaced with the same pseudonym.

    Args:
        game (Game): game to anonymise teams for
        keys (pd.DataFrame): dataframe containing all keys of teams and players

    Returns:
        tuple[Game, pd.DataFrame]: tuple containing the game with anonymised teams
            and the potentially updated keys dataframe.
    """

    team_id_map, team_name_map, keys = get_team_mappings(
        game.home_team_name,
        game.away_team_name,
        game.home_team_id,
        game.away_team_id,
        keys,
    )

    game.home_team_id = team_id_map[game.home_team_id]
    game.away_team_id = team_id_map[game.away_team_id]
    game.home_team_name = team_name_map[game.home_team_name]
    game.away_team_name = team_name_map[game.away_team_name]

    if len(game.event_data) > 0:
        game.event_data["team_id"] = game.event_data["team_id"].map(team_id_map)
        game.pass_events["team_id"] = game.pass_events["team_id"].map(team_id_map)
        game.shot_events["team_id"] = game.shot_events["team_id"].map(team_id_map)
        game.dribble_events["team_id"] = game.dribble_events["team_id"].map(team_id_map)

    return game, keys


def anonymise_datetime(
    game: Game, base_time: Timestamp = pd.to_datetime("1980-1-1 15:00:00", utc=True)
) -> Game:
    """Function to anonymise the datetime of a game. The function will replace all
    datetime objects so that the tracking data starts exactly at the base_time. The
    function will change the event data datetime accordingly to keep the same time
    difference between the two sources. The function will also change the frame column
    in the tracking data since it is sometimes based on a timestamp.

    Args:
        game (Game): game to anonymise datetime for
        base_time (Timestamp, optional): The base timestamp.
            Defaults to pd.to_datetime("1980-1-1 15:00:00", utc=True).

    Returns:
        Game: game with anonymised datetime
    """

    dt_delta_defined = False
    if len(game.tracking_data) > 0:
        dt_delta, dt_delta_defined = (
            game.tracking_data["datetime"].iloc[0] - base_time,
            True,
        )

        game.tracking_data["datetime"] = (
            game.tracking_data["datetime"].dt.tz_convert("UTC") - dt_delta
        )
        game.periods["start_datetime_td"] = (
            game.periods["start_datetime_td"].dt.tz_convert("UTC") - dt_delta
        )
        game.periods["end_datetime_td"] = (
            game.periods["end_datetime_td"].dt.tz_convert("UTC") - dt_delta
        )

        new_frames = np.arange(len(game.tracking_data)) + 1
        frame_map = dict(zip(game.tracking_data["frame"], new_frames))
        max_frame = frame_map[game.tracking_data["frame"].max()]
        game.tracking_data["frame"] = np.arange(len(game.tracking_data)) + 1
        game.periods["start_frame"] = (
            game.periods["start_frame"].map(frame_map).fillna(max_frame).astype(int)
        )
        game.periods["end_frame"] = (
            game.periods["end_frame"].map(frame_map).fillna(max_frame).astype(int)
        )

        game.home_players["start_frame"] = (
            game.home_players["start_frame"].map(frame_map).fillna(max_frame).astype(int)
        )
        game.home_players["end_frame"] = (
            game.home_players["end_frame"].map(frame_map).fillna(max_frame).astype(int)
        )
        game.away_players["start_frame"] = (
            game.away_players["start_frame"].map(frame_map).fillna(max_frame).astype(int)
        )
        game.away_players["end_frame"] = (
            game.away_players["end_frame"].map(frame_map).fillna(max_frame).astype(int)
        )

    if len(game.event_data) > 0:
        if not dt_delta_defined:
            dt_delta, dt_delta_defined = (
                game.event_data["datetime"].iloc[0] - base_time,
                True,
            )

        game.event_data["datetime"] = (
            game.event_data["datetime"].dt.tz_convert("UTC") - dt_delta
        )
        game.periods["start_datetime_ed"] = (
            game.periods["start_datetime_ed"].dt.tz_convert("UTC") - dt_delta
        )
        game.periods["end_datetime_ed"] = (
            game.periods["end_datetime_ed"].dt.tz_convert("UTC") - dt_delta
        )

        if game._is_synchronised:
            game.event_data["tracking_frame"] = (
                game.event_data["tracking_frame"]
                .map(frame_map)
                .fillna(MISSING_INT)
                .astype(int)
            )

        game.pass_events["datetime"] = (
            game.pass_events["datetime"].dt.tz_convert("UTC") - dt_delta
        )
        game.shot_events["datetime"] = (
            game.shot_events["datetime"].dt.tz_convert("UTC") - dt_delta
        )
        game.dribble_events["datetime"] = (
            game.dribble_events["datetime"].dt.tz_convert("UTC") - dt_delta
        )

    return game
