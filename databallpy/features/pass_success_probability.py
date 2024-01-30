import pandas as pd
import numpy as np
import line_profiler
import math
from databallpy.utils.utils import sigmoid
from databallpy.utils.logging import create_logger

LOGGER = create_logger(__name__)

def get_pass_success_probability_grid(
        frame: pd.Series,
        defending_team: str,
        grid_size: tuple[int, int] = (32, 24),
        pitch_size: tuple[float, float] = (106., 68.),
        max_velocity_players:float=7.8,
        max_acceleration_players:float=3.,
        decay_constant:float=0.379,
    ) -> np.ndarray:
    """Function to get the pass success probability grid for a frame. The pass 
    success probability is a function of the pass location, the pass end location,
    the current player locations, and the current player velocities. This function
    only changes the possible end location based on a grid to get a pass success
    probability for any location on the pitch.

    Args:
        frame (pd.Series): The tracking data frame when the pass is happening.
        defending_team (str): The defending team side, either "home" or "away".
        grid_size (tuple[int, int], optional): Size of the grid to put over the pitch. 
            Defaults to (32, 24).
        pitch_size (tuple[float, float], optional): The pitch size in meters (x, y).
            Defaults to (106., 68.).
        max_velocity_players (float, optional): The max velocity of the players in m/s.
            Defaults to 7.8.
        max_acceleration_players (float, optional): The max acceleration of the players
            in m/(s^2). Defaults to 3.0.
        min_ball_velocity (float, optional): The minimal ball velocity during the pass
            in m/s. Defaults to 4.0.
        max_ball_velocity (float, optional): The maximal ball velocity during the pass
            in m/s. Defaults to 20.0.

    Returns:
        np.ndarray: Grid with for every cell the pass success probability given the 
            current frame. The shape of the array is the same shape as the grid size.
    """

    try:
        if not isinstance(frame, pd.Series):
            raise TypeError(
               f"'frame' should be a pd.Series, not a {type(frame)}"
                )
        if not defending_team in ["home", "away"]:
            raise ValueError(
                "'defending_team' should be either 'home' or "
                f"'away', not {defending_team}"
            )
        for val, name in zip([grid_size, pitch_size], ["grid_size", "pitch_size"]):
            if not isinstance(val, (tuple, list, np.ndarray)):
                raise TypeError(
                    f"'{name}' should be a tuple, not a {type(val)}"
                )
            if not len(val) == 2:
                raise ValueError(
                    f"'{name}' should have a length of 2, not {len(val)}"
                )
        for val, name in zip(
            [max_velocity_players, max_acceleration_players], 
            ["max_velocity_players", "max_acceleration_players"]
            ):
            if not isinstance(val, (float, np.floating)):
                raise TypeError(
                    f"'{name}' should be a float, not a {type(val)}"
                )

        x_range = np.linspace(-pitch_size[0] / 2, pitch_size[0] / 2, grid_size[0] + 1)
        y_range = np.linspace(pitch_size[1] / 2, -pitch_size[1] / 2, grid_size[1] + 1)
        grid = np.zeros(grid_size)
        pass_start_loc = np.array([frame["ball_x"], frame["ball_y"]])
        attacking_team = "home" if defending_team == "away" else "away"
        defending_column_ids = [x[:-2] for x in frame.index if x.startswith(defending_team) and x.endswith("_x")]
        attacking_column_ids = [x[:-2] for x in frame.index if x.startswith(attacking_team) and x.endswith("_x")]
        
        pos_0_defenders = np.array([[frame[f"{column_id}_x"], frame[f"{column_id}_y"]] for column_id in defending_column_ids if not pd.isnull(frame[f"{column_id}_x"])])
        vel_0_defenders = np.array([[frame[f"{column_id}_vx"], frame[f"{column_id}_vy"]] for column_id in defending_column_ids if not pd.isnull(frame[f"{column_id}_vx"])])
        pos_0_attackers= np.array([[frame[f"{column_id}_x"], frame[f"{column_id}_y"]] for column_id in attacking_column_ids if not pd.isnull(frame[f"{column_id}_x"])])
        vel_0_attackers = np.array([[frame[f"{column_id}_vx"], frame[f"{column_id}_vy"]] for column_id in attacking_column_ids if not pd.isnull(frame[f"{column_id}_vx"])])

        if len(vel_0_attackers) == 0:
            return grid

        for i, (x1, x2) in enumerate(zip(x_range[:-1], x_range[1:])):
            x = (x1 + x2) / 2
            for j, (y1, y2) in enumerate(zip(y_range[:-1], y_range[1:])):
                y = (y1 + y2) / 2
                
                grid[i, j] = get_pass_success_probability(
                    frame, 
                    np.array([x, y]), 
                    defending_team, 
                    pass_start_loc=pass_start_loc, 
                    defending_column_ids=defending_column_ids, 
                    attacking_column_ids=attacking_column_ids,
                    pos_0_defenders=pos_0_defenders,
                    vel_0_defenders=vel_0_defenders,
                    pos_0_attackers=pos_0_attackers,
                    vel_0_attackers=vel_0_attackers,
                    max_velocity_players=max_velocity_players,
                    max_acceleration_players=max_acceleration_players,
                    decay_constant=decay_constant,
                )

        return grid
    except Exception as e:
        LOGGER.exception(
            "Found an unexpected exception in get_pass_success_probability_grid()"
            f" \n{e}"
            )
        raise e


def get_pass_success_probability(
        frame:pd.Series, 
        pass_end_loc:tuple[float, float], 
        defending_team:str, 
        max_velocity_players: float = 7.8,
        max_acceleration_players:float=3.0,
        pass_start_loc:np.ndarray = None,
        defending_column_ids:str = None,
        attacking_column_ids:str = None,
        pos_0_defenders:np.ndarray = None,
        vel_0_defenders:np.ndarray = None,
        pos_0_attackers:np.ndarray = None,
        vel_0_attackers:np.ndarray = None,
        decay_constant:float = 0.378,
        
        ) -> float:
    """
    Function to get the pass success probability grid for a frame. The pass 
    success probability is a function of the pass location, the pass end location,
    the current player locations, and the current player velocities.

    Args:
        frame (pd.Series): _description_
        end_loc (tuple[float, float]): _description_
        defending_team (str): _description_
        reaction_time (float, optional): _description_. Defaults to 0.7.

    Returns:
        float: _description_
    """
    try:
        if not isinstance(frame, pd.Series):
            raise TypeError(
               f"'frame' should be a pd.Series, not a {type(frame)}"
                )
        if not defending_team in ["home", "away"]:
            raise ValueError(
                "'defending_team' should be either 'home' or "
                f"'away', not {defending_team}"
            )
        if not isinstance(pass_end_loc, np.ndarray):
            raise TypeError(
                f"'end_loc' should be a np.ndarray, not a '{type(pass_end_loc)}'"
            )
        if not len(pass_end_loc) == 2:
            raise ValueError(
                f"'end_loc' should be of length 2 (x, y), not length {(len(pass_end_loc))}."
            )

        for val, name in zip(
            [max_velocity_players, max_acceleration_players], 
            ["max_velocity_players", "max_acceleration_players"]
            ):
            if not isinstance(val, (float, np.floating)):
                raise TypeError(
                    f"'{name}' should be a float, not a {type(val)}"
                )

        if defending_column_ids is None or attacking_column_ids is None:
            attacking_team = "home" if defending_team == "away" else "away"
            defending_column_ids = [x[:-2] for x in frame.index if x.startswith(defending_team) and x.endswith("_x")]
            attacking_column_ids = [x[:-2] for x in frame.index if x.startswith(attacking_team) and x.endswith("_x")]

        if pass_start_loc is None:
            pass_start_loc = np.array([frame["ball_x"], frame["ball_y"]])

        ball_to_end_distance = math.dist(pass_start_loc, pass_end_loc)

        # Find the attacker closest to determine the ball speed
        if pos_0_attackers is None or vel_0_attackers is None:
            pos_0_attackers = np.array([
                [frame[f"{column_id}_x"], frame[f"{column_id}_y"]] for column_id in attacking_column_ids if not pd.isnull(frame[f"{column_id}_vx"])
                ])
            vel_0_attackers = np.array([
                [frame[f"{column_id}_vx"], frame[f"{column_id}_vy"]] for column_id in attacking_column_ids if not pd.isnull(frame[f"{column_id}_vx"])
                ])
        if len(vel_0_attackers) == 0:
            return np.nan
        distances = np.linalg.norm(pos_0_attackers.astype(np.float64) - pass_end_loc.astype(np.float64), axis=1)
         
        att_idx = np.argmin(distances)
        player_time_to_end = get_player_time_to_k(
            pass_end_loc, 
            pos_0_attackers[att_idx], 
            vel_0_attackers[att_idx], 
            max_velocity_players=max_velocity_players, 
            max_acceleration_players=max_acceleration_players
            )
        
        # get the ball initial speed based on the player_time_to_end
        ball_time_to_end = get_ball_time_to_k_linear(ball_to_end_distance)
        diff = np.abs(player_time_to_end - ball_time_to_end)
        ball_time_to_end = ball_time_to_end + diff if player_time_to_end > ball_time_to_end else ball_time_to_end - np.min([diff, .6])
        ball_start_velocity = get_ball_start_velocity(ball_to_end_distance, player_time_to_end, decay_constant=decay_constant)

        attacker_claim_probabilities = np.zeros(len(vel_0_attackers))
        for i, (pos_0, vel_0) in enumerate(zip(pos_0_attackers, vel_0_attackers)):
            attacker_claim_probabilities[i] = get_interception_probability(
                k_pos=pass_end_loc,
                player_pos=pos_0,
                player_vel=vel_0,
                ball_start_velocity=ball_start_velocity,
                ball_to_k_distance=ball_to_end_distance,
                ball_time_to_k=ball_time_to_end,
                max_velocity_players=max_velocity_players,
                max_acceleration_players=max_acceleration_players,
                decay_constant=decay_constant,
            )
        attacker_claim_probability = 1 - np.prod(1 - np.sort(attacker_claim_probabilities))

        # Find the defender that is closest to the pass line for possible interception
        if pos_0_defenders is None or vel_0_defenders is None:
            pos_0_defenders = np.array([[frame[f"{column_id}_x"], frame[f"{column_id}_y"]] for column_id in defending_column_ids if not pd.isnull(frame[f"{column_id}_vx"])])
            vel_0_defenders = np.array([[frame[f"{column_id}_vx"], frame[f"{column_id}_vy"]] for column_id in defending_column_ids if not pd.isnull(frame[f"{column_id}_vx"])])

        if len(vel_0_defenders) == 0:
            return np.nan
        k_positions = np.zeros((len(vel_0_defenders), 2))
        for i, pos_0 in enumerate(pos_0_defenders):
            k_positions[i] = get_loc_shortest_to_ball_trajectory(pass_start_loc, pass_end_loc, pos_0)

        distances = np.linalg.norm(k_positions - pos_0_defenders, axis=1)

        interception_probabilities = np.zeros(len(vel_0_defenders))
        defender_claim_probabilities = np.zeros(len(vel_0_defenders))
        for i, (k_pos, pos_0, vel_0) in enumerate(zip(k_positions, pos_0_defenders, vel_0_defenders)):
            interception_probabilities[i] = get_interception_probability(
                k_pos=k_pos,
                player_pos=pos_0,
                player_vel=vel_0,
                ball_start_velocity=ball_start_velocity,
                ball_to_k_distance=math.dist(pass_start_loc, k_pos),
                max_velocity_players=max_velocity_players,
                max_acceleration_players=max_acceleration_players,
                decay_constant=decay_constant,   
            )
            defender_claim_probabilities[i] = get_interception_probability(
                k_pos=pass_end_loc,
                player_pos=pos_0,
                player_vel=vel_0,
                ball_start_velocity=ball_start_velocity,
                ball_to_k_distance=ball_to_end_distance,
                ball_time_to_k=ball_time_to_end,
                max_velocity_players=max_velocity_players,
                max_acceleration_players=max_acceleration_players,
                decay_constant=decay_constant,
            )
        interception_probability = 1 - np.prod(1 - np.sort(interception_probabilities))
        defender_claim_probability = 1 - np.prod(1 - np.sort(defender_claim_probabilities))

        high_ball_prob = sigmoid(ball_to_end_distance,  -0.85947493, -0.33047364, 30.0412692 ,  0.93385372)
        interception_probability = (1 - high_ball_prob) * interception_probability        
        attacker_wins_claim_probability = sigmoid(attacker_claim_probability - defender_claim_probability, 1, 4, 0, 0)

        return (1 - interception_probability) * attacker_wins_claim_probability
    except Exception as e:
        LOGGER.exception(
            "Found an unexpected exception in get_pass_success_probability_grid()"
            f" \n{e}"
            )
        raise e
    

def get_loc_shortest_to_ball_trajectory(start_loc: np.ndarray, end_loc: np.ndarray, player_pos: np.ndarray) -> np.ndarray:
    """Find the location on the line segment between start_loc and end_loc closest to player_pos.

    Args:
        start_loc (np.ndarray): The start location of the line segment.
        end_loc (np.ndarray): The end location of the line segment.
        player_pos (np.ndarray): The position of the player.

    Returns:
        np.ndarray: The location on the line segment closest to player_pos.
    """

    ball_trajectory_vec = end_loc - start_loc
    player_start_vec = player_pos - start_loc

    scalar_projection = np.dot(player_start_vec, ball_trajectory_vec) / np.dot(ball_trajectory_vec, ball_trajectory_vec)
    t = max(0, min(scalar_projection, 1))
    closest_point = start_loc + t * ball_trajectory_vec

    return closest_point

def get_interception_probability(
        k_pos:np.ndarray,
        player_pos:np.ndarray,
        player_vel:np.ndarray,
        ball_start_velocity:float,
        ball_to_k_distance:float ,
        ball_time_to_k:float = None,
        max_velocity_players:float = 7.8,
        max_acceleration_players:float = 3.0,
        player_time_to_k:float = None,
        decay_constant:float = 0.378,
        ) -> float:

    if ball_time_to_k is None:
        ball_time_to_k = get_ball_time_to_k(ball_to_k_distance, ball_start_velocity, decay_constant=decay_constant)
    if player_time_to_k is None:
        player_time_to_k = get_player_time_to_k(k_pos, player_pos, player_vel, max_velocity_players=max_velocity_players, max_acceleration_players=max_acceleration_players)

    return sigmoid(ball_time_to_k-player_time_to_k, 1, 5, -0.25, 0)

def get_ball_time_to_k_linear(ball_to_k_distance:float, beta_0=0.64, beta_1=0.057) -> float:
    return beta_0 + beta_1 * ball_to_k_distance

def get_ball_time_to_k(ball_to_k_distance:float, ball_start_velocity:float=17., decay_constant=0.379) -> float:
    """Simple function to calculate the time the ball needs to get
    from the start location to the end location. This function is
    based on the assumption that the ball travels with a constant
    velocity.

    Args:
        ball_to_k_distance(float): The distance between the start
            location and the end location in meters.
        ball_velocity (float, optional): The moving velocity of the ball. 
            Defaults to 12..

    Returns:
        float: Time for the ball to reach the k position.
    """
    if ball_to_k_distance == 0:
        return 0

    numerator = np.log(-np.clip((ball_to_k_distance*decay_constant - ball_start_velocity), a_min=-np.inf, a_max=-1.) / ball_start_velocity)

    denominator = decay_constant
    return min((-numerator / denominator), 7.0)

#     return 
    # return -1 / decay_constant * np.log(
    #     (-decay_constant*ball_to_k_distance / ball_start_velocity) + 1
    #     )

def get_ball_start_velocity(ball_to_k_distance:float, ball_time_to_k:float, decay_constant=0.379) -> float:
    """_summary_

    Args:
        ball_to_k_distance (float): _description_
        ball_time_to_k (float): _description_
        decay_constant (float, optional): _description_. Defaults to 0.388.

    Returns:
        float: _description_
    """

    return (ball_to_k_distance*decay_constant) / (-np.exp(-decay_constant*ball_time_to_k) + 1)

    # return ball_to_k_distance / (ball_time_to_k * (-.5*ball_time_to_k*decay_constant + 1) )

def get_player_time_to_k(k_pos: np.ndarray, pos_0: np.ndarray, vel_0: np.ndarray, max_velocity_players=7.8, max_acceleration_players=3.0) -> float:
    """
    Calculate the time it takes for a player to reach a certain position (k_pos) on the pitch.

    Args:
        k_pos (np.ndarray): Position of point k on the pitch.
        pos_0 (np.ndarray): Initial position of the player.
        vel_0 (np.ndarray): Initial velocity vector of the player.
        maximal_velocity_players (float, optional): Maximum velocity of the player. Defaults to 7.8.
        maximal_acceleration_players (float, optional): Maximum acceleration of the player. Defaults to 3.0.

    Returns:
        float: Time it takes for the player to reach point k.
    """
    initial_k_dist = math.dist(pos_0, k_pos)

    # Calculate the component of the initial velocity vector in the direction of k_pos
    initial_speed_along_k = np.dot(vel_0, (k_pos - pos_0)) / initial_k_dist
    
    # Calculate the time to reach full velocity in the direction of k_pos
    time_to_max_vel = max(0, (max_velocity_players - initial_speed_along_k) / max_acceleration_players)
    
    # Calculate the distance covered during the acceleration phase
    distance_accel = 0.5 * max_acceleration_players * time_to_max_vel**2

    if initial_k_dist <= distance_accel:
        return np.sqrt(2 * initial_k_dist / max_acceleration_players)
    else:
        return time_to_max_vel + (initial_k_dist - distance_accel) / max_velocity_players  


if __name__ == "__main__":
    import os
    from databallpy import get_saved_match
    from databallpy.visualize import plot_soccer_pitch
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # plt.plot(np.linspace(0, 100, 100), [get_ball_time_to_k(ball_to_k_distance=x,  ball_start_velocity=20, decay_constant=0.1) for x in np.linspace(0, 100, 100)])
    # plt.show()
    
    data_dir = "/Volumes/Storage Alexander/Tracking Data/Eredivisie_2021_2022_processed"
    files = sorted([x for x in os.listdir(data_dir) if x.endswith(".pickle")])
    file = files[0]
    match = get_saved_match(file, path=data_dir)
    frame = match.tracking_data[match.tracking_data["databallpy_event"]=="pass"].iloc[8]
    pass_event = match.passes_df[match.passes_df["event_id"] == frame["event_id"]].iloc[0]
    start_loc = frame[["ball_x" , "ball_y"]].values
    pass_end_loc = pass_event[["end_x"   , "end_y"    ]].values
    pass_end_loc = np.array([-40., -30.,])
    # end_loc = np.array([-40., 0.])
    defending_team = "away" if pass_event["team_id"] == match.home_team_id else "home"

    prob = get_pass_success_probability(
        frame, 
        pass_end_loc, 
        defending_team=defending_team,
        pass_start_loc=start_loc,
        max_velocity_players=7.8,
        max_acceleration_players=3.,
        decay_constant=0.1,
        )
    print(prob)

    grid = get_pass_success_probability_grid(
        frame, 
        defending_team=defending_team, 
        max_velocity_players=7.8,
        max_acceleration_players=3.,
        grid_size=(128, 96),
        decay_constant=0.1,
        )

    fig, ax = plot_soccer_pitch(pitch_color="white", field_dimen=(106, 68))
    ax.scatter(frame["ball_x"], frame["ball_y"], color="black")
    arrow = mpatches.FancyArrowPatch(
        start_loc,
        pass_end_loc,
        mutation_scale=10,
        color="black",
    )
    ax.add_patch(arrow)
    x_home = frame[[x + "_x" for x in match.home_players_column_ids()]].values
    y_home = frame[[y + "_y" for y in match.home_players_column_ids()]].values
    vx_home = frame[[x + "_vx" for x in match.home_players_column_ids()]].values
    vy_home = frame[[y + "_vy" for y in match.home_players_column_ids()]].values
    x_away = frame[[x + "_x" for x in match.away_players_column_ids()]].values
    y_away = frame[[y + "_y" for y in match.away_players_column_ids()]].values
    vx_away = frame[[x + "_vx" for x in match.away_players_column_ids()]].values
    vy_away = frame[[y + "_vy" for y in match.away_players_column_ids()]].values
    if defending_team == "home":
        colors = ["blue", "red"]
    else:
        colors = ["red", "blue"]
    ax.scatter(x_home, y_home, color=colors[0])
    ax.scatter(x_away, y_away, color=colors[1])

    for x, y, vx, vy in zip(x_home, y_home, vx_home, vy_home):
        if pd.isnull(vx):
            continue
        arrow = mpatches.FancyArrowPatch(
            (x, y),
            (x + vx, y + vy),
            mutation_scale=7
        )

        ax.add_patch(arrow)

    for x, y, vx, vy in zip(x_away, y_away, vx_away, vy_away):
        if pd.isnull(vx):
            continue
        arrow = mpatches.FancyArrowPatch(
            (x, y),
            (x + vx, y + vy),
            mutation_scale=7
        )
        ax.add_patch(arrow)
    
    # grid = np.flip(grid.T, axis=1)
    # grid.T[5, 11] = 0

    imshow = ax.imshow(grid.T, extent=[-53, 53, -34, 34], cmap="coolwarm", alpha=0.7, vmin=0, vmax=1, origin='upper', aspect='auto')
    plt.colorbar(imshow, label='Pass success probability')
    plt.title(f"Pass success probability: {prob}, True succes: {pass_event['outcome']}, Defending team: {defending_team}")
    plt.show()
    # prob = get_pass_success_probability(frame, end_loc, defending_team=defending_team)
    import pdb; pdb.set_trace()

    lp = line_profiler.LineProfiler()
    lp_wrapper = lp(get_pass_success_probability)
    lp_wrapper( 
        frame, 
        pass_end_loc,
        defending_team=defending_team,
        )
    lp.print_stats()
    # lp = line_profiler.LineProfiler()
    # lp_wrapper = lp(get_interception_probability)
    # lp_wrapper(
    #     frame[["ball_x", "ball_y"]].values, 
    #     end_loc, 
    #     frame[["home_1_x", "home_1_y"]].values,
    #     frame[["home_1_vx", "home_1_vy"]].values,
    #     )
    # lp.print_stats()
    
