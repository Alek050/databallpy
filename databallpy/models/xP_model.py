
import numpy as np
import pandas as pd
import math
import pickle
import os

with open(os.path.join(os.path.dirname(__file__), "passing_model_decay_params.pkl"), "rb") as file:
    OPTIMAL_DECAY_PARAMS = pickle.load(file)
LOOP_INITIAL_VELOCITIES = np.linspace(8, 25, (25-8)*10 + 1)


def exponential_decay(X:np.ndarray, c0, c1, c2, c3):
    """
    Exponential decay function.
    
    Args:
        X (array-like): Time values.
        v0 (float): Initial velocity.
        decay_constant (float): Decay constant.
    
    Returns:
        array-like: Decay model values.
    """
    t= X[0]
    v0 = X[1]
    return c0 + v0 * np.exp(-c1*(t+c2) + c3)


def get_ball_distance_during_pass(initial_velocity: np.ndarray, time: np.ndarray, decay_params:np.ndarray) -> float:
    """
    Calculate the distance covered by a ball during a pass given initial conditions.

    Args:
        initial_velocity (float, np.ndarray): Initial velocity of the ball.
        time (np.ndarray, float): Time of the pass.
        decay_params (np.ndarray): [any, 5] with the c0, c1, c2, c3, and K

    Returns:
        np.ndarray: Distance covered by the ball during the pass for every t.
    """
    if decay_params.ndim == 1:
        decay_params = decay_params[np.newaxis, :]
    if time.ndim == 1:
        time = time[:, np.newaxis]

    e_part = np.exp(decay_params[:, 3].T - decay_params[:, 1].T * (decay_params[:, 2].T + time))
    return (decay_params[:, 0].T * time) - (initial_velocity * e_part) / decay_params[:, 1].T + decay_params[:, 4].T
def get_initial_ball_velocity(pass_distance: float, pass_time: float) -> float:
    """
    Calculate the initial velocity of a ball during a pass given the desired pass distance and time.

    Args:
        pass_distance (float): Desired distance covered by the pass.
        pass_time (float): Time of the pass.

    Returns:
        float: Initial velocity of the ball.
    """
    distances = get_ball_distance_during_pass(LOOP_INITIAL_VELOCITIES, pass_time[:, np.newaxis], OPTIMAL_DECAY_PARAMS)
    return np.clip(LOOP_INITIAL_VELOCITIES[np.argmin(np.abs(distances - pass_distance[:, np.newaxis]), axis=1)], a_min=8.0, a_max=25.0)

def get_ball_time_to_k(initial_ball_velocity:float, pass_distance:float) -> float:
    """
    Calculate the time for the ball to reach a certain distance during a pass.

    Args:
        initial_ball_velocity (float): Initial velocity of the ball (in meters/second).
        pass_distance (float): Distance covered by the pass (in meters).

    Returns:
        float: Time taken for the ball to reach the specified distance during the pass.

    Notes:
        - This function calculates the time taken for the ball to cover a specific distance during a pass.
        - It iteratively computes the ball's distance at various time points and returns the time when the distance is closest to the specified pass distance.
        - The optimal decay parameters are retrieved from the provided dictionary based on the initial ball velocity.
        - The function returns the time (in seconds) for the ball to reach the specified distance during the pass.
    """

    if isinstance(pass_distance, np.ndarray) and pass_distance.ndim == 1:
        pass_distance = pass_distance[:, np.newaxis]

    idx = np.where(initial_ball_velocity[:, np.newaxis] == LOOP_INITIAL_VELOCITIES)[1]
    t = np.arange(0, 800)/100.
    decay_params = OPTIMAL_DECAY_PARAMS[idx]
    distance = get_ball_distance_during_pass(initial_ball_velocity, t, decay_params)


    return t[np.argmin(np.abs(pass_distance[:, :, np.newaxis] - distance.T[:, np.newaxis, :]), axis=2)]



def get_loc_shortest_to_ball_trajectory(ball_start_loc: np.ndarray, ball_end_loc: np.ndarray, player_pos: np.ndarray) -> np.ndarray:
    """Find the location on the line segment between ball_start_loc and ball_end_loc closest to player_pos.

    Args:
        ball_start_loc (np.ndarray): The start location of the pass.
        ball_end_loc (np.ndarray): The end location of the pass.
        player_pos (np.ndarray): The position of the player.

    Returns:
        np.ndarray: The location on the pass line closest to the player_pos.
    """

    ball_trajectory_vec = ball_end_loc - ball_start_loc
    player_start_vec = player_pos - ball_start_loc

    player_start_vec = player_start_vec[np.newaxis, :, :]
    ball_trajectory_vec = ball_trajectory_vec[:, np.newaxis, :]

    scalar_projection = np.sum(player_start_vec * ball_trajectory_vec, axis=2) / np.sum(ball_trajectory_vec * ball_trajectory_vec, axis=2)
    t = np.clip(scalar_projection, a_min=0, a_max=1)
    closest_points = ball_start_loc + t[:, :, np.newaxis] * ball_trajectory_vec

    return closest_points


def get_player_time_to_k(pos_k: np.ndarray, pos_0: np.ndarray, vel_0: np.ndarray, max_velocity_players=7.8, max_acceleration_players=3.0) -> float:
    """
    Calculate the time it takes for a player to reach a certain position (k_pos) on the pitch.

    Args:
        pos_k (np.ndarray): Position of point k on the pitch.
        pos_0 (np.ndarray): Initial position of the player.
        vel_0 (np.ndarray): Initial velocity vector of the player.
        maximal_velocity_players (float, optional): Maximum velocity of the player. Defaults to 7.8.
        maximal_acceleration_players (float, optional): Maximum acceleration of the player. Defaults to 3.0.

    Returns:
        float: Time it takes for the player to reach point k.
    """
    if pos_k.ndim == 1:
        pos_k = pos_k[np.newaxis, :]
    
    pos_0 = pos_0[np.newaxis, :, :] 
    if pos_k.ndim == 2:
        pos_k = pos_k[:, np.newaxis, :]
    vel_0 = vel_0[np.newaxis, :, :]

    initial_distance_to_k = np.clip(np.linalg.norm(pos_0 - pos_k, axis=2), a_min=1e-6, a_max=None)
    initial_speed_along_k = np.sum(vel_0 * (pos_k - pos_0), axis=2) / initial_distance_to_k

    time_to_max_vel = np.clip((max_velocity_players - initial_speed_along_k) / max_acceleration_players, a_min=0, a_max=None)
    distance_accel = 0.5 * max_acceleration_players * time_to_max_vel**2
    

    short_idx = np.where(initial_distance_to_k <= distance_accel)
    result = np.zeros(initial_distance_to_k.shape)
    result[short_idx] =  np.sqrt(2 * initial_distance_to_k[short_idx] / max_acceleration_players)  
    long_idx = np.where(initial_distance_to_k > distance_accel)
    result[long_idx] = time_to_max_vel[long_idx] + (initial_distance_to_k[long_idx] - distance_accel[long_idx]) / max_velocity_players

    return result

    
def sigmoid(x:np.ndarray, a:float = 1, b:float = 1, c:float = 0, d:float = 0) -> np.ndarray:
    """Sigmoid function in the shape of a / (1 + e^(-b * (x - c)) ) + d. The defaults 
    are initialized so that the the default function is 1 / (1 + e^(-x)).

    Args:
        x (np.ndarray): Input values.
        a (float, optional): Formula constant. Defaults to 1.
        b (float, optional): Formula constant. Defaults to 1.
        c (float, optional): Formula constant. Defaults to 0.
        d (float, optional): Formula constant. Defaults to 0.

    Returns:
        np.ndarray: Values after passing through the sigmoid. Returned array is
            the same shape as the input shape of x
    """
    return a / (1 + np.exp(-b * (x - c))) + d

def get_interception_probability(ball_time_to_k:float, player_time_to_k) -> float:
    """
    Calculate the interception probability of a ball by a player.

    Args:
        ball_time_to_k (float): Time for the ball to reach the player (seconds).
        player_time_to_k (float): Time for the player to reach the interception point (seconds).

    Returns:
        float: Probability of interception, ranging from 0 to 1.

    Notes:
        - This function calculates the probability of a player intercepting a ball based on the difference
          between the time for the ball to reach the player and the time for the player to reach the interception point.
        - The probability is calculated using the sigmoid function with parameters (1, 5, -0.25, 0), which is defined as:
            sigmoid(x) = 1 / (1 + e^(-5 * (x + 0.25)) ) + 0
        - The resulting probability ranges from 0 (low probability of interception) to 1 (high probability of interception).
    """
    if ball_time_to_k.ndim == 1:
        ball_time_to_k = ball_time_to_k[:, np.newaxis]
    return sigmoid(ball_time_to_k-player_time_to_k, 1, 5, -0.25, 0)

def get_pass_through_air_ratio(pass_distance: float, a: float = -1., b: float = -0.344, c: float = 29.692, d: float = 1.045) -> float:
    """
    Calculate the pass-through-air ratio based on the pass distance.

    Args:
        pass_distance (float): Distance covered by the pass (in meters).
        a (float, optional): Parameter controlling the shape of the sigmoid curve. Defaults to -1.0.
        b (float, optional): Parameter controlling the steepness of the sigmoid curve. Defaults to -0.344.
        c (float, optional): Parameter shifting the curve along the x-axis. Defaults to 29.692.
        d (float, optional): Parameter shifting the curve along the y-axis. Defaults to 1.045.

    Returns:
        float: Pass-through-air ratio, ranging from 0 to 1.

    Notes:
        - This function calculates the pass-through-air ratio based on the sigmoid function applied to the pass distance.
        - The sigmoid function is defined as:
            sigmoid(x) = 1 / (1 + exp(-(a * (x - c) - b * (x - c))))
        - The resulting pass-through-air ratio ranges from 0 (low ratio, indicating more airtime) to 1 (high ratio, indicating less airtime).
        - The default parameters (a, b, c, d) control the shape and positioning of the sigmoid curve, but can be adjusted as needed.
    """
    return np.clip(sigmoid(pass_distance, a, b, c, d), a_min=0.0, a_max=1.0)


def get_attacker_claim_probability(attacker_vals: np.ndarray, pass_distance: float, pass_end_loc: np.ndarray, max_velocity_players:float=7.8, max_acceleration_players:float=3.0) -> tuple:
    """
    Calculate the claim probability for the attacking team's player to claim a pass at its end location.

    Args:
        attacker_vals (np.ndarray): For every attacker the x, y, vx, vy.
        pass_distance (float): Distance covered by the pass (in meters).
        pass_end_loc (np.ndarray): Location where the pass ends (2D array of x and y coordinates).
        max_velocity_players (float, optional): Maximal velocity of players in m/s. Defaults to 7.8.
        max_acceleration_players (float, optional): Maximal acceleration of palyers in m/s^2. Defaults to 3.0.

    Returns:
        tuple: A tuple containing the claim probability, the optimal initial ball velocity, and the
            ball time till to reach the end location of the pass.

    Notes:
        - This function calculates the claim probability for the attacking team's player to intercept a pass.
        - It considers the positions and velocities of the attacking team's players to compute the claim probability.
        - The claim probability is computed based on the shortest time for an attacker to reach the pass end location.
        - The optimal initial ball velocity is computed based on the pass distance and the shortest time for an 
            attacker to reach the end location.
        - The function returns a tuple containing the claim probability and the optimal initial ball velocity.
    """
 
    if len(attacker_vals) == 0:
        return np.nan, np.nan, np.nan

    player_time_to_end = get_player_time_to_k(
        pos_k=pass_end_loc.astype(np.float64),
        pos_0=attacker_vals[:, :2],
        vel_0=attacker_vals[:, 2:],
        max_velocity_players=max_velocity_players,
        max_acceleration_players=max_acceleration_players
    )

    optimal_initial_ball_velocity = get_initial_ball_velocity(
        pass_distance,
        np.min(player_time_to_end, axis=1)
    )

    # ball time to k may be different from player time to k if min or max of initial ball velocity is reached
    to_calc = np.where(np.logical_or(optimal_initial_ball_velocity == 25, optimal_initial_ball_velocity == 8.0))[0]
    ball_time_to_end = np.min(player_time_to_end, axis=1)
    ball_time_to_end[to_calc] = get_ball_time_to_k(optimal_initial_ball_velocity[to_calc], pass_distance[to_calc])[:, 0]
    
    attacker_claim_probabilities = get_interception_probability(
        ball_time_to_k=ball_time_to_end,
        player_time_to_k=player_time_to_end
    )
    
    # Take the claim probability of the two most likely attackers
    claim_probability = 1 - np.prod(1 - attacker_claim_probabilities, axis=1)

    return claim_probability, optimal_initial_ball_velocity, ball_time_to_end

def get_defender_claim_and_intercept_probability(defender_vals: np.ndarray, pass_start_loc: np.ndarray, pass_end_loc: np.ndarray, initial_ball_velocity:float, ball_time_to_end:float, max_velocity_players:float=7.8, max_acceleration_players:float=3.0) -> tuple:
    """
    Calculate defender claim and intercept probabilities for a pass for the defenders.

    Args:
        defender_vals (np.ndarray): for every defender the x, y, vx, vy
        pass_start_loc (np.ndarray): Starting location of the pass as a NumPy array.
        pass_end_loc (np.ndarray): Ending location of the pass as a NumPy array.
        initial_ball_velocity (float): Initial velocity of the ball (in meters/second).
        ball_time_to_end (float): Time taken for the ball to reach the end location of the pass (in seconds).
        max_velocity_players (float, optional): Maximal velocity of players in m/s. Defaults to 7.8.
        max_acceleration_players (float, optional): Maximal acceleration of palyers in m/s^2. Defaults to 3.0.

    Returns:
        tuple: A tuple containing claim probability and intercept probability.

    Notes:
        - This function calculates the probabilities of defending players claiming and intercepting a pass.
        - It uses player positions and velocities at the current frame along with pass start and end locations.
        - The defender claim probability is computed based on the time for each defender to reach the end location of the pass.
        - The defender intercept probability is computed based on the time for each defender to intercept the ball trajectory between pass start and end locations.
        - The function returns a tuple containing the claim probability and intercept probability.
        - If no defenders are present, NaN values are returned for both probabilities.
    """

    if len(defender_vals) == 0:
        return np.nan, np.nan

   

    
    player_time_to_end = get_player_time_to_k(
        pos_k=pass_end_loc,
        pos_0=defender_vals[:, :2],
        vel_0=defender_vals[:, 2:],
        max_velocity_players=max_velocity_players,
        max_acceleration_players=max_acceleration_players
    )
    defender_claim_probabilities = get_interception_probability(
        ball_time_to_k=ball_time_to_end,
        player_time_to_k=player_time_to_end
    )

    # k_positions = get_loc_shortest_to_ball_trajectory(pass_start_loc, pass_end_loc, defender_vals[:, :2])

    # player_time_to_k = get_player_time_to_k(
    #     pos_k=k_positions,
    #     pos_0=defender_vals[:, :2],
    #     vel_0=defender_vals[:, 2:],
    #     max_velocity_players=max_velocity_players,
    #     max_acceleration_players=max_acceleration_players
    # )    
    
    # ball_time_to_k = get_ball_time_to_k(initial_ball_velocity, np.linalg.norm(k_positions - pass_start_loc, axis=2))

    # defender_intercept_probabilities = get_interception_probability(
    #     ball_time_to_k=ball_time_to_k,
    #     player_time_to_k=player_time_to_k
    # )
    
    
    # final probability
    claim_probability = 1 - np.prod(1 - defender_claim_probabilities, axis=1)
    #  intercept_probability = 1 - np.prod(1 - defender_intercept_probabilities, axis=1)
    return claim_probability, None

def get_pass_success_probability(frame:pd.Series, pass_start_loc:np.ndarray, pass_end_loc:np.ndarray, defending_team:str, max_velocity_players:float=7.8, max_acceleration_players:float=3.0, attacker_vals:np.ndarray=None, defender_vals:np.ndarray=None) -> float:
    """
    Calculate the probability of a successful pass.

    Args:
        frame (pd.Series): DataFrame row representing the current frame.
        pass_start_loc (np.ndarray): Starting location of the pass as a NumPy array.
        pass_end_loc (np.ndarray): Ending location of the pass as a NumPy array.
        defending_team (str): The team defending against the pass ('home' or 'away').
        max_velocity_players (float, optional): Maximal velocity of players in m/s. Defaults to 7.8.
        max_acceleration_players (float, optional): Maximal acceleration of palyers in m/s^2. Defaults to 3.0.
        attacker_vals (np.ndarray, optional): For every attacker the x, y, vx, vy. Defaults to None.
        defender_vals (np.ndarray, optional): For every defender the x, y, vx, vy. Defaults to None.

    Returns:
        float: The probability of a successful pass.

    Notes:
        - This function calculates the probability of a successful pass based on various factors.
        - It considers the positions of both attacking and defending players, along with pass start and end locations.
        - The defending team's player positions are determined based on the given team label.
        - The function calculates claim and intercept probabilities for both attacking and defending players.
        - It also computes the probability of a high pass through the air, which affects interception likelihood.
        - The final pass success probability is calculated based on a combination of interception probability and player claims.
    """
    if pass_end_loc.ndim == 1:
        pass_end_loc = pass_end_loc[np.newaxis, :]

    if pass_start_loc.dtype != np.float64:
        pass_start_loc = pass_start_loc.astype(np.float64)
    if pass_end_loc.dtype != np.float64:
        pass_end_loc = pass_end_loc.astype(np.float64)

    if attacker_vals is None or defender_vals is None:
        attacking_team = "home" if defending_team == "away" else "away"
        defending_column_ids = [x[:-3] for x in frame.index if x.startswith(defending_team) and x.endswith("_vx") and not pd.isnull(frame[x])]
        attacking_column_ids = [x[:-3] for x in frame.index if x.startswith(attacking_team) and x.endswith("_vx") and not pd.isnull(frame[x])]
        attacker_vals = np.array([[frame[f"{column_id}_x"], frame[f"{column_id}_y"], frame[f"{column_id}_vx"], frame[f"{column_id}_vy"]] for column_id in attacking_column_ids])
        defender_vals = np.array([[frame[f"{column_id}_x"], frame[f"{column_id}_y"], frame[f"{column_id}_vx"], frame[f"{column_id}_vy"]] for column_id in defending_column_ids])


    pass_distance = np.linalg.norm(pass_start_loc - pass_end_loc, axis=1)  
    claim_probability_att, initial_ball_velocity, ball_time_to_end = get_attacker_claim_probability(
        attacker_vals=attacker_vals, 
        pass_distance=pass_distance, 
        pass_end_loc=pass_end_loc,
        max_velocity_players=max_velocity_players,
        max_acceleration_players=max_acceleration_players,
    )

    claim_probability_def, defender_intercept_probability = get_defender_claim_and_intercept_probability(
        defender_vals, 
        pass_start_loc, 
        pass_end_loc, 
        initial_ball_velocity, 
        ball_time_to_end,
        max_velocity_players=max_velocity_players,
        max_acceleration_players=max_acceleration_players,
    )

    attacker_wins_claim_probability = sigmoid(claim_probability_att - claim_probability_def, 1, 4, 0, 0)
    if len(attacker_wins_claim_probability) == 1:
        return attacker_wins_claim_probability[0]
    return attacker_wins_claim_probability
    

def get_pass_success_probability_grid(frame:pd.Series, pass_start_loc:np.ndarray, defending_team:str,  grid_shape:tuple=(64, 48)) -> np.ndarray:
    """
    Calculate the pass success probability grid over a specified grid shape.

    Args:
        frame (pd.Series): DataFrame row representing the current frame.
        pass_start_loc (np.ndarray): Starting location of the pass as a NumPy array.
        defending_team (str): The team defending against the pass ('home' or 'away').
        grid_shape (tuple, optional): Shape of the grid for calculating pass success probabilities. Defaults to (64, 48).

    Returns:
        np.ndarray: Grid of pass success probabilities over the specified grid shape.

    Notes:
        - This function calculates a grid of pass success probabilities over a specified grid shape.
        - The grid covers a rectangular area defined by the provided grid shape.
        - Pass success probabilities are computed for each grid cell based on the given frame and pass start location.
        - The defending team's positions are determined based on the given team label.
        - The resulting grid provides a spatial distribution of pass success probabilities across the defined area.
    """
    attacking_team = "home" if defending_team == "away" else "away"
    defending_column_ids = [x[:-3] for x in frame.index if x.startswith(defending_team) and x.endswith("_vx") and not pd.isnull(frame[x])]
    attacking_column_ids = [x[:-3] for x in frame.index if x.startswith(attacking_team) and x.endswith("_vx") and not pd.isnull(frame[x])]
    attacker_vals = np.array([[frame[f"{column_id}_x"], frame[f"{column_id}_y"], frame[f"{column_id}_vx"], frame[f"{column_id}_vy"]] for column_id in attacking_column_ids])
    defender_vals = np.array([[frame[f"{column_id}_x"], frame[f"{column_id}_y"], frame[f"{column_id}_vx"], frame[f"{column_id}_vy"]] for column_id in defending_column_ids])

    x_ranges = np.linspace(-53, 53, grid_shape[0] + 1)
    y_ranges = np.linspace(34, -34, grid_shape[1] + 1)

    x_coords = (x_ranges[:-1] + x_ranges[1:]) / 2
    y_coords = (y_ranges[:-1] + y_ranges[1:]) / 2

    grid = np.zeros(grid_shape)
    grid = grid.ravel()
    end_locs = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

    result = get_pass_success_probability(
        frame, 
        pass_start_loc,
        end_locs,
        defending_team,
        attacker_vals=attacker_vals,
        defender_vals=defender_vals,
    )
   
    return result.reshape(grid_shape)

if __name__ == "__main__":
    import line_profiler
    from databallpy.features import get_velocity, get_individual_player_possessions_and_duels
    from databallpy import get_match
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from databallpy.visualize import plot_soccer_pitch

    td = "data/20230226 Fortuna Sittard vs Feyenoord 2291929.dat"
    td_md = "data/20230226 Fortuna Sittard vs Feyenoord 2291929 metadata.xml"
    ed = "data/20230226 Fortuna Sittard vs Feyenoord 2291929 f24.xml"
    ed_md = "data/20230226 Fortuna Sittard vs Feyenoord 2291929 f7.xml"
    
    def plot_frame(frame:pd.Series, pass_start_loc:float, pass_end_loc:float, defending_team:str, grid=None):
        """
        Plot a frame of soccer match with player positions and a pass between two locations.

        Args:
            frame (pd.Series): DataFrame row representing the current frame.
            pass_start_loc (np.ndarray): Starting location of the pass as a NumPy array.
            pass_end_loc (np.ndarray): Ending location of the pass as a NumPy array.
            defending_team (str): The team defending against the pass ('home' or 'away').
            grid (np.ndarray, optional): Grid of pass success probabilities over the pitch. Defaults to None.

        Returns:
            tuple: Figure and Axes objects of the plotted frame.

        Notes:
            - This function plots a single frame of a soccer match with player positions and a pass between two locations.
            - The DataFrame row `frame` should contain information about player positions and velocities.
            - `pass_start_loc` and `pass_end_loc` represent the starting and ending locations of the pass, respectively.
            - Player positions are plotted as scatter points, with different colors representing the home and away teams.
            - Player velocities are visualized as arrows indicating direction and speed.
            - If provided, the `grid` parameter allows overlaying a grid of pass success probabilities on the pitch.
            - Pass success probability grid is visualized using a colormap and a color bar is added to the plot.
        """
        fig, ax = plot_soccer_pitch(pitch_color="white", field_dimen=(106, 68))
        ax.scatter(frame["ball_x"], frame["ball_y"], color="black")
        arrow = mpatches.FancyArrowPatch(
            pass_start_loc,
            pass_end_loc,
            mutation_scale=10,
            color="black",
        )
        ax.add_patch(arrow)

        home_column_ids = [x[:-2] for x in frame.index if x.startswith("home") and x.endswith("_x")]
        away_column_ids = [x[:-2] for x in frame.index if x.startswith("away") and x.endswith("_x")]
        x_home = frame[[x + "_x" for x in home_column_ids]].values
        y_home = frame[[y + "_y" for y in home_column_ids]].values
        vx_home = frame[[x + "_vx" for x in home_column_ids]].values
        vy_home = frame[[y + "_vy" for y in home_column_ids]].values
        x_away = frame[[x + "_x" for x in away_column_ids]].values
        y_away = frame[[y + "_y" for y in away_column_ids]].values
        vx_away = frame[[x + "_vx" for x in away_column_ids]].values
        vy_away = frame[[y + "_vy" for y in away_column_ids]].values
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

        if grid is not None:
            imshow = ax.imshow(grid.T, extent=[-53, 53, -34, 34], cmap="coolwarm", alpha=0.7, vmin=0, vmax=1, origin='upper', aspect='auto')
            plt.colorbar(imshow, label='Pass success probability')
        return fig, ax
    
    match = get_match(
        tracking_data_loc=td,
        tracking_metadata_loc=td_md,
        tracking_data_provider="tracab",
        event_data_loc=ed,
        event_metadata_loc=ed_md,
        event_data_provider="opta",
    )

    match.tracking_data = get_velocity(match.tracking_data, match.home_players_column_ids() + match.away_players_column_ids(), match.frame_rate)
    match.synchronise_tracking_and_event_data()

    match.tracking_data["player_possession"], _ = get_individual_player_possessions_and_duels(match.tracking_data, match.frame_rate)
    # match.add_tracking_data_features_to_passes()

    event_id = list(match.pass_events.keys())[0]
    event = match.pass_events[event_id]
    frame = match.tracking_data.loc[match.tracking_data["event_id"]==event_id].iloc[0]
    defending_team = "home" if event.team_side == "away" else "away"
    pass_start_loc = np.array([event.start_x, event.start_y])
    pass_end_loc = np.array([event.end_x, event.end_y])

    pass_end_loc = np.array([-10., 2.])

    attacking_team = "home" if defending_team == "away" else "away"
    defending_column_ids = [x[:-3] for x in frame.index if x.startswith(defending_team) and x.endswith("_vx") and not pd.isnull(frame[x])]
    attacking_column_ids = [x[:-3] for x in frame.index if x.startswith(attacking_team) and x.endswith("_vx") and not pd.isnull(frame[x])]
    attacker_vals = np.array([[frame[f"{column_id}_x"], frame[f"{column_id}_y"], frame[f"{column_id}_vx"], frame[f"{column_id}_vy"]] for column_id in attacking_column_ids])
    defender_vals = np.array([[frame[f"{column_id}_x"], frame[f"{column_id}_y"], frame[f"{column_id}_vx"], frame[f"{column_id}_vy"]] for column_id in defending_column_ids])
    
    
    # xP = get_pass_success_probability(
    #     frame,
    #     pass_start_loc,
    #     pass_end_loc,
    #     defending_team = "home" if event.team_side == "away" else "away",
    # )
    # # xP = 1000
    # grid = get_pass_success_probability_grid(frame, pass_start_loc, "home" if event.team_side == "away" else "away")
    # fig, ax = plot_frame(frame, pass_start_loc, pass_end_loc, "home" if event.team_side == "away" else "away", grid=grid )
    # ax.set_title(f"xP: {xP:.2f}, outcome: {event.outcome}")
    # plt.show()
    lp = line_profiler.LineProfiler()
    lp_wrapper = lp(get_pass_success_probability_grid)
    lp_wrapper(
        frame,
        pass_start_loc,
        defending_team,
    )

    lp.print_stats()
    # import pdb; pdb.set_trace()
