import re
import numpy as np
import pandas as pd

def build_ndarray(
        data: pd.Series, 
        pattern: re.Pattern = None, 
        type: str = "ball"
        ) -> np.ndarray:
    """
    Takes data and puts it in certain form for later usage

    Parameters
    ------------
    data: pd.Series
    Series object representing a single frame of the tracking data

    pattern: re.Pattern
    Pattern object necessary for identifying the correct columns to put into the needed form

    type: str
    String object necessary to differentiate what type of data is supposed to be transformed

    
    Output
    ----------
    output: np.ndarray
    Ndarray with the data reformed in the necessary shape

    """


    output = []

    groups = {}

    if type == "ball" or pattern == None:
        pattern = re.compile(rf"ball_(x|y)")
        x_val = None
        y_val = None

        for col in data.index:
            match = pattern.match(col)
            if match:
                if match.group(1) == "x":
                    x_val = data[col]
                else:
                    y_val = data[col]

                if x_val is not None and y_val is not None:
                    return np.asarray([[x_val, y_val]])


    for col in data.index:
        match = pattern.match(col)
        if match:
            idx = int(match.group(1))
            xy = match.group(2)

            if idx not in groups:
                groups[idx] = {}

            groups[idx][xy] = data[col]

    if type == "position":
        for idx in sorted(groups.keys()):
            x_vals = groups[idx].get("x")
            y_vals = groups[idx].get("y")
            if np.isnan(x_vals):
                continue
            output.append((x_vals, y_vals))
    
    elif type == "velocity":
        for idx in sorted(groups.keys()):
            x_vals = groups[idx].get("vx")
            y_vals = groups[idx].get("vy")
            if np.isnan(x_vals):
                continue
            output.append((x_vals, y_vals))


    return np.asarray(output)

def prepare_data(
        data: pd.Series
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Function to prepare all needed data from a single frame

    Parameters
    ------------
    data: pd.Series
    Series object representing a single frame in the tracking data


    Output
    ---------
    home_pos: np.ndarray
    Positions of the home players in the form of an array with the dimensions (n, 2)

    away_pos: np.ndarray
    Positions of the away players in the form of an array with the dimensions (n, 2)

    home_vel: np.ndarray
    Velocities of the home players in the form of an array with the dimensions (n, 2)

    away_vel: np.ndarray
    Velocities of the away players in the form of an array with the dimensions (n, 2)
    """

    ball_pos = build_ndarray(data)

    pattern_home_pos = re.compile(rf"home_(\d+)_(x|y)$")
    home_pos = build_ndarray(data, pattern_home_pos, "position")

    pattern_away_pos = re.compile(rf"away_(\d+)_(x|y)$")
    away_pos = build_ndarray(data, pattern_away_pos, "position")

    pattern_home_vel = re.compile(rf"home_(\d+)_(vx|vy)$")
    home_vel = build_ndarray(data, pattern_home_vel, "velocity")
    
    pattern_away_vel = re.compile(rf"away_(\d+)_(vx|vy)$")
    away_vel = build_ndarray(data, pattern_away_vel, "velocity")

    return home_pos, away_pos, home_vel, away_vel, ball_pos

def time_to_intercept(
    p_players: np.ndarray,
    p_ball: np.ndarray,
    v_players: np.ndarray,
    reaction_time: float,
    max_velocity: float
) -> np.ndarray:
    """
    Calculates time needed for players to reach the location of the ball. Taken from unravel sports, then altered.

    TODO: check math, finetune for realistic values

    Parameters
    ----------
    p_players : ndarray
        An array of shape (n, 2) representing the positions of players.
        Each row corresponds to a player's position as (x, y) coordinates.

    p_ball : ndarray
        An array of shape (1, 2) representing the positions of the ball

    v_players : ndarray
        An array of shape (n, 2) representing the velocities corresponding to p_players. Each row corresponds
        to a player's velocity as (vx, vy).

    reaction_time : float
        The reaction time of players (in seconds) before they start moving towards the ball.
        Default = 0.21 
        Şenel, Özgür & Eroğlu, Hüseyin. (2006). Correlation between reaction time and speed in elite soccer players. Journal of Exercise Science and Fitness. 4. 126-130. 

    max_velocity : float
        The maximum running velocity of players (in meters per second).
        Default: 32.75km/h => 9.1m/s
        Estimated by looking at the data from https://www.bundesliga.com/en/bundesliga/stats/players/top-speed

    Returns
    -------
    t : ndarray
        A 2D array of shape (n, 1) where t[j] represents the time required for Player[j]
        to get to the ball.
    """

    v = (
        p_ball[:, None, :] - p_players[None, :, :]
    )  # Relative motion vector between Pressing Players and Players Under Pressure

    u_mag = np.linalg.norm(p_players, axis=-1)  # velocitie of Pressing Players velocity
    v_mag = np.linalg.norm(v, axis=-1)  # velocitie of relative motion vector
    dot_product = np.sum(p_players * v, axis=-1)

    epsilon = 1e-10  # We add epsilon to avoid dividing by zero (which throws a warning)
    angle = np.arccos(dot_product / (u_mag * v_mag + epsilon))

    r_reaction = (
        p_players + v_players * reaction_time
    )  # Adjusted position of Pressing Players after reaction time
    d = p_ball[:, None, :] - r_reaction[None, :, :]  # Distance vector after reaction time

    t = (
        u_mag * angle / np.pi  # Time contribution from angular adjustment
        + reaction_time  # Add reaction time
        + np.linalg.norm(d, axis=-1) / max_velocity
    )  # Time contribution from running

    return t

def compute_team_tti(
        home_player_position: np.ndarray,
        home_player_velocity: np.ndarray,
        away_player_position: np.ndarray,
        away_player_velocity: np.ndarray,
        ball_position: np.ndarray,
        reaction_time: float = 0.21,
        max_velocity: float = 9.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the time to intercept the ball for each player on both teams

    Parameters
    -------------
    home_player_position : ndarray
        An array of shape (n, 2) representing the positions of the home players.
        Each row corresponds to a player's position as (x, y) coordinates.

    home_player_velocity : ndarray
        An array of shape (n, 2) representing the velocities corresponding to home_player_position. Each row corresponds
        to a player's velocity as (vx, vy).

    away_player_position : ndarray
        An array of shape (n, 2) representing the positions of the away players.
        Each row corresponds to a player's position as (x, y) coordinates.

    away_player_velocity : ndarray
        An array of shape (n, 2) representing the velocities corresponding to away_player_position. Each row corresponds
        to a player's velocity as (vx, vy).

    ball_position : ndarray
        An array of shape (1, 2) representing the positions of the ball

    reaction_time : float
        The reaction time of players (in seconds) before they start moving towards the ball.
        Default = 0.21 
        Şenel, Özgür & Eroğlu, Hüseyin. (2006). Correlation between reaction time and speed in elite soccer players. Journal of Exercise Science and Fitness. 4. 126-130. 

    max_velocity : float
        The maximum running velocity of players (in meters per second).
        Default: 32.75km/h => 9.1m/s
        Estimated by looking at the data from https://www.bundesliga.com/en/bundesliga/stats/players/top-speed

    Returns
    -------
    t : ndarray
        A 2D array of shape (n, 1) where t[j] represents the time required for Player[j]
        to get to the ball.
    """
    
    time_home_players = time_to_intercept(home_player_position, ball_position, home_player_velocity, reaction_time, max_velocity)
    time_away_players = time_to_intercept(away_player_position, ball_position, away_player_velocity, reaction_time, max_velocity)

    return time_home_players, time_away_players


def probability_to_intercept(
    time_to_intercept: np.ndarray,
    tti_sigma: float, 
    tti_time_threshold: float
) -> np.ndarray:
    
    """
    Calculate the probability of a player intercepting the ball depending on his time to intercept and the time threshold.

    Parameters
    -------------
    time_to_intercept: np.ndarray
    Array of size (n,1) indicating the time a player needs to reach a certain point of the field

    tti_sigma: float
    Sigma-coefficient for the probability function suggested in Spearman(2017). 

    tti_threshold: float
    The time at which the ball would be at the position

    Output
    ---------
    p: np.ndarray
    Array holding the probability values for each player to intercept the ball at the given point
    """
    exponent = (
        #-np.pi / np.sqrt(3.0) / tti_sigma * (tti_time_threshold - time_to_intercept)
        -(tti_time_threshold - time_to_intercept)/(np.sqrt(3)*tti_sigma/np.pi)
    )
    # we take the below step to avoid Overflow errors, np.exp does not like values above ~700.
    # exp(25) should already result in p ~ 0.000%
    exponent = np.clip(exponent, -700, 700)
    p = 1 / (1.0 + np.exp(exponent))
    return p

def compute_team_probs(
        home_tti: np.ndarray,
        away_tti: np.ndarray,
        tti_threshold: float,
        tti_sigma:float = 0.45, #as per Spearman 2017
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the probability of the players of both teams intercepting the ball depending on his time to intercept and the time threshold.

    Parameters
    -------------
    home_tti: np.ndarray
    Array of size (n,1) with the tti's for the home players

    away_tti: np.ndarray
    Array of size (n,1) with the tti's for the away players

    tti_threshold: float
    The time at which the ball would be at the position

    tti_sigma: float
    Sigma-coefficient for the probability function suggested in Spearman(2017). 


    Output
    ---------
    prob_home: np.ndarray
    Array holding the probability values for each home player to intercept the ball at the given point

    prob_away: np.ndarray
    Array holding the probability values for each away player to intercept the ball at the given point
    """
    
    prob_home = probability_to_intercept(home_tti, tti_sigma, tti_threshold)
    prob_away = probability_to_intercept(away_tti, tti_sigma, tti_threshold)

    return prob_home, prob_away


def compute_control_prob(
        tti: np.ndarray,
        tti_threshold: np.ndarray,
        tti_lambda: float = 4.3 #as per Spearman(2017),
) -> np.ndarray:
    """
    Computes the probability of controling the ball in the time between getting to a point and the ball arriving

    Parameters
    -------------
    tti: np.ndarray
    The estimated time to intercept for the different players

    tti_threshold: np.ndarray
    The estimated time the ball arrives at the point of intercept

    tti_lambda: float
    Probability parameter, default = 4.3 as suggested in Spearman(2017)

    Output
    --------
    probs: np.ndarray
    Array holding the probabilities for each player being able to control the ball in the given time.
    """
    t_diff = tti_threshold - tti
    t_diff[0][t_diff[0] < 0] = 0
    exponents = -tti_lambda * t_diff

    probs = 1- np.exp(exponents)

    return probs

def team_control_probs(
        tti_home: np.ndarray, 
        tti_away: np.ndarray, 
        tti_threshold: float, 
        tti_lambda: float = 4.3
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the probabilities of controling the ball for all players on both teams in the time between getting to a point and the ball arriving

    Parameters
    -------------
    tti_home: np.ndarray
    The estimated time to intercept for the home players

    tti_away: np.ndarray
    The estimated time to intercept for the away players

    tti_threshold: np.ndarray
    The estimated time the ball arrives at the point of intercept

    tti_lambda: float
    Probability parameter, default = 4.3 as suggested in Spearman(2017)

    Output
    --------
    home: np.ndarray
    Array holding the probabilities for each home player being able to control the ball in the given time.

    away: np.ndarray
    Array holding the probabilities for each away player being able to control the ball in the given time.
    """
    
    home = compute_control_prob(tti_home, tti_threshold, tti_lambda)
    away = compute_control_prob(tti_away, tti_threshold, tti_lambda)

    return home, away

def calculate_local_pitch_control(
        int_prob_home: np.ndarray, 
        int_prob_away: np.ndarray, 
        cont_prob_home: np.ndarray, 
        cont_prob_away: np.ndarray
        ) -> float:
    """
    Calculating the pitch control value for a single point on the field.

    Parameters
    -------------
    int_prob_home: np.ndarray
    Probabilties for the home players being able to intercept the ball.

    int_prob_away: np.ndarray
    Probabilties for the away players being able to intercept the ball.

    cont_prob_home: np.ndarray
    Probabilities for the home players being able to intercept the ball

    cont_prob_away: np.ndarray
    Probabilities for the away players being able to intercept the ball

    Output
    -----------
    pc: float
    Pitch control coefficient for the point of intercept
    """
    prob_away = np.sum(np.multiply(int_prob_away, cont_prob_away))
    prob_home = np.sum(np.multiply(int_prob_home, cont_prob_home))
    return prob_home - prob_away

def calc_time_threshold(
        ball_pos: np.ndarray, 
        pitch_pos: np.ndarray
        ) -> float:
    """
    Calculating the time threshold indicating the estimated arrival time of the ball at the point of intercept.

    Parameters
    --------------
    ball_pos: np.ndarray
    Current position of the ball on the field

    pitch_pos: np.ndarray
    Position where the ball should be passed

    Output
    ----------
    time: float
    Estimated time needed for the ball to get from its current to its needed position
    """
    #TODO: proper implementation, for now just rudimentary calculation with distance/velocity
    
    movement_vector = pitch_pos - ball_pos
    

    movement_distance = np.sqrt(movement_vector[0][0]**2 + movement_vector[0][1]**2)

    time = movement_distance/12 #default passing velocity = 12 m/s

    return time


def get_spearman_pitch_control_single_frame(
    frame: pd.Series,
    pitch_dimensions: list[float, float],
    n_x_bins: int = 106,
    n_y_bins: int = 68,
) -> np.ndarray:
    """
    Calculate the pitch control of the whole field for a single frame

    Parameters
    --------------
    frame: pd.Series
    A single row of the tracking data from a football match holding the players and balls positions and velocities.

    pitch_dimensions: list[float, float]
    The dimensions of the pitch that is played on.

    n_x_bins: int, default = 106
    Number of points calculated along the x-axis.

    n_y_bins: int, default = 68
    Number of points calculated along the y-axis.

    Output
    --------
    control_grid: np.ndarray
    Array holding the pitch control values for every point in a single frame
    """
    grid = np.meshgrid(
        np.linspace(-pitch_dimensions[0] / 2, pitch_dimensions[0] / 2, n_x_bins),
        np.linspace(-pitch_dimensions[1] / 2, pitch_dimensions[1] / 2, n_y_bins),
    )

    home_pos, away_pos, home_vel, away_vel, ball_pos = prepare_data(frame)

    control_grid = []
    grid_shape = grid[0].shape

    for x in range(grid_shape[0]):
        x_grid = []
        for y in range(grid_shape[1]):

            pitch_pos = np.asarray([[grid[0][x][y], grid[1][x][y]]])

            time_threshold = calc_time_threshold(ball_pos, pitch_pos)

            home_tti, away_tti = compute_team_tti(home_pos, home_vel, away_pos, away_vel, pitch_pos)

            int_prob_home, int_prob_away = compute_team_probs(home_tti, away_tti, time_threshold)

            cont_prob_home, cont_prob_away = team_control_probs(home_tti, away_tti, time_threshold)

            local_pitch_control = calculate_local_pitch_control(int_prob_home, int_prob_away, cont_prob_home, cont_prob_away)

            x_grid.append(local_pitch_control)
        
        control_grid.append(x_grid)

    return control_grid #sigmoid(control_grid, d=5)