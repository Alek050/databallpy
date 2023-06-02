import pandas as pd


def get_velocity(
    df: pd.DataFrame, input_columns: list, framerate: float
) -> pd.DataFrame:
    """Function that adds velocity columns based on the position columns

    Args:
        df (pd.DataFrame): tracking data
        input_columns (list): columns for which velocity should be calculated
        framerate (float): framerate of the tracking data

    Returns:
        pd.DataFrame: tracking data with the added velocity columns
    """
    # for input_column in input_columns:
    #     output_column = input_column + "_v"
    #     v = df[input_column].diff() / (1 / framerate)
    #     df[output_column] = v

    velocities = []
    for input_column in input_columns:
        v = df[input_column].diff() / (1 / framerate)
        velocities.append(v)

    velocity_df = pd.concat(velocities, axis=1)
    velocity_df.columns = [f"{col}_v" for col in input_columns]
    df = pd.concat([df, velocity_df], axis=1)

    return df

    return df
