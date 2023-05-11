import pandas as pd

def get_velocity(df: pd.DataFrame, input_columns: list, framerate:float, filter: str = "")-> pd.DataFrame:
    for input_column in input_columns:
        output_column = input_column + "_v"
        v = df[input_column].diff()/(1/framerate)
        df[output_column] = v
    
    if filter == "":
        pass

    return df