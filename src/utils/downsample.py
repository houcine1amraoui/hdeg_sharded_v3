import pandas as pd

import pandas as pd

def detect_sensor_types(df):
    """
    Automatically detect binary and continuous sensors in a CSV time-series dataset.
    """
def detect_sensor_types(df):

    df_temp = df.copy()

    binary_cols = []
    continuous_cols = []

    for col in df_temp.columns:

        unique_vals = df_temp[col].dropna().unique()

        if set(unique_vals).issubset({0, 1}):
            binary_cols.append(col)
        else:
            continuous_cols.append(col)

    return binary_cols, continuous_cols