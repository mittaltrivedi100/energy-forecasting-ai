import pandas as pd

def load_data(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.dropna(inplace=True)
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    return df
