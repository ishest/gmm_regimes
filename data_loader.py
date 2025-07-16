import pandas as pd

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        df = pd.read_csv(self.filepath, parse_dates=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        self.df = df
        return df
