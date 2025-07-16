import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()

    def compute_features(self):
        df = self.df
        df['LogReturn'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
        df['Volatility_30d'] = df['LogReturn'].rolling(window=30).std()
        df = df.dropna().reset_index(drop=True)
        self.df = df
        return df

    def scale_features(self, features=['LogReturn', 'Volatility_30d']):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df[features].values)
        return X_scaled, scaler
