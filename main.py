import sys
import os

sys.path.append(os.path.dirname(__file__))

from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from regime_modeler import RegimeModeler
from visualizer import Visualizer

def main():
    # 1. Load Data
    data_path = "../data/spy_mwf.csv"
    dl = DataLoader(data_path)
    df = dl.load_data()

    # 2. Feature Engineering
    fe = FeatureEngineer(df)
    df_features = fe.compute_features()
    X_scaled, scaler = fe.scale_features()

    # 3. GMM Model Fitting & Selection
    rm = RegimeModeler(X_scaled, max_regimes=6)
    gmm, best_k, bic_scores = rm.fit_gmm_range()
    labels, probs = rm.assign_regimes()
    params = rm.get_model_params()

    # 4. Visualizations
    Visualizer.plot_bic(bic_scores, max_regimes=6)
    Visualizer.plot_scatter(df_features, labels)
    Visualizer.plot_regime_probabilities(df_features, probs)

    # 5. Attach labels to DataFrame and save results
    df_features['GMM_Regime'] = labels
    df_features.to_csv("../data/spy_mwf_regimes.csv", index=False)
    print("Best K (regimes):", best_k)
    print("Model Params:", params)

if __name__ == "__main__":
    main()
