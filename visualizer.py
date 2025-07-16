import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    @staticmethod
    def plot_bic(bic_scores, max_regimes):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_regimes + 1), bic_scores, marker='o')
        plt.title("BIC Scores for GMM with Different Regimes")
        plt.xlabel("Number of Regimes (Components)")
        plt.ylabel("BIC Score")
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_scatter(df, labels):
        plt.figure(figsize=(8, 6))
        for regime in np.unique(labels):
            idx = labels == regime
            plt.scatter(df.loc[idx, 'Volatility_30d'], df.loc[idx, 'LogReturn'],
                        label=f"Regime {regime}", s=10, alpha=0.5)
        plt.title("Return vs. Volatility by Regime")
        plt.xlabel("30-Day Rolling Volatility")
        plt.ylabel("Log Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_regime_probabilities(df, probs):
        plt.figure(figsize=(14, 4))
        plt.imshow(probs.T, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.title("Regime Posterior Probabilities (Soft Assignments)")
        plt.ylabel("Regime")
        plt.xlabel("Time Index")
        plt.colorbar(label="Probability")
        plt.tight_layout()
        plt.show()
