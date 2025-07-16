import numpy as np
from sklearn.mixture import GaussianMixture

class RegimeModeler:
    def __init__(self, X, max_regimes=6, random_state=42):
        self.X = X
        self.max_regimes = max_regimes
        self.random_state = random_state
        self.bic_scores = []
        self.models = {}
        self.selected_k = None
        self.gmm = None

    def fit_gmm_range(self):
        lowest_bic = np.infty
        bic = []
        n_components_range = range(1, self.max_regimes + 1)
        for n_components in n_components_range:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=self.random_state
            ).fit(self.X)
            bic.append(gmm.bic(self.X))
            self.models[n_components] = gmm
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                self.selected_k = n_components
        self.bic_scores = bic
        self.gmm = self.models[self.selected_k]
        return self.gmm, self.selected_k, self.bic_scores

    def assign_regimes(self):
        labels = self.gmm.predict(self.X)
        probs = self.gmm.predict_proba(self.X)
        return labels, probs

    def get_model_params(self):
        return {
            "means": self.gmm.means_,
            "covariances": self.gmm.covariances_,
            "weights": self.gmm.weights_
        }
