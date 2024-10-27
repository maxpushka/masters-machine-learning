import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score


class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        learning_rate=0.01,
        regularization=0.01,
        max_iter=1000,
        tol=1e-4,
        early_stopping=True,
        patience=10,
        verbose=False,
    ):
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose
        self.scaler = StandardScaler()  # Standardize features

    def sigmoid(self, z):
        return expit(z)

    def compute_cost(self, X, y, weights):
        m = len(y)
        h = self.sigmoid(np.dot(X, weights))

        # Add epsilon to prevent log(0) and maintain numerical stability
        epsilon = 1e-15
        h = np.clip(
            h, epsilon, 1 - epsilon
        )  # restrict h values to [epsilon, 1 - epsilon]

        cost = -1 / m * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h))) \
            + (self.regularization / (2 * m)) * np.sum(np.square(weights[1:]))

        return cost

    def compute_gradient(self, X, y, weights):
        m = len(y)
        h = self.sigmoid(np.dot(X, weights))

        # Compute the gradient for the logistic regression cost function
        error = h - y
        gradient = (1 / m) * np.dot(X.T, error)

        # Apply L2 regularization, excluding the intercept term
        regularization_term = (self.regularization / m) * weights
        regularization_term[0] = 0  # No regularization for the intercept term
        gradient += regularization_term

        return gradient

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # for compatibility with scikit-learn estimators
        X = self.scaler.fit_transform(X)
        X = np.c_[np.ones((X.shape[0], 1)), X]  # add intercept term
        self.weights = np.zeros(X.shape[1])
        best_cost = np.inf
        no_improve = 0

        for i in range(self.max_iter):
            gradient = self.compute_gradient(X, y, self.weights)

            # Update weights
            self.weights -= self.learning_rate * gradient
            cost = self.compute_cost(X, y, self.weights)

            if self.verbose:
                print(f"Iteration {i + 1}: Cost {cost:.4f}")

            if not self.early_stopping: continue
            if cost < best_cost - self.tol:
                best_cost = cost
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                if self.verbose:
                    print("Early stopping triggered.")
                break

    def predict_proba(self, X):
        X = self.scaler.transform(X)  # standardize features
        X = np.c_[np.ones((X.shape[0], 1)), X]  # add intercept term
        return self.sigmoid(np.dot(X, self.weights))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


if __name__ == "__main__":  # simple test driver
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

    model = LogisticRegression(
        learning_rate=0.1,
        regularization=0.1,
        max_iter=1000,
        tol=1e-4,
        early_stopping=True,
        patience=5,
        verbose=True,
    )
    model.fit(X, y)

    print("Training accuracy:", model.score(X, y))
