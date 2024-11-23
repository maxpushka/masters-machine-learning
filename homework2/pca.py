import numpy as np

class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the PCA model to the dataset by calculating the principal components.
        """
        # Standardize the data
        X = self.standardize(X)

        # Compute the covariance matrix
        covariance_matrix = np.cov(X, rowvar=False)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and corresponding eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the dataset using the fitted PCA model.
        """
        # Standardize the data
        X = self.standardize(X)

        # Project data onto principal components
        return np.dot(X, self.components)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def standardize(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize data by subtracting the mean and dividing by the standard deviation.
        """
        if self.mean is None:
            self.mean = np.mean(X, axis=0)
        std_dev = np.std(X, axis=0)
        # Avoid division by zero for features with zero variance
        std_dev[std_dev == 0] = 1
        return (X - self.mean) / std_dev
