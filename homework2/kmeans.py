import numpy as np
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go


class KMeans:
    def __init__(self, n_clusters: int, max_iterations: int):
        self.n_clusters = n_clusters
        self.max_iter = max_iterations
        self.centroids = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the model to the data by iteratively updating centroids.
        """
        # Validate the number of clusters
        if self.n_clusters <= 0:
            raise ValueError("Number of clusters must be greater than 0.")
        if self.n_clusters > len(X):
            raise ValueError("Number of clusters cannot exceed the number of samples.")

        # Randomly initialize centroids by selecting n_clusters samples from X
        np.random.seed(42)  # For reproducibility
        random_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iter):
            # Assign clusters
            clusters = self.assign_clusters(self.centroids, X)
            # Compute new centroids
            new_centroids = self.compute_means(clusters, X)

            # If centroids do not change, convergence is reached
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster for each sample in X.
        """
        return self.assign_clusters(self.centroids, X)

    def assign_clusters(self, centroids: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Assign each sample in X to the nearest centroid.
        """
        distances = np.zeros((len(X), self.n_clusters))

        for idx, centroid in enumerate(centroids):
            distances[:, idx] = np.linalg.norm(X - centroid, axis=1)

        return np.argmin(distances, axis=1)

    def compute_means(self, clusters: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Compute new centroids as the mean of samples assigned to each cluster.
        """
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))

        for cluster_idx in range(self.n_clusters):
            cluster_points = X[clusters == cluster_idx]
            if len(cluster_points) > 0:
                new_centroids[cluster_idx] = cluster_points.mean(axis=0)
            else:
                # Handle empty clusters by reinitializing to a random point
                new_centroids[cluster_idx] = X[np.random.choice(len(X))]

        return new_centroids

    def euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate the Euclidean distance between two vectors a and b.
        """
        return np.sqrt(np.sum(np.power(a - b, 2)))


def optimal_clusters(data, max_k=10):
    """
    Determines the optimal number of clusters using the Elbow Method (Distortion)
    and the Silhouette Score, then visualizes the results with dual Y-axes.

    Parameters:
        data (np.ndarray): The dataset to cluster.
        max_k (int): The maximum number of clusters to evaluate (default: 10).

    Returns:
        None
    """
    distortions = []
    silhouettes = []
    cluster_range = range(2, max_k + 1)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, max_iterations=300)
        kmeans.fit(data)
        kmeans_labels = kmeans.predict(data)

        # distortion aka elbow method
        distortion = sum(
            np.linalg.norm(data[i] - kmeans.centroids[label]) ** 2
            for i, label in enumerate(kmeans_labels)
        )
        distortions.append(distortion)

        # silhouette score
        silhouette = silhouette_score(data, kmeans_labels)
        silhouettes.append(silhouette)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(cluster_range),
            y=distortions,
            mode="lines+markers",
            name="Distortion (Elbow Method)",
            yaxis="y1",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(cluster_range),
            y=silhouettes,
            mode="lines+markers",
            name="Silhouette Score",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Optimal Number of Clusters",
        xaxis_title="Number of Clusters",
        yaxis=dict(
            title="Distortion (Elbow Method)",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
        ),
        yaxis2=dict(
            title="Silhouette Score",
            titlefont=dict(color="orange"),
            tickfont=dict(color="orange"),
            overlaying="y",
            side="right",
        ),
        legend_title="Metrics",
    )

    fig.show()
