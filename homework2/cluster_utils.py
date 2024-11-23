import plotly.graph_objects as go
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def visualize_clusters(embedding, labels, cluster_labels, title, n_components=3):
    """
    Visualize embeddings with original labels and cluster labels.

    Parameters:
        embedding (np.ndarray): The data embeddings to visualize (2D or 3D array for plotting).
        labels (list or np.ndarray): The original labels for the data points.
        cluster_labels (list or np.ndarray): Cluster assignments for the data points.
        title (str): Title of the plot.
        n_components (int): Number of components to visualize (2 or 3).
    """
    fig = go.Figure()

    if n_components == 3:
        # 3D Plot
        fig.add_trace(
            go.Scatter3d(
                x=embedding[:, 0],
                y=embedding[:, 1],
                z=embedding[:, 2],
                mode="markers",
                marker=dict(
                    size=6,
                    opacity=0.8,
                    color=cluster_labels,
                    colorscale="Viridis",
                    colorbar=dict(title="Cluster Label"),
                ),
                text=labels,
                hoverinfo="text",
                name=title,
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="PCA Component 1"),
                yaxis=dict(title="PCA Component 2"),
                zaxis=dict(title="PCA Component 3"),
            )
        )
    elif n_components == 2:
        # 2D Plot
        fig.add_trace(
            go.Scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode="markers",
                marker=dict(
                    size=6,
                    opacity=0.8,
                    color=cluster_labels,
                    colorscale="Viridis",
                    colorbar=dict(title="Cluster Label"),
                ),
                text=labels,
                hoverinfo="text",
                name=title,
            )
        )
        fig.update_layout(
            xaxis=dict(title="PCA Component 1"),
            yaxis=dict(title="PCA Component 2"),
        )
    else:
        raise ValueError("n_components must be 2 or 3.")

    # General layout updates
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(title="Cluster Assignments"),
    )

    fig.show()


def compare_clusters(labels, kmeans_labels, hierarchical_labels):
    ari_kmeans = adjusted_rand_score(labels, kmeans_labels)
    nmi_kmeans = normalized_mutual_info_score(labels, kmeans_labels)
    ari_hierarchical = adjusted_rand_score(labels, hierarchical_labels)
    nmi_hierarchical = normalized_mutual_info_score(labels, hierarchical_labels)

    print(f"K-Means Clustering - ARI: {ari_kmeans:.4f}, NMI: {nmi_kmeans:.4f}")
    print(
        f"Hierarchical Clustering - ARI: {ari_hierarchical:.4f}, NMI: {nmi_hierarchical:.4f}"
    )
