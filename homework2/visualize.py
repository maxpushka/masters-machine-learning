import pandas as pd
import plotly.graph_objects as go


def visualize_all(embeddings_dict, labels, descriptions):
    """
    Visualize original, PCA, and t-SNE embeddings on the same plot.
    """
    labels_color = ["Animal" if label == 1 else "Human" for label in labels]
    fig = go.Figure()

    for method, embeddings in embeddings_dict.items():
        n_components = embeddings.shape[1]
        columns = [f"Component {i + 1}" for i in range(n_components)]
        df = pd.DataFrame(embeddings, columns=columns)
        df["Label"] = labels_color
        df["Description"] = descriptions

        if n_components == 2:
            fig.add_trace(
                go.Scatter(
                    x=df["Component 1"],
                    y=df["Component 2"],
                    mode="markers",
                    marker=dict(size=6, opacity=0.7),
                    name=f"{method} Embeddings",
                    text=df["Description"],
                    hoverinfo="text",
                )
            )
        elif n_components == 3:
            fig.add_trace(
                go.Scatter3d(
                    x=df["Component 1"],
                    y=df["Component 2"],
                    z=df["Component 3"],
                    mode="markers",
                    marker=dict(size=6, opacity=0.7),
                    name=f"{method} Embeddings",
                    text=df["Description"],
                    hoverinfo="text",
                )
            )

    fig.update_layout(
        title="Embeddings Visualization (Original, PCA, t-SNE)",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        legend_title="Embedding Type",
        template="plotly_white",
    )
    fig.show()
