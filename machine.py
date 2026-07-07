import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Iris K-Means Clustering",
    page_icon="🌸",
    layout="wide",
)


@st.cache_resource
def train_model():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target

    X = df.drop(columns=["species"]).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    model.fit(X_scaled)

    score = silhouette_score(X_scaled, model.labels_)
    return model, scaler, df, iris, X_scaled, score


model, scaler, df, iris, X_scaled, silhouette = train_model()

CLUSTER_NAMES = {0: "Cluster 0", 1: "Cluster 1", 2: "Cluster 2"}
CLUSTER_COLORS = ["#4C72B0", "#DD8452", "#55A868"]

st.title("🌸 Iris Flower K-Means Clustering")
st.markdown("Unsupervised clustering of the Iris dataset using the **K-Means algorithm** (k=3).")

tab1, tab2, tab3 = st.tabs(["🔍 Predict Cluster", "📊 Cluster Visualizations", "📈 Model Metrics"])

with tab1:
    st.subheader("Predict the Cluster for a New Flower")
    st.markdown("Enter measurements below and click **Predict** to see which cluster the flower belongs to.")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
        sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
    with col2:
        petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
        petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

    if st.button("Predict Cluster", type="primary"):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_scaled = scaler.transform(input_data)
        cluster = int(model.predict(input_scaled)[0])
        distance = model.transform(input_scaled)[0][cluster]

        st.success(f"**Assigned to: {CLUSTER_NAMES[cluster]}**")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Cluster", f"#{cluster}")
        col_b.metric("Distance to Center", f"{distance:.3f}")
        col_c.metric("Total Clusters (k)", "3")

        st.markdown("---")
        st.markdown("#### Cluster Centers (original scale)")
        centers_original = scaler.inverse_transform(model.cluster_centers_)
        centers_df = pd.DataFrame(
            centers_original,
            columns=iris.feature_names,
            index=[f"Cluster {i}" for i in range(3)],
        ).round(2)
        st.dataframe(centers_df, width="stretch")

with tab2:
    st.subheader("Cluster Scatter Plots")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = [CLUSTER_COLORS[l] for l in model.labels_]

    axes[0].scatter(
        df["sepal length (cm)"],
        df["sepal width (cm)"],
        c=colors,
        edgecolor="k",
        alpha=0.8,
        s=60,
    )
    centers_orig = scaler.inverse_transform(model.cluster_centers_)
    axes[0].scatter(
        centers_orig[:, 0],
        centers_orig[:, 1],
        c="red",
        marker="X",
        s=200,
        zorder=5,
        label="Centroids",
    )
    axes[0].set_xlabel("Sepal Length (cm)")
    axes[0].set_ylabel("Sepal Width (cm)")
    axes[0].set_title("Sepal Length vs Sepal Width")
    axes[0].legend()

    axes[1].scatter(
        df["petal length (cm)"],
        df["petal width (cm)"],
        c=colors,
        edgecolor="k",
        alpha=0.8,
        s=60,
    )
    axes[1].scatter(
        centers_orig[:, 2],
        centers_orig[:, 3],
        c="red",
        marker="X",
        s=200,
        zorder=5,
        label="Centroids",
    )
    axes[1].set_xlabel("Petal Length (cm)")
    axes[1].set_ylabel("Petal Width (cm)")
    axes[1].set_title("Petal Length vs Petal Width")
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Elbow Curve (Optimal k)")

    inertias = []
    k_range = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(list(k_range), inertias, marker="o", color="steelblue", linewidth=2)
    ax2.axvline(x=3, color="red", linestyle="--", alpha=0.7, label="Selected k=3")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Inertia (WCSS)")
    ax2.set_title("Elbow Method — Choosing Optimal k")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

with tab3:
    st.subheader("K-Means Model Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Algorithm", "K-Means")
    col2.metric("Clusters (k)", "3")
    col3.metric("Silhouette Score", f"{silhouette:.4f}")

    col4, col5 = st.columns(2)
    col4.metric("Inertia (WCSS)", f"{model.inertia_:.2f}")
    col5.metric("Iterations to Converge", str(model.n_iter_))

    st.markdown("---")
    st.subheader("Cluster Size Distribution")
    unique, counts = np.unique(model.labels_, return_counts=True)
    dist_df = pd.DataFrame({
        "Cluster": [f"Cluster {u}" for u in unique],
        "Count": counts,
        "Percentage": [f"{c/len(model.labels_)*100:.1f}%" for c in counts],
    })
    st.dataframe(dist_df, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("Cluster Centers (original feature scale)")
    centers_original = scaler.inverse_transform(model.cluster_centers_)
    centers_df = pd.DataFrame(
        centers_original,
        columns=iris.feature_names,
        index=[f"Cluster {i}" for i in range(3)],
    ).round(3)
    st.dataframe(centers_df, width="stretch")

    st.markdown("---")
    st.subheader("Dataset with Cluster Labels")
    labeled_df = df.copy()
    labeled_df["cluster"] = model.labels_
    labeled_df["cluster_name"] = labeled_df["cluster"].map(CLUSTER_NAMES)
    st.dataframe(labeled_df, width="stretch", hide_index=True)
