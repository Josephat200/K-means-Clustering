from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
IMAGES_DIR = PROJECT_DIR / "images"


def ensure_directories() -> None:
    for directory in (MODELS_DIR, IMAGES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset() -> tuple[pd.DataFrame, object]:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    return df, iris


def explore_dataset(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("\nStatistics:\n", df.describe())
    print("\nMissing values:\n", df.isnull().sum())


def find_optimal_k(X_scaled: np.ndarray, max_k: int = 10) -> None:
    inertias = []
    silhouettes = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(k_range, inertias, marker="o", color="steelblue")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (WCSS)")
    axes[0].set_title("Elbow Method")
    axes[0].axvline(x=3, color="red", linestyle="--", alpha=0.6, label="k=3")
    axes[0].legend()

    axes[1].plot(k_range, silhouettes, marker="o", color="seagreen")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Analysis")
    axes[1].axvline(x=3, color="red", linestyle="--", alpha=0.6, label="k=3")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "elbow_silhouette.png", dpi=150)
    plt.close()
    print("Saved elbow/silhouette plot.")


def train_kmeans(X_scaled: np.ndarray, n_clusters: int = 3) -> KMeans:
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(X_scaled)
    score = silhouette_score(X_scaled, model.labels_)
    print(f"\nKMeans fitted with k={n_clusters}")
    print(f"Inertia (WCSS): {model.inertia_:.4f}")
    print(f"Silhouette Score: {score:.4f}")
    return model


def plot_clusters(df: pd.DataFrame, labels: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(
        df["sepal length (cm)"],
        df["sepal width (cm)"],
        c=labels,
        cmap="viridis",
        edgecolor="k",
        alpha=0.8,
    )
    axes[0].set_xlabel("Sepal Length (cm)")
    axes[0].set_ylabel("Sepal Width (cm)")
    axes[0].set_title("Clusters: Sepal Length vs Width")

    axes[1].scatter(
        df["petal length (cm)"],
        df["petal width (cm)"],
        c=labels,
        cmap="viridis",
        edgecolor="k",
        alpha=0.8,
    )
    axes[1].set_xlabel("Petal Length (cm)")
    axes[1].set_ylabel("Petal Width (cm)")
    axes[1].set_title("Clusters: Petal Length vs Width")

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "cluster_scatter.png", dpi=150)
    plt.close()
    print("Saved cluster scatter plot.")


def save_artifacts(model: KMeans, scaler: StandardScaler) -> None:
    joblib.dump(model, MODELS_DIR / "kmeans_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print("Artifacts saved to models/")


def main() -> None:
    ensure_directories()
    df, iris = load_dataset()
    explore_dataset(df)

    X = df.drop(columns=["species"]).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    find_optimal_k(X_scaled)
    model = train_kmeans(X_scaled, n_clusters=3)
    plot_clusters(df, model.labels_)
    save_artifacts(model, scaler)

    print("\nDone. All artifacts saved.")


if __name__ == "__main__":
    main()
