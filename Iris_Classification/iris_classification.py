from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODELS_DIR = PROJECT_DIR / "models"
IMAGES_DIR = PROJECT_DIR / "images"


def ensure_directories() -> None:
    for directory in (DATASET_DIR, MODELS_DIR, IMAGES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset() -> tuple[pd.DataFrame, object]:
    iris = load_iris()
    data_frame = pd.DataFrame(iris.data, columns=iris.feature_names)
    data_frame["species"] = iris.target
    return data_frame, iris


def explore_dataset(data_frame: pd.DataFrame, target_names: list[str]) -> None:
    print("Shape:", data_frame.shape)
    print("\nInfo:")
    print(data_frame.info())
    print("\nMissing values:")
    print(data_frame.isnull().sum())
    print("\nStatistics:")
    print(data_frame.describe())
    print("\nClass distribution:")
    print(data_frame["species"].value_counts().rename(index=dict(enumerate(target_names))))

    plt.figure(figsize=(8, 5))
    plt.scatter(data_frame["sepal length (cm)"], data_frame["petal length (cm)"], c=data_frame["species"], cmap="viridis", edgecolor="k")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.title("Sepal Length vs Petal Length")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "sepal_vs_petal.png", dpi=150)
    plt.close()


def train_and_evaluate(data_frame: pd.DataFrame, target_names: list[str]) -> tuple[LogisticRegression, StandardScaler]:
    X = data_frame.drop(columns=["species"])
    y = data_frame["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    print("\nAccuracy:", accuracy_score(y_test, predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions, target_names=target_names))

    return model, scaler


def save_artifacts(model: LogisticRegression, scaler: StandardScaler) -> None:
    joblib.dump(model, MODELS_DIR / "iris_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")


def main() -> None:
    ensure_directories()
    data_frame, iris = load_dataset()
    explore_dataset(data_frame, list(iris.target_names))
    model, scaler = train_and_evaluate(data_frame, list(iris.target_names))
    save_artifacts(model, scaler)
    print("\nArtifacts saved to the models directory.")


if __name__ == "__main__":
    main()
