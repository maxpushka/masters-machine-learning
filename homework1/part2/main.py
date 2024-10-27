import os
import numpy as np
import matplotlib.pyplot as plt
import click
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from PIL import Image
from logistic_regression import LogisticRegression


def load_data(image_folder: str, label_file: str, target_size=(224, 224)):
    """Loads and resizes images to the specified target size, along with labels."""
    # Load labels and map 'animal' to 1, 'human' to 0
    labels_df = pd.read_csv(label_file, sep="|")
    labels_df["label"] = labels_df["label"].map({"animal": 1, "human": 0})

    # Extract filenames and labels
    image_names = labels_df["image_name"].values
    labels = labels_df["label"].values

    # Load and resize images
    images = []
    for img_name in image_names:
        img_path = os.path.join(image_folder, img_name.strip())
        img = Image.open(img_path).convert("RGB")
        img = img.resize(target_size)  # Resize to the target size (e.g., 224x224)
        images.append(np.array(img))

    # Convert to a NumPy array with consistent shape
    images = np.array(
        images
    )  # Shape will be (N, 224, 224, 3) if target_size=(224, 224)
    return images, labels


def vectorize_images(images: np.ndarray):
    """Vectorizes images by converting them to grayscale and flattening."""
    # Convert images to grayscale by averaging across the RGB channels and flatten them
    gray_images = np.mean(images, axis=-1)  # Taking the mean across RGB channels
    X = gray_images.reshape(
        gray_images.shape[0], -1
    )  # Flatten the images to 1D vectors
    return X


def validation_split(X: np.ndarray, y: np.ndarray, test_size=0.2):
    """Splits data into train and test."""
    return train_test_split(X, y, test_size=test_size, random_state=42)


def create_model(model_name: str):
    """Creates a model of the specified name."""
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1_000_000, verbose=True)
    elif model_name == "knn":
        model = KNeighborsClassifier()
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier()
    else:
        raise ValueError(
            "Invalid model name. Choose from ['logistic_regression', 'knn', 'decision_tree']"
        )
    return model


def k_fold_validation(model, X_train, y_train, k=5, stratified=False):
    """Performs K-fold or Stratified K-fold cross-validation."""
    accuracies = []
    if stratified:
        kf = StratifiedKFold(n_splits=k)
    else:
        kf = KFold(n_splits=k)

    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        accuracies.append(accuracy_score(y_val, y_pred))

    avg_accuracy = np.mean(accuracies)
    print(f"K-Fold Validation (k={k}) Accuracy: {avg_accuracy:.2f}")
    return avg_accuracy


@click.command()
@click.option("--image_folder", type=str, help="Path to the folder containing images")
@click.option("--label_file", type=str, help="Path to the file containing labels")
@click.option("--model_name", type=str, help="Name of the model to use")
@click.option("--test_size", type=float, default=0.2, help="Size of the test split")
@click.option(
    "--k_folds", type=int, default=5, help="Number of folds for K-fold validation"
)
@click.option("--stratified", is_flag=True, help="Use Stratified K-fold validation")
def main(
    image_folder: str,
    label_file: str,
    model_name: str,
    test_size: float,
    k_folds: int,
    stratified: bool,
):
    # Load and preprocess data
    images, labels = load_data(image_folder, label_file)
    X = vectorize_images(images)
    y = labels  # Convert labels if needed

    # Split data into train and test
    X_train, X_test, y_train, y_test = validation_split(X, y, test_size)

    # Create model
    model = create_model(model_name)

    # Validation strategies
    print("Training with Simple Train/Test Split")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy (Simple Train/Test): {accuracy:.2f}")

    # K-Fold validation
    print("Performing K-Fold Validation")
    k_fold_validation(model, X_train, y_train, k=k_folds, stratified=stratified)

    # Error analysis and confusion matrix
    print("Error Analysis")
    incorrect_indices = np.where(y_pred != y_test)[0]
    print(f"Number of misclassified samples: {len(incorrect_indices)}")

    # Display misclassified images
    for i, idx in enumerate(incorrect_indices[:10]):
        plt.imshow(X_test[idx].reshape(images.shape[1], images.shape[2]), cmap="gray")
        plt.title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
        plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


if __name__ == "__main__":
    main()
