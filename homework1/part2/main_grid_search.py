import os
import time
import numpy as np
import matplotlib.pyplot as plt
import click
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    KFold,
    StratifiedKFold,
    LeaveOneOut,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from PIL import Image
from logistic_regression import LogisticRegression


def load_data(image_folder: str, label_file: str, target_size=(224, 224)):
    """Loads and resizes images to the specified target size, along with labels."""
    labels_df = pd.read_csv(label_file, sep="|")
    labels_df["label"] = labels_df["label"].map({"animal": 1, "human": 0})
    image_names = labels_df["image_name"].values
    labels = labels_df["label"].values

    images = []
    for img_name in image_names:
        img_path = os.path.join(image_folder, img_name.strip())
        img = Image.open(img_path).convert("RGB")
        img = img.resize(target_size)
        images.append(np.array(img))

    images = np.array(images)
    return images, labels


def vectorize_images(images: np.ndarray):
    gray_images = np.mean(images, axis=-1)
    X = gray_images.reshape(gray_images.shape[0], -1)
    print("Shape of X after vectorizing images:", X.shape)
    return X


def create_model_factory(model_name: str):
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1_000_000)
        param_grid = {
            "learning_rate": [0.01, 0.1, 1],
            "regularization": [0.01, 0.1, 1],
            "early_stopping": [True, False],
            "patience": [5, 10, 15],
        }
    elif model_name == "knn":
        model = KNeighborsClassifier(n_jobs=-1)
        param_grid = {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": [10, 20, 30, 40, 50],
            "p": [1, 2],
            "metric": ["minkowski", "euclidean", "manhattan"],
        }
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier(random_state=0)
        param_grid = {
            "criterion": ["gini", "entropy", "log_loss"],
            "splitter": ["best", "random"],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
            "max_features": [None, "sqrt", "log2"],
            "max_leaf_nodes": [None, 10, 20, 30],
            "ccp_alpha": [0.0, 0.01, 0.1],
        }
    else:
        raise ValueError("Invalid model name.")

    return lambda: GridSearchCV(
        model,
        param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
    )


def execute_validation_strategy(
    validation_strategy: str,
    model_factory,
    X_train,
    X_test,
    y_train,
    y_test,
    k_folds: int,
    stratified: bool,
):
    if validation_strategy == "train_test_split":
        print("Training with Simple Train/Test Split")
        return train_test_split_validation(
            model_factory, X_train, X_test, y_train, y_test
        )
    elif validation_strategy == "k_fold":
        print("Performing K-Fold Validation")
        return k_fold_validation(
            model_factory, X_test, y_test, k=k_folds, stratified=stratified
        )
    elif validation_strategy == "leave_one_out":
        print("Performing Leave-One-Out Cross-Validation")
        return leave_one_out_validation(model_factory, X_test, y_test)
    else:
        raise ValueError("Invalid validation strategy.")


def train_test_split_validation(model_factory, X_train, X_test, y_train, y_test):
    grid_search = model_factory()
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy (Simple Train/Test): {accuracy:.2f}")
    return best_model


def k_fold_validation(model_factory, X_train, y_train, k=5, stratified=False):
    kf = StratifiedKFold(n_splits=k) if stratified else KFold(n_splits=k)
    accuracies = []
    models = []

    for train_idx, val_idx in kf.split(X_train, y_train):
        # log with timestamp
        print(f"{time.ctime(time.time())}: Training on fold {len(accuracies) + 1}")
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        grid_search = model_factory()
        grid_search.fit(X_tr, y_tr)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_val)
        accuracies.append(accuracy_score(y_val, y_pred))
        models.append(best_model)

    best_accuracy_idx = np.argmax(accuracies)
    best_accuracy = accuracies[best_accuracy_idx]
    best_model = models[best_accuracy_idx]
    print(f"K-Fold Validation (k={k}) Accuracy: {best_accuracy:.2f}")
    return best_model


def leave_one_out_validation(model_factory, X_train, y_train):
    loo = LeaveOneOut()
    accuracies = []
    models = []

    for train_idx, val_idx in loo.split(X_train):
        print(f"{time.ctime(time.time())}: Training on fold {len(accuracies) + 1}")
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        grid_search = model_factory()
        grid_search.fit(X_tr, y_tr)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_val)
        accuracies.append(accuracy_score(y_val, y_pred))
        models.append(best_model)

    best_accuracy_idx = np.argmax(accuracies)
    best_accuracy = accuracies[best_accuracy_idx]
    best_model = models[best_accuracy_idx]
    print(f"Leave-One-Out Cross-Validation Accuracy: {best_accuracy:.2f}")
    return best_model


@click.command()
@click.option("--image_folder", type=str, help="Path to the folder containing images")
@click.option("--label_file", type=str, help="Path to the file containing labels")
@click.option(
    "--model_name",
    type=click.Choice(["logistic_regression", "knn", "decision_tree"]),
    default="train_test_split",
    help="Name of the model to use",
)
@click.option(
    "--validation_strategy",
    type=click.Choice(["train_test_split", "k_fold", "leave_one_out"]),
    default="train_test_split",
    help="Validation strategy to use",
)
@click.option(
    "--test_size",
    type=float,
    default=0.2,
    help="Size of the test split for train/test validation",
)
@click.option(
    "--k_folds", type=int, default=5, help="Number of folds for K-fold validation"
)
@click.option("--stratified", is_flag=True, help="Use Stratified K-fold validation")
def main(
    image_folder: str,
    label_file: str,
    model_name: str,
    validation_strategy: str,
    test_size: float,
    k_folds: int,
    stratified: bool,
):
    images, labels = load_data(image_folder, label_file)
    X = vectorize_images(images)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = execute_validation_strategy(
        validation_strategy,
        create_model_factory(model_name),
        X_train,
        X_test,
        y_train,
        y_test,
        k_folds,
        stratified,
    )

    print("Error Analysis")
    y_pred = model.predict(X_test if validation_strategy == "train_test_split" else X)
    cm = confusion_matrix(
        y_test if validation_strategy == "train_test_split" else y, y_pred
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


if __name__ == "__main__":
    main()
