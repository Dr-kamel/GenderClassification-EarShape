import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


from configs.config import (
    MODEL_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    HISTORY_DIR,
    CLASS_NAMES
)

from dataset import (
    load_dataframes,
    create_dataset
)

from models import (
    AVAILABLE_MODELS
)

def evaluate_model(model_name):

    print("\n" + "=" * 70)
    print(
        f"EVALUATING MODEL: {model_name}"
    )
    print("=" * 70)

    (
        _,
        _,
        test_df
    ) = load_dataframes()

    test_dataset = create_dataset(

        test_df,

        model_name=model_name,

        training=False
    )

    model_path = (
        MODEL_DIR
        / f"best_{model_name}.keras"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = tf.keras.models.load_model(
        model_path
    )

    y_true = []

    y_prob = []

    for images, labels in test_dataset:

        predictions = model.predict(
            images,
            verbose=0
        ).flatten()

        y_true.extend(
            labels.numpy()
        )

        y_prob.extend(
            predictions
        )

    y_true = np.asarray(
        y_true
    ).astype(int)

    y_prob = np.asarray(
        y_prob
    )

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    auc_roc = roc_auc_score(
        y_true,
        y_prob
    )

    print("\n" + "=" * 60)

    print(
        f"{model_name} - TEST RESULTS"
    )

    print("=" * 60)

    print(
        f"Accuracy   : "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision  : "
        f"{precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall     : "
        f"{recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1-score   : "
        f"{f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"AUC-ROC    : "
        f"{auc_roc:.4f}"
    )

    report = classification_report(

        y_true,

        y_pred,

        target_names=[
            CLASS_NAMES[0],
            CLASS_NAMES[1]
        ],

        zero_division=0,

        output_dict=True
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_path = (
        METRICS_DIR
        / f"{model_name}_classification_report.csv"
    )

    report_df.to_csv(
        report_path
    )

    print("\nClassification Report:")

    print(
        classification_report(

            y_true,

            y_pred,

            target_names=[
                CLASS_NAMES[0],
                CLASS_NAMES[1]
            ],

            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\nConfusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=[
            CLASS_NAMES[0],
            CLASS_NAMES[1]
        ]
    )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    disp.plot(
        ax=ax
    )

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.tight_layout()

    cm_path = (
        FIGURES_DIR
        / f"{model_name}_confusion_matrix.png"
    )

    plt.savefig(
        cm_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(

        fpr,

        tpr,

        label=(
            f"{model_name} "
            f"(AUC = {auc_roc:.4f})"
        )
    )

    plt.plot(

        [0, 1],

        [0, 1],

        linestyle="--",

        label="Random Classifier"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        f"ROC Curve - {model_name}"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    roc_path = (
        FIGURES_DIR
        / f"{model_name}_roc_curve.png"
    )

    plt.savefig(
        roc_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    metrics = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "AUC": auc_roc
            }
        ]
    )

    metrics_path = (
        METRICS_DIR
        / f"{model_name}_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False
    )

    history_path = (
        HISTORY_DIR
        / f"{model_name}_history.csv"
    )

    if history_path.exists():

        history_df = pd.read_csv(
            history_path
        )

        epochs = range(
            1,
            len(history_df) + 1
        )

        plt.figure(
            figsize=(8, 6)
        )

        plt.plot(

            epochs,

            history_df["accuracy"],

            label="Training Accuracy"
        )

        plt.plot(

            epochs,

            history_df["val_accuracy"],

            label="Validation Accuracy"
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Accuracy"
        )

        plt.title(
            f"{model_name} - Training and Validation Accuracy"
        )

        plt.legend()

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        accuracy_path = (
            FIGURES_DIR
            / f"{model_name}_training_accuracy.png"
        )

        plt.savefig(
            accuracy_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        plt.figure(
            figsize=(8, 6)
        )

        plt.plot(

            epochs,

            history_df["loss"],

            label="Training Loss"
        )

        plt.plot(

            epochs,

            history_df["val_loss"],

            label="Validation Loss"
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Loss"
        )

        plt.title(
            f"{model_name} - Training and Validation Loss"
        )

        plt.legend()

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        loss_path = (
            FIGURES_DIR
            / f"{model_name}_training_loss.png"
        )

        plt.savefig(
            loss_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    print("\nResults saved successfully.")

    print(
        f"Metrics: {metrics_path}"
    )

    print(
        f"Confusion matrix: {cm_path}"
    )

    print(
        f"ROC curve: {roc_path}"
    )

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Evaluate a trained "
            "ear gender classification model."
        )
    )

    parser.add_argument(

        "--model",

        type=str,

        required=True,

        choices=AVAILABLE_MODELS,

        help="Model architecture to evaluate."
    )

    args = parser.parse_args()

    evaluate_model(
        args.model
    )


if __name__ == "__main__":
    main()