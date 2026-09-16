"""
Regenerisanje svih grafika sa najnovijim modelom i rezultatima.

Pokretanje: python src/regenerate_plots.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_architecture import Model
from data_pipeline import create_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_FULL_DIR = RESULTS_DIR / "plots_full"
MODELS_DIR = PROJECT_ROOT / "models"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_FULL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")


def load_dataframes():
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    class_names = sorted(train_df["class"].unique().tolist())
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    for df in (train_df, val_df, test_df):
        df["label"] = df["class"].map(class_to_idx)

    return train_df, val_df, test_df, class_names


def load_model():
    """Ucitava best_model.pth (podrzava cist state_dict i checkpoint dict)."""
    model = Model(num_classes=6)
    state = torch.load(MODELS_DIR / "best_model.pth", map_location=DEVICE)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.eval()
    return model


def get_predictions(model, test_loader):
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())

    return np.array(all_labels), np.concatenate(all_probs, axis=0)


def plot_learning_curves(all_results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, result in all_results.items():
        if result.get("status") != "completed":
            continue
        history = result.get("history", {})
        if not history.get("train_acc"):
            continue

        epochs = range(1, len(history["train_acc"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label=f"{name} (train)", alpha=0.7)
        axes[0].plot(epochs, history["val_loss"], label=f"{name} (val)", linestyle='--', alpha=0.7)
        axes[1].plot(epochs, history["train_acc"], label=f"{name} (train)", alpha=0.7)
        axes[1].plot(epochs, history["val_acc"], label=f"{name} (val)", linestyle='--', alpha=0.7)

    axes[0].set_title("Learning Curves - Loss")
    axes[0].set_xlabel("Epoha")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Learning Curves - Accuracy")
    axes[1].set_xlabel("Epoha")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_FULL_DIR / "learning_curves.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("  [OK] learning_curves.png")


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Matrica konfuzije - najbolji model")
    plt.xlabel("Predikovana klasa")
    plt.ylabel("Stvarna klasa")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=100, bbox_inches="tight")
    plt.savefig(PLOTS_FULL_DIR / "confusion_matrix.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("  [OK] confusion_matrix.png")


def plot_per_class_metrics(y_true, y_pred, class_names):
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    precisions = [report[c]["precision"] for c in class_names]
    recalls = [report[c]["recall"] for c in class_names]
    f1s = [report[c]["f1-score"] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precisions, width, label='Precision')
    ax.bar(x, recalls, width, label='Recall')
    ax.bar(x + width, f1s, width, label='F1')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15)
    ax.set_ylabel("Vrednost")
    ax.set_title("Metrike po klasama - najbolji model")
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "metrics_per_class.png", dpi=100, bbox_inches="tight")
    plt.savefig(PLOTS_FULL_DIR / "metrics_per_class.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("  [OK] metrics_per_class.png")


def plot_roc_curves(y_true, y_probs, class_names):
    plt.figure(figsize=(8, 6))

    for i, name in enumerate(class_names):
        y_bin = (np.array(y_true) == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC krive - najbolji model")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_FULL_DIR / "roc_curves.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("  [OK] roc_curves.png")


def plot_pr_curves(y_true, y_probs, class_names):
    plt.figure(figsize=(8, 6))

    for i, name in enumerate(class_names):
        y_bin = (np.array(y_true) == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_bin, y_probs[:, i])
        ap = average_precision_score(y_bin, y_probs[:, i])
        plt.plot(recall, precision, label=f"{name} (AP = {ap:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall krive - najbolji model")
    plt.legend(loc="best", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_FULL_DIR / "precision_recall_curves.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("  [OK] precision_recall_curves.png")


def plot_class_distribution():
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    counts = train_df["class"].value_counts()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values)
    plt.xlabel("Klasa")
    plt.ylabel("Broj slika")
    plt.title("Raspodela uzoraka po klasama")
    plt.xticks(rotation=15)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(PLOTS_FULL_DIR / "class_distribution.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("  [OK] class_distribution.png")


def main():
    print("=" * 50)
    print("Regenerisanje grafika")
    print("=" * 50)

    # 1. Learning curves
    print("\n[1/5] Learning curves...")
    with open(RESULTS_DIR / "all_experiments.json", "r", encoding="utf-8") as f:
        all_results = json.load(f)
    plot_learning_curves(all_results)

    # 2. Model i podaci
    print("\n[2/5] Ucitavanje modela i test skupa...")
    model = load_model()
    train_df, val_df, test_df, class_names = load_dataframes()
    dataloaders = create_dataloaders(
        train_df=train_df, val_df=val_df, test_df=test_df,
        image_size=224, batch_size=32, num_workers=0, augment=False,
    )

    # 3. Predikcije
    print("\n[3/5] Predikcije...")
    y_true, y_probs = get_predictions(model, dataloaders["test"])
    y_pred = np.argmax(y_probs, axis=1)

    # 4. Grafici
    print("\n[4/5] Grafici...")
    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_per_class_metrics(y_true, y_pred, class_names)
    plot_roc_curves(y_true, y_probs, class_names)
    plot_pr_curves(y_true, y_probs, class_names)
    plot_class_distribution()

    print("\n[5/5] Gotovo!")
    print(f"\nGrafici sacuvani u:")
    print(f"  - {PLOTS_DIR}")
    print(f"  - {PLOTS_FULL_DIR}")


if __name__ == "__main__":
    main()