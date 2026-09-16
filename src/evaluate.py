"""
Evaluacija modela na test skupu.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from data_pipeline import create_dataloaders, get_class_names
from model_architecture import Model, create_model
from split_dataset import load_splits


class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
    DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "Garbage classification" / "Garbage classification"
    
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, num_classes):
    model = Model(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Podrska za oba formata: cist state_dict ili checkpoint dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list,
) -> dict:
    """
    Evaluacija modela na test skupu.
    
    Returns:
        dict: {
            'accuracy': float,
            'precision': dict,
            'recall': dict,
            'f1': dict,
            'confusion_matrix': np.ndarray,
            'classification_report': str,
            'predictions': list,
            'true_labels': list,
        }
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Metrike
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    return {
        "accuracy": accuracy,
        "precision": dict(zip(class_names, precision)),
        "recall": dict(zip(class_names, recall)),
        "f1": dict(zip(class_names, f1)),
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": all_preds,
        "true_labels": all_labels,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: Path,
) -> None:
    """Plotuje i cuva confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Confusion matrix saved to {save_path}")


def plot_metrics_per_class(
    precision: dict,
    recall: dict,
    f1: dict,
    class_names: list,
    save_path: Path,
) -> None:
    """Plotuje metrike po klasama."""
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, [precision[cls] for cls in class_names], width, label="Precision")
    ax.bar(x, [recall[cls] for cls in class_names], width, label="Recall")
    ax.bar(x + width, [f1[cls] for cls in class_names], width, label="F1-score")
    
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Metrics per Class")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Metrics plot saved to {save_path}")


def save_results(results: dict, save_path: Path) -> None:
    """Cuva rezultate evaluacije u JSON fajl."""
    # Konvertuj numpy array u listu za JSON
    results_json = {
        "accuracy": results["accuracy"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1": results["f1"],
        "confusion_matrix": results["confusion_matrix"].tolist(),
        "classification_report": results["classification_report"],
    }
    
    with open(save_path, "w") as f:
        json.dump(results_json, f, indent=2)
    
    print(f"✓ Results saved to {save_path}")


def print_results(results: dict) -> None:
    """Stampanje rezultata u konzolu."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    
    print(f"\nAccuracy: {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)")
    
    print("\nPer-class metrics:")
    print("-" * 50)
    for cls in results["precision"].keys():
        print(f"  {cls:12s} | P: {results['precision'][cls]:.4f} | R: {results['recall'][cls]:.4f} | F1: {results['f1'][cls]:.4f}")
    
    print("\nClassification Report:")
    print("-" * 50)
    print(results["classification_report"])


def main():
    """Glavna funkcija za evaluaciju."""
    print("=" * 50)
    print("Evaluacija modela")
    print("=" * 50)
    
    # 1. Ucitavanje podataka
    print("\n[1/4] Ucitavanje test skupa...")
    train_df, val_df, test_df = load_splits(Config.SPLITS_DIR)
    
    # Mapiranje klasa
    class_names = sorted(train_df["class"].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    # Dodaj label kolonu za sve skupove
    train_df["label"] = train_df["class"].map(class_to_idx)
    val_df["label"] = val_df["class"].map(class_to_idx)
    test_df["label"] = test_df["class"].map(class_to_idx)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # DataLoader
    dataloaders = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        image_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        augment=False,
    )
    
    test_loader = dataloaders["test"]
    print(f"Test samples: {len(test_df)}")
    
    # 2. Ucitavanje modela
    print("\n[2/4] Ucitavanje modela...")
    best_model_path = Config.MODELS_DIR / "best_model.pth"
    
    if not best_model_path.exists():
        print(f"❌ Model not found at {best_model_path}")
        print("   Please run training first: python src/train.py")
        return
    
    model = load_model(best_model_path, num_classes=len(class_names))
    
    # 3. Evaluacija
    print("\n[3/4] Evaluacija na test skupu...")
    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=Config.DEVICE,
        class_names=class_names,
    )
    
    # 4. Cuvanje i prikaz rezultata
    print("\n[4/4] Cuvanje rezultata...")
    
    # Cuvanje rezultata
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_results(results, Config.RESULTS_DIR / "evaluation_results.json")
    
    # Grafikoni
    plots_dir = Config.RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_confusion_matrix(
        results["confusion_matrix"],
        class_names,
        plots_dir / "confusion_matrix.png",
    )
    
    plot_metrics_per_class(
        results["precision"],
        results["recall"],
        results["f1"],
        class_names,
        plots_dir / "metrics_per_class.png",
    )
    
    # Stampanje rezultata
    print_results(results)
    
    print("\n✅ Evaluation completed!")
    print(f"   Results saved to {Config.RESULTS_DIR}")


if __name__ == "__main__":
    main()