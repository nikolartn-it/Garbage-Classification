"""
Napredna vizualizacija rezultata.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

from data_pipeline import create_dataloaders, get_class_names
from model_architecture import create_model
from split_dataset import load_splits


class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
    DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "Garbage classification" / "Garbage classification"
    PLOTS_DIR = RESULTS_DIR / "plots_full"
    
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_results():
    """Ucitava rezultate treninga i evaluacije."""
    # Ucitaj history iz training_results.json
    with open(Config.RESULTS_DIR / "training_results.json", "r") as f:
        train_results = json.load(f)
    
    # Ucitaj evaluacione rezultate
    with open(Config.RESULTS_DIR / "evaluation_results.json", "r") as f:
        eval_results = json.load(f)
    
    return train_results, eval_results


def plot_learning_curves(history: dict, save_dir: Path) -> None:
    """Plotuje learning curves (loss i accuracy)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss", marker='o')
    axes[0].plot(history["val_loss"], label="Val Loss", marker='s')
    axes[0].set_title("Learning Curves - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history["train_acc"], label="Train Acc", marker='o')
    axes[1].plot(history["val_acc"], label="Val Acc", marker='s')
    axes[1].set_title("Learning Curves - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "learning_curves.png", dpi=300)
    plt.close()
    print("✓ Learning curves saved")


def plot_class_distribution(train_df, val_df, test_df, save_dir: Path) -> None:
    """Plotuje distribuciju klasa u sva tri skupa."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, df, title in zip(axes, [train_df, val_df, test_df], ["Train", "Validation", "Test"]):
        counts = df["class"].value_counts()
        ax.bar(counts.index, counts.values)
        ax.set_title(f"{title} - Class Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / "class_distribution.png", dpi=300)
    plt.close()
    print("✓ Class distribution saved")


def plot_roc_curves(model, dataloader, class_names, save_dir: Path) -> None:
    """Plotuje ROC krive za svaku klasu (One-vs-Rest)."""
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(Config.DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Binarizacija labela za One-vs-Rest
    n_classes = len(class_names)
    y_bin = label_binarize(all_labels, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "roc_curves.png", dpi=300)
    plt.close()
    print("✓ ROC curves saved")


def plot_precision_recall_curves(model, dataloader, class_names, save_dir: Path) -> None:
    """Plotuje Precision-Recall krive za svaku klasu."""
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(Config.DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    n_classes = len(class_names)
    y_bin = label_binarize(all_labels, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], all_probs[:, i])
        plt.plot(recall, precision, label=f"{class_names[i]}")
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "precision_recall_curves.png", dpi=300)
    plt.close()
    print("✓ Precision-Recall curves saved")


def plot_error_examples(model, dataloader, class_names, save_dir: Path, n_examples: int = 16) -> None:
    """Prikazuje primere pogresno klasifikovanih slika."""
    model.eval()
    images_list = []
    true_labels = []
    pred_labels = []
    probs_list = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(Config.DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    images_list.append(images[i].cpu())
                    true_labels.append(labels[i].item())
                    pred_labels.append(preds[i].item())
                    probs_list.append(probs[i].cpu().numpy())
                    
                    if len(images_list) >= n_examples:
                        break
            if len(images_list) >= n_examples:
                break
    
    if not images_list:
        print("No errors found!")
        return
    
    # Prikaz slika
    n_cols = 4
    n_rows = (len(images_list) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, (img, true, pred, probs) in enumerate(zip(images_list, true_labels, pred_labels, probs_list)):
        # Denormalizacija
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        axes[idx].imshow(img.permute(1, 2, 0))
        axes[idx].set_title(f"True: {class_names[true]}\nPred: {class_names[pred]}\nProb: {probs[pred]:.3f}")
        axes[idx].axis('off')
    
    # Sakrij prazne subplotove
    for idx in range(len(images_list), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / "error_examples.png", dpi=300)
    plt.close()
    print(f"✓ Error examples saved ({len(images_list)} examples)")


def plot_top_k_accuracy(history: dict, save_dir: Path) -> None:
    """Plotuje Top-1, Top-2, Top-3 accuracy (ako je dostupno)."""
    # Ovo zahteva da se belezi top-k accuracy tokom treninga
    # Ako nema podataka, preskoci
    if "top1_acc" not in history:
        print("⚠️ Top-k accuracy not available (not logged during training)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(history["top1_acc"], label="Top-1", marker='o')
    if "top2_acc" in history:
        ax.plot(history["top2_acc"], label="Top-2", marker='s')
    if "top3_acc" in history:
        ax.plot(history["top3_acc"], label="Top-3", marker='^')
    
    ax.set_title("Top-k Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "top_k_accuracy.png", dpi=300)
    plt.close()
    print("✓ Top-k accuracy saved")


def main():
    """Generise sve grafike."""
    Config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("Generisanje svih grafika")
    print("=" * 50)
    
    # 1. Ucitavanje podataka
    print("\n[1/6] Ucitavanje podataka...")
    train_df, val_df, test_df = load_splits(Config.SPLITS_DIR)
    
    class_names = sorted(train_df["class"].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    train_df["label"] = train_df["class"].map(class_to_idx)
    val_df["label"] = val_df["class"].map(class_to_idx)
    test_df["label"] = test_df["class"].map(class_to_idx)
    
    # 2. Ucitavanje rezultata
    print("\n[2/6] Ucitavanje rezultata...")
    train_results, eval_results = load_results()
    history = train_results["history"]
    
    # 3. Ucitavanje modela
    print("\n[3/6] Ucitavanje modela...")
    model_path = Config.MODELS_DIR / "best_model.pth"
    model = create_model(num_classes=len(class_names))
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(Config.DEVICE)
    model.eval()
    
    # 4. Kreiranje dataloadera
    print("\n[4/6] Kreiranje dataloadera...")
    dataloaders = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        image_size=224,
        batch_size=32,
        num_workers=0,
        augment=False,
    )
    test_loader = dataloaders["test"]
    
    # 5. Generisanje grafika
    print("\n[5/6] Generisanje grafika...")
    
    # Learning curves
    plot_learning_curves(history, Config.PLOTS_DIR)
    
    # Class distribution
    plot_class_distribution(train_df, val_df, test_df, Config.PLOTS_DIR)
    
    # ROC curves
    plot_roc_curves(model, test_loader, class_names, Config.PLOTS_DIR)
    
    # Precision-Recall curves
    plot_precision_recall_curves(model, test_loader, class_names, Config.PLOTS_DIR)
    
    # Error examples
    plot_error_examples(model, test_loader, class_names, Config.PLOTS_DIR, n_examples=16)
    
    # Top-k accuracy (ako postoji)
    plot_top_k_accuracy(history, Config.PLOTS_DIR)
    
    # 6. Gotovo
    print("\n[6/6] Gotovo!")
    print(f"✓ Sve grafike sacuvane u: {Config.PLOTS_DIR}")
    print("\nGenerisani fajlovi:")
    for f in Config.PLOTS_DIR.glob("*.png"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()