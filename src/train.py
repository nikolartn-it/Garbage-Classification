"""
Trening pipeline za klasifikaciju otpada.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Uvozimo nase module
from data_pipeline import (
    create_dataloaders,
    discover_images,
    find_dataset_root,
    get_class_names,
    validate_images,
)
from model_architecture import create_model
from split_dataset import load_splits


# ============================================
# KONFIGURACIJA
# ============================================

class Config:
    # Putanje
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "raw"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    RESULTS_DIR = PROJECT_ROOT / "results"

    # Hiperparametri
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4

    # Random seed
    SEED = 42

    # Uredjaji
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Broj klasa (automatski ce se detektovati)
    NUM_CLASSES = None
    CLASS_NAMES = None


def set_seed(seed: int = 42) -> None:
    """Fiksira random seed za reproduktivnost."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================
# TRENING FUNKCIJE
# ============================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Jedna epoha treninga."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistika
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{(correct / total):.4f}",
        })

    epoch_loss = total_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validacija za jednu epohu."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = total_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    num_epochs: int,
    device: torch.device,
    model_dir: Path,
    writer: Optional[SummaryWriter] = None,
) -> Dict:
    """
    Glavna trening petlja.

    Returns:
        Dict sa istorijom treninga i najboljim modelom.
    """
    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # Trening
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validacija
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )

        # Sacuvaj istoriju
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Logger
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

        # Scheduler (ako postoji)
        if scheduler:
            scheduler.step()

        # Sacuvaj najbolji model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = model_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, best_model_path)
            print(f"✓ Best model saved (val_acc: {val_acc:.4f})")

    print(f"\nTraining completed! Best val_acc: {best_val_acc:.4f}")

    return history


# ============================================
# GLAVNA FUNKCIJA
# ============================================

def main():
    """Glavna funkcija za pokretanje treninga."""
    # 1. Podesavanje
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Pronalazak dataset-a
    print("\n[1/6] Pronalazim dataset...")
    dataset_root = find_dataset_root(Config.PROJECT_ROOT)
    print(f"Dataset root: {dataset_root}")

     # 3. Ucitavanje splitova
    print("\n[2/6] Ucitavam splitove...")
    splits_dir = Config.PROJECT_ROOT / "data" / "splits"
    train_df, val_df, test_df = load_splits(splits_dir)

    # Mapiranje klasa u brojcane oznake
    class_names = sorted(train_df["class"].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    train_df["label"] = train_df["class"].map(class_to_idx)
    val_df["label"] = val_df["class"].map(class_to_idx)
    test_df["label"] = test_df["class"].map(class_to_idx)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 4. Kreiranje DataLoader-a
    print("\n[3/6] Kreiram DataLoader-e...")
    dataloaders = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        image_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        augment=True,
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # 5. Broj klasa
    print("\n[4/6] Detektujem klase...")
    class_names = get_class_names(dataset_root)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    Config.NUM_CLASSES = num_classes
    Config.CLASS_NAMES = class_names

    # 6. Kreiranje modela
    print("\n[5/6] Kreiram model...")
    model = create_model(num_classes=num_classes)
    model = model.to(device)

    # Broj parametara
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 7. Trening
    print("\n[6/6] Pokrecem trening...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1,
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=Config.LOGS_DIR)

    # Trening
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=Config.NUM_EPOCHS,
        device=device,
        model_dir=Config.MODELS_DIR,
        writer=writer,
    )

    writer.close()

    # 8. Evaluacija na test skupu
    print("\nEvaluacija na test skupu...")
    test_loss, test_acc = validate_epoch(
        model, test_loader, criterion, device
    )
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # 9. Cuvanje rezultata
    results = {
        "history": history,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "best_val_acc": max(history["val_acc"]),
        "config": {
            "image_size": Config.IMAGE_SIZE,
            "batch_size": Config.BATCH_SIZE,
            "num_epochs": Config.NUM_EPOCHS,
            "learning_rate": Config.LEARNING_RATE,
            "weight_decay": Config.WEIGHT_DECAY,
            "num_classes": Config.NUM_CLASSES,
            "class_names": Config.CLASS_NAMES,
        }
    }

    results_path = Config.RESULTS_DIR / "training_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {results_path}")


if __name__ == "__main__":
    main()