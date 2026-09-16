"""
Pokretanje svih eksperimenata definisanih u experiment_utils.py.

Za svaku konfiguraciju:
- kreira model sa odgovarajucim filters/dropout
- trenira model
- evaluira NAJBOLJI model na test skupu
- cuva rezultate u results/all_experiments.json
"""

import os
import sys
import json
import time
from pathlib import Path

import pandas as pd
import torch

# Dodaj src/ u sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_architecture import create_model
from data_pipeline import create_dataloaders
from experiment_utils import create_experiment_configs, ExperimentLogger


# ============================================================
# Putanje
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"


def load_dataframes():
    """
    Ucitava train/val/test CSV-ove i dodaje 'label' kolonu (int).
    
    Returns:
        train_df, val_df, test_df, class_names
    """
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    # Mapiranje: class (string) -> label (int)
    class_names = sorted(train_df["class"].unique().tolist())
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    def add_label(df):
        df = df.copy()
        df["label"] = df["class"].map(class_to_idx)
        return df

    train_df = add_label(train_df)
    val_df = add_label(val_df)
    test_df = add_label(test_df)

    return train_df, val_df, test_df, class_names


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Jedna epoha treninga. Vraca (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    """Validacija. Vraca (loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def evaluate_on_test(model, test_loader, device, class_names):
    """Detaljna evaluacija na test skupu."""
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    cm = confusion_matrix(
        all_labels, all_preds,
        labels=list(range(len(class_names))),
    ).tolist()

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "test_acc": float(acc),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def run_experiment(config, train_df, val_df, test_df, class_names, device):
    """Pokrece jedan eksperiment i vraca rezultate."""
    name = config["name"]
    image_size = config.get("image_size", 224)
    batch_size = config.get("batch_size", 32)
    learning_rate = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 0.0001)
    num_epochs = config.get("num_epochs", 30)
    augmentation = config.get("augmentation", True)
    filters = config.get("filters", None)
    dropout = config.get("dropout", 0.5)

    print(f"\n{'='*60}")
    print(f"Starting experiment: {name}")
    print(f"{'='*60}")
    print(f"Config: {json.dumps(config, indent=2, default=str)}")

    # --- DataLoader-i ---
    dataloaders = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=0,  # Windows-friendly
        augment=augmentation,
    )

    # --- Model ---
    model = create_model(
        num_classes=len(class_names),
        config={"filters": filters, "dropout": dropout},
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # --- Loss i optimizer ---
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1
    )

    # --- Logger ---
    logger = ExperimentLogger(name, log_dir=str(LOGS_DIR))
    logger.save_config(config)

    # --- Trening ---
    best_val_acc = 0.0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODELS_DIR / f"{name}_best.pth"

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 50)

        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            model, dataloaders["val"], criterion, device
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"[OK] Best model saved (val_acc: {val_acc:.4f})")

    elapsed = time.time() - start_time

    # --- Evaluacija NAJBOLJEG modela na test skupu ---
    print(f"\nEvaluacija najboljeg modela na test skupu...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate_on_test(
        model, dataloaders["test"], device, class_names
    )

    print(f"Test Accuracy: {test_metrics['test_acc']:.4f}")

    # --- Sacuvaj rezultate ---
    result = {
        "name": name,
        "status": "completed",
        "config": config,
        "total_params": total_params,
        "best_val_acc": float(best_val_acc),
        "test_acc": test_metrics["test_acc"],
        "per_class": test_metrics["per_class"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "history": history,
        "elapsed_seconds": elapsed,
        "model_path": str(best_model_path.relative_to(PROJECT_ROOT)),
    }

    logger.save_results(result)
    return result


def main():
    print("=" * 60)
    print("Pokretanje svih eksperimenata")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading dataset...")
    train_df, val_df, test_df, class_names = load_dataframes()
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Classes: {class_names}")

    configs = create_experiment_configs()
    print(f"\nUkupno konfiguracija: {len(configs)}")

    all_results = {}
    for config in configs:
        try:
            result = run_experiment(
                config=config,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                class_names=class_names,
                device=device,
            )
            all_results[config["name"]] = result
        except Exception as e:
            print(f"\n[ERROR] Experiment '{config['name']}' failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[config["name"]] = {
                "name": config["name"],
                "status": "failed",
                "error": str(e),
            }

    # Sacuvaj sve rezultate
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "all_experiments.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Finalni ispis
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)
    print(f"\nResults saved to: {results_path}\n")

    print(f"{'Experiment':<20} | {'Best Val':<10} | {'Test':<10}")
    print("-" * 50)
    for name, result in all_results.items():
        if result.get("status") == "completed":
            print(
                f"{name:<20} | {result['best_val_acc']:.4f}     | "
                f"{result['test_acc']:.4f}"
            )
        else:
            print(f"{name:<20} | FAILED")


if __name__ == "__main__":
    main()