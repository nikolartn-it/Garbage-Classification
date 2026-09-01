"""
Pokretanje svih 5 eksperimenata.
"""

import json
import sys
from pathlib import Path

# Dodaj src u path
sys.path.append(str(Path(__file__).parent))

from data_pipeline import create_dataloaders, find_dataset_root, get_class_names
from experiment_utils import ExperimentLogger, create_experiment_configs
from model_architecture import create_model
from split_dataset import load_splits
from train import train, Config as TrainConfig


def run_experiment(
    config: dict,
    train_df,
    val_df,
    test_df,
    class_names: list,
    device: str = "cpu",
) -> dict:
    """Pokrece jedan eksperiment sa datom konfiguracijom."""
    
    print("\n" + "=" * 60)
    print(f"Starting experiment: {config['name']}")
    print("=" * 60)
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Kreiranje logger-a
    logger = ExperimentLogger(
        experiment_name=config["name"],
        config=config,
        logs_dir=Path("logs"),
    )
    
    # Kreiranje dataloader-a
    dataloaders = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        image_size=config.get("image_size", 224),
        batch_size=config.get("batch_size", 32),
        num_workers=TrainConfig.NUM_WORKERS,
        augment=config.get("augmentation", True),
    )
    
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]
    
    # Kreiranje modela
    num_classes = len(class_names)
    model = create_model(num_classes=num_classes)
    model = model.to(device)
    
    # Broj parametara
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Trening
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 0.001),
        weight_decay=config.get("weight_decay", 0.0001),
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1,
    )
    
    # Treniraj
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config.get("num_epochs", 30),
        device=device,
        model_dir=Path("models") / config["name"],
        writer=None,  # TensorBoard opciono
    )
    
    # Sacuvaj najbolji model
    best_val_acc = max(history["val_acc"])
    logger.save_model(model, epoch=len(history["train_loss"]) - 1, val_acc=best_val_acc)
    
    # Evaluacija na test skupu
    from train import validate_epoch
    test_loss, test_acc = validate_epoch(
        model, test_loader, criterion, device
    )
    
    # Cuvanje rezultata
    results = {
        "name": config["name"],
        "config": config,
        "history": history,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
    }
    
    logger.save_results(results)
    
    print(f"\n✅ Experiment {config['name']} completed!")
    print(f"   Best Val Acc: {best_val_acc:.4f}")
    print(f"   Test Acc: {test_acc:.4f}")
    
    return results


def main():
    """Pokrece sve eksperimente redom."""
    
    # Podesavanje
    device = TrainConfig.DEVICE
    print(f"Using device: {device}")
    
    # Ucitavanje podataka
    print("\nLoading dataset...")
    project_root = Path(__file__).parent.parent
    dataset_root = find_dataset_root(project_root)
    splits_dir = project_root / "data" / "splits"
    
    train_df, val_df, test_df = load_splits(splits_dir)
    
    # Mapiranje klasa
    class_names = sorted(train_df["class"].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    train_df["label"] = train_df["class"].map(class_to_idx)
    val_df["label"] = val_df["class"].map(class_to_idx)
    test_df["label"] = test_df["class"].map(class_to_idx)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Classes: {class_names}")
    
    # Konfiguracije eksperimenata
    configs = create_experiment_configs()
    results = {}
    
    for config in configs:
        result = run_experiment(
            config=config,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            class_names=class_names,
            device=device,
        )
        results[config["name"]] = result
    
    # Cuvanje rezultata svih eksperimenata
    results_path = Path("results") / "all_experiments.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print(f"Results saved to {results_path}")
    print("=" * 60)
    
    # Prikaz summary-a
    print("\nSummary:")
    print("-" * 40)
    for name, result in results.items():
        print(f"{name:20s} | Best Val: {result['best_val_acc']:.4f} | Test: {result['test_acc']:.4f}")


if __name__ == "__main__":
    main()