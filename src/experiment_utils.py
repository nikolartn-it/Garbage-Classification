"""
Utility funkcije za logovanje eksperimenata.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch


class ExperimentLogger:
    """
    Logger za pracenje eksperimenata.
    
    Loguje:
    - Hiperparametre
    - Metrike po epohama
    - Konfiguraciju modela
    - Vreme trajanja
    """
    
    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        logs_dir: Path,
    ):
        self.experiment_name = experiment_name
        self.config = config
        self.logs_dir = logs_dir / experiment_name
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Istorija metrika
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        
        # Vreme pocetka
        self.start_time = datetime.now()
        
        # Cuvanje konfiguracije
        self._save_config()
    
    def _save_config(self) -> None:
        """Cuva konfiguraciju eksperimenta."""
        config_path = self.logs_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
    ) -> None:
        """Loguje jednu epohu."""
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        
        # Cuvanje u CSV
        self._save_history()
    
    def _save_history(self) -> None:
        """Cuva istoriju treninga u CSV."""
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.logs_dir / "history.csv", index=False)
    
    def save_model(
        self,
        model: torch.nn.Module,
        epoch: int,
        val_acc: float,
    ) -> None:
        """Cuva model."""
        model_path = self.logs_dir / f"model_epoch_{epoch}_acc_{val_acc:.4f}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
            "config": self.config,
        }, model_path)
    
    def save_results(self, test_results: Dict[str, Any]) -> None:
        """Cuva konacne rezultate."""
        results = {
            "experiment_name": self.experiment_name,
            "config": self.config,
            "history": self.history,
            "test_results": test_results,
            "duration": str(datetime.now() - self.start_time),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(self.logs_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    def get_best_val_acc(self) -> float:
        """Vraca najbolju validation accuracy."""
        return max(self.history["val_acc"]) if self.history["val_acc"] else 0.0


def create_experiment_configs() -> List[Dict[str, Any]]:
    """
    Kreira 5 razlicitih konfiguracija za eksperimente.
    
    Returns:
        List of config dictionaries
    """
    configs = [
        {
            "name": "baseline",
            "image_size": 224,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "num_epochs": 30,
            "model": "cnn_baseline",
            "augmentation": True,
            "dropout": 0.5,
        },
        {
            "name": "larger_model",
            "image_size": 224,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "num_epochs": 30,
            "model": "cnn_large",
            "augmentation": True,
            "dropout": 0.5,
            "filters": [64, 128, 256, 512],
        },
        {
            "name": "high_dropout",
            "image_size": 224,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "num_epochs": 30,
            "model": "cnn_baseline",
            "augmentation": True,
            "dropout": 0.7,
        },
        {
            "name": "no_augmentation",
            "image_size": 224,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "num_epochs": 30,
            "model": "cnn_baseline",
            "augmentation": False,
            "dropout": 0.5,
        },
        {
            "name": "higher_lr",
            "image_size": 224,
            "batch_size": 32,
            "learning_rate": 5e-3,
            "weight_decay": 1e-4,
            "num_epochs": 30,
            "model": "cnn_baseline",
            "augmentation": True,
            "dropout": 0.5,
        },
    ]
    
    return configs


def save_experiment_results(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Cuva rezultate svih eksperimenata u jedan CSV fajl.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for exp_name, exp_data in results.items():
        row = {
            "experiment": exp_name,
            "best_val_acc": exp_data.get("best_val_acc", 0),
            "test_acc": exp_data.get("test_acc", 0),
            "test_loss": exp_data.get("test_loss", 0),
            "duration": exp_data.get("duration", ""),
        }
        # Dodaj hiperparametre
        for key, value in exp_data.get("config", {}).items():
            row[f"config_{key}"] = value
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "all_experiments.csv", index=False)
    print(f"✓ Results saved to {output_dir / 'all_experiments.csv'}")