"""
Utility funkcije za logovanje eksperimenata.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class ExperimentLogger:
    """
    Logger za pracenje jednog eksperimenta.
    
    Cuva:
    - config.json (konfiguracija)
    - history.csv (metrike po epohama)
    - results.json (konacni rezultati)
    """

    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = datetime.now()
        self.config = None

    def save_config(self, config: Dict[str, Any]) -> None:
        """Cuva konfiguraciju eksperimenta."""
        self.config = config
        config_path = self.log_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def save_history(self, history: Dict[str, List[float]]) -> None:
        """Cuva istoriju treninga (po epohama) u CSV."""
        history_df = pd.DataFrame(history)
        history_df.to_csv(self.log_dir / "history.csv", index=False)

    def save_results(self, result: Dict[str, Any]) -> None:
        """Cuva konacne rezultate eksperimenta."""
        # Sacuvaj config odvojeno ako postoji u result-u
        if "config" in result and self.config is None:
            self.save_config(result["config"])

        # Sacuvaj history ako postoji
        if "history" in result:
            self.save_history(result["history"])

        # Sacuvaj rezultate
        results_with_meta = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            **{k: v for k, v in result.items() if k != "config"},
        }

        with open(self.log_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results_with_meta, f, indent=2, ensure_ascii=False)


def create_experiment_configs() -> List[Dict[str, Any]]:
    """
    Kreira 5 razlicitih konfiguracija za eksperimente.
    """
    configs = [
       # {
       #     "name": "baseline",
       #     "image_size": 224,
       #     "batch_size": 32,
       #     "learning_rate": 1e-3,
       #     "weight_decay": 1e-4,
       #     "num_epochs": 30,
       #     "augmentation": True,
       #     "dropout": 0.5,
       # },
        {
            "name": "larger_model",
            "image_size": 224,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "num_epochs": 15,
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
            "num_epochs": 15,
            "augmentation": True,
            "dropout": 0.7,
        },
        {
            "name": "no_augmentation",
            "image_size": 224,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "num_epochs": 15,
            "augmentation": False,
            "dropout": 0.5,
        },
        {
            "name": "higher_lr",
            "image_size": 224,
            "batch_size": 32,
            "learning_rate": 5e-3,
            "weight_decay": 1e-4,
            "num_epochs": 15,
            "augmentation": True,
            "dropout": 0.5,
        },
    ]
    return configs