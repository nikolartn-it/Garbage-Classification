"""
Retroaktivno logovanje postojecih eksperimenata u MLFlow.

Cita results/all_experiments.json i upisuje sve u MLFlow bazu (mlflow.db).
"""

import json
from pathlib import Path

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Putanja do SQLite baze
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")


def log_all_experiments():
    """Ucitava sve rezultate i loguje ih u MLFlow."""
    results_path = RESULTS_DIR / "all_experiments.json"

    if not results_path.exists():
        print(f"[ERROR] Ne postoji: {results_path}")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    # Postavi (ili kreiraj) MLFlow eksperiment
    mlflow.set_experiment("Garbage-Classification")

    print(f"Pronadjeno {len(all_results)} eksperimenata\n")

    for name, result in all_results.items():
        if result.get("status") != "completed":
            print(f"[SKIP] {name} - status: {result.get('status')}")
            continue

        print(f"[LOG] {name}")

        with mlflow.start_run(run_name=name):
            # --- Parametri ---
            config = result.get("config", {})
            for key, value in config.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(key, value)
                elif isinstance(value, list):
                    mlflow.log_param(key, str(value))

            # --- Metrike ---
            mlflow.log_metric("best_val_acc", result.get("best_val_acc", 0))
            mlflow.log_metric("test_acc", result.get("test_acc", 0))
            mlflow.log_metric("total_params", result.get("total_params", 0))

            if "elapsed_seconds" in result:
                mlflow.log_metric("elapsed_seconds", result["elapsed_seconds"])

            # --- Per-class metrike ---
            per_class = result.get("per_class", {})
            for class_name, metrics in per_class.items():
                mlflow.log_metric(f"{class_name}_precision", metrics["precision"])
                mlflow.log_metric(f"{class_name}_recall", metrics["recall"])
                mlflow.log_metric(f"{class_name}_f1", metrics["f1"])

            # --- Istorija po epohama ---
            history = result.get("history", {})
            if history.get("train_loss"):
                for epoch in range(len(history["train_loss"])):
                    mlflow.log_metric("train_loss", history["train_loss"][epoch], step=epoch)
                    mlflow.log_metric("train_acc", history["train_acc"][epoch], step=epoch)
                    mlflow.log_metric("val_loss", history["val_loss"][epoch], step=epoch)
                    mlflow.log_metric("val_acc", history["val_acc"][epoch], step=epoch)

        print(f"  -> test_acc: {result.get('test_acc'):.4f}, "
              f"best_val_acc: {result.get('best_val_acc'):.4f}")

    print("\n" + "=" * 50)
    print("Svi eksperimenti su logovani u MLFlow!")
    print(f"Baza: {MLFLOW_DB}")
    print("Pokreni: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("Otvori:  http://localhost:5000")
    print("=" * 50)


if __name__ == "__main__":
    log_all_experiments()