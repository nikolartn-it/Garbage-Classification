"""
Vizualizacija rezultata svih eksperimenata.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Ucitava rezultate svih eksperimenata."""
    rows = []
    
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        results_file = exp_dir / "results.json"
        if not results_file.exists():
            continue
        
        with open(results_file, "r") as f:
            data = json.load(f)
        
        row = {
            "experiment": data["experiment_name"],
            "best_val_acc": max(data["history"]["val_acc"]),
            "final_train_acc": data["history"]["train_acc"][-1],
            "final_val_acc": data["history"]["val_acc"][-1],
            "duration": data["duration"],
        }
        
        # Dodaj hiperparametre
        for key, value in data["config"].items():
            row[f"config_{key}"] = value
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, save_dir: Path) -> None:
    """Plotuje uporedne grafike."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Uporedi validation accuracy
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df["experiment"], df["best_val_acc"])
    plt.ylim(0, 1)
    plt.title("Best Validation Accuracy per Experiment")
    plt.ylabel("Accuracy")
    plt.xlabel("Experiment")
    
    # Dodaj vrednosti na vrh stubova
    for bar, val in zip(bars, df["best_val_acc"]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f"{val:.3f}", ha="center", va="bottom")
    
    plt.tight_layout()
    plt.savefig(save_dir / "comparison_val_acc.png", dpi=300)
    plt.close()
    
    # 2. Uporedi konfiguracije
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Learning rate
    ax = axes[0, 0]
    df_sorted = df.sort_values("config_learning_rate")
    ax.bar(df_sorted["experiment"], df_sorted["best_val_acc"])
    ax.set_title("Accuracy vs Learning Rate")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Best Val Acc")
    
    # Dropout
    ax = axes[0, 1]
    df_sorted = df.sort_values("config_dropout")
    ax.bar(df_sorted["experiment"], df_sorted["best_val_acc"])
    ax.set_title("Accuracy vs Dropout")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Best Val Acc")
    
    # Augmentation
    ax = axes[1, 0]
    aug_df = df[df["config_augmentation"] == True]
    no_aug_df = df[df["config_augmentation"] == False]
    ax.bar(["With Aug", "Without Aug"], 
           [aug_df["best_val_acc"].mean(), no_aug_df["best_val_acc"].mean()])
    ax.set_title("Accuracy with/without Augmentation")
    ax.set_ylabel("Average Best Val Acc")
    
    # Model size
    ax = axes[1, 1]
    # Ovo ce raditi ako imamo razlicite modele
    if "config_model" in df.columns:
        sns.boxplot(data=df, x="config_model", y="best_val_acc", ax=ax)
        ax.set_title("Accuracy by Model Type")
        ax.set_xlabel("Model")
        ax.set_ylabel("Best Val Acc")
    
    plt.tight_layout()
    plt.savefig(save_dir / "comparison_all.png", dpi=300)
    plt.close()
    
    print(f"✓ Plots saved to {save_dir}")


def main():
    """Glavna funkcija."""
    results_dir = Path("results")
    plots_dir = results_dir / "comparison_plots"
    
    if not results_dir.exists():
        print("❌ No results found. Run experiments first!")
        return
    
    print("Loading results...")
    df = load_all_results(results_dir)
    
    if df.empty:
        print("❌ No experiment results found!")
        return
    
    print("\nResults summary:")
    print(df[["experiment", "best_val_acc", "duration"]].to_string(index=False))
    
    print("\nGenerating plots...")
    plot_comparison(df, plots_dir)
    
    print(f"\n✅ Done! Check {plots_dir}")


if __name__ == "__main__":
    main()