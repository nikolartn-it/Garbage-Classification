"""
Podela dataset-a na train, validation i test skupove.
"""

import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


SEED = 42
N_SPLITS = 5


def discover_images(dataset_root: Path) -> pd.DataFrame:
    """
    Pronalazi sve slike u dataset-u i vraća DataFrame.
    """
    records = []
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for class_dir in dataset_root.iterdir():
        if not class_dir.is_dir():
            continue

        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({
                    "filepath": str(image_path),
                    "filename": image_path.name,
                    "class": class_dir.name,
                })

    return pd.DataFrame(records)


def create_class_mapping(dataframe: pd.DataFrame) -> dict:
    """Kreira mapping između naziva klase i brojčane labele."""
    classes = sorted(dataframe["class"].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def create_splits(
    dataframe: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Kreira train, validation i test splitove.
    """
    # Prvo odvojimo test
    train_val_df, test_df = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_state,
        stratify=dataframe["class"],
    )

    # Zatim odvojimo validation iz train_val
    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val_df["class"],
    )

    return train_df, val_df, test_df


def create_stratified_folds(
    dataframe: pd.DataFrame,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Kreira stratifikovanu K-fold cross-validation podelu.
    """
    dataframe = dataframe.copy()
    dataframe["fold"] = -1

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    for fold_idx, (_, val_indices) in enumerate(
        splitter.split(dataframe["filename"], dataframe["class"])
    ):
        dataframe.loc[dataframe.index[val_indices], "fold"] = fold_idx

    return dataframe


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Čuva splitove kao CSV fajlove.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"✓ Splits saved to {output_dir}")
    print(f"  Train: {len(train_df)}")
    print(f"  Val: {len(val_df)}")
    print(f"  Test: {len(test_df)}")


def load_splits(splits_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Učitava splitove iz CSV fajlova.
    """
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    return train_df, val_df, test_df


# ============================================
# GLAVNA FUNKCIJA ZA KREIRANJE SPLITOVA
# ============================================

def main():
    """Kreira i čuva splitove za dataset."""
    # Putanje
    project_root = Path(__file__).parent.parent
    dataset_root = project_root / "data" / "raw" / "Garbage classification" / "Garbage classification"
    splits_dir = project_root / "data" / "splits"

    print("=" * 50)
    print("Kreiranje splitova za Garbage Classification dataset")
    print("=" * 50)

    # 1. Pronalazak slika
    print("\n[1/4] Pronalazim slike...")
    df = discover_images(dataset_root)
    print(f"Pronađeno {len(df)} slika")

    # 2. Mapiranje klasa
    print("\n[2/4] Mapiranje klasa...")
    class_to_idx, idx_to_class = create_class_mapping(df)
    print(f"Klase: {class_to_idx}")

    # 3. Kreiranje splitova
    print("\n[3/4] Kreiranje train/val/test splitova...")
    train_df, val_df, test_df = create_splits(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 4. Čuvanje splitova
    print("\n[4/4] Čuvanje splitova...")
    save_splits(train_df, val_df, test_df, splits_dir)

    print("\n✅ Gotovo!")


if __name__ == "__main__":
    main()