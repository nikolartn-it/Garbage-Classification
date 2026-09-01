from pathlib import Path
from typing import Tuple

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(
        self,
        image_paths: list,
        labels: list,
        image_size: int = 224,
        augment: bool = False,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size

        # Transformacije za trening (sa augmentacijom)
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        # Transformacije za validaciju/test (bez augmentacije)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

#FIND DATASET

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def find_dataset_root(project_root: Path) -> Path:
    """
    Pronalazi folder koji sadrži direktorijume sa klasama.
    """

    possible_roots = [
        project_root / "data" / "raw" / "Garbage classification" / "Garbage classification",
        project_root / "data" / "raw" / "Garbage classification",
    ]

    for root in possible_roots:
        if not root.exists():
            continue

        class_dirs = [
            path for path in root.iterdir()
            if path.is_dir()
        ]

        if class_dirs:
            return root

    raise FileNotFoundError(
        "Dataset folder nije pronađen. "
        "Proverite strukturu data/raw/."
    )


def discover_images(dataset_root: Path) -> pd.DataFrame:
    """
    Pronalaženje svih validnih fajlova slika i formiranje DataFrame-a.
    """

    records = []

    class_dirs = sorted(
        path for path in dataset_root.iterdir()
        if path.is_dir()
    )

    for class_dir in class_dirs:

        for image_path in class_dir.rglob("*"):

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            records.append(
                {
                    "filepath": str(image_path),
                    "filename": image_path.name,
                    "class": class_dir.name,
                    "extension": image_path.suffix.lower(),
                }
            )

    if not records:
        raise ValueError(
            "Nisu pronađene slike u datasetu."
        )

    return pd.DataFrame(records)



#VALIDACIJA SLIKA
def validate_images(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Proverava da li se slike mogu otvoriti i učitati.

    Returns
    -------
    valid_df:
        DataFrame sa validnim slikama.

    invalid_df:
        DataFrame sa neispravnim slikama i opisom greške.
    """

    valid_records = []
    invalid_records = []

    for _, row in dataframe.iterrows():

        filepath = row["filepath"]

        try:
            with Image.open(filepath) as image:
                image.verify()

            valid_records.append(row.to_dict())

        except Exception as error:

            invalid_records.append(
                {
                    **row.to_dict(),
                    "error": str(error),
                }
            )

    valid_df = pd.DataFrame(valid_records)
    invalid_df = pd.DataFrame(invalid_records)

    return valid_df, invalid_df



#OSNOVNA TRANSFORMACIJA
def load_image(
    filepath: str,
    image_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Učitava sliku, konvertuje je u RGB i menja dimenziju.

    Rezultat je NumPy niz oblika:

        (height, width, channels)
    """

    with Image.open(filepath) as image:

        image = image.convert("RGB")
        image = image.resize(image_size)

        array = np.asarray(
            image,
            dtype=np.float32,
        )

    return array

#NORMALIZACIJA
def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizuje vrednosti piksela iz [0, 255] u [0, 1].
    """

    return image / 255.0

#KOMPLETNA F-JA PRIMENE
def preprocess_image(
    filepath: str,
    image_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Kompletna obrada jedne slike.
    """

    image = load_image(
        filepath=filepath,
        image_size=image_size,
    )

    image = normalize_image(image)

    return image
# ============================================
# DATALOADERI
# ============================================

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = True,
) -> dict:
    """
    Kreira DataLoader-ove za trening, validaciju i test skupove.

    Returns:
        dict: {
            'train': DataLoader,
            'val': DataLoader,
            'test': DataLoader
        }
    """
    # Priprema podataka
    train_paths = train_df["filepath"].tolist()
    train_labels = train_df["label"].tolist()

    val_paths = val_df["filepath"].tolist()
    val_labels = val_df["label"].tolist()

    test_paths = test_df["filepath"].tolist()
    test_labels = test_df["label"].tolist()

    # Kreiranje dataset-ova
    train_dataset = ImageDataset(
        train_paths,
        train_labels,
        image_size=image_size,
        augment=augment,
    )

    val_dataset = ImageDataset(
        val_paths,
        val_labels,
        image_size=image_size,
        augment=False,
    )

    test_dataset = ImageDataset(
        test_paths,
        test_labels,
        image_size=image_size,
        augment=False,
    )

    # Kreiranje DataLoader-ova
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def get_class_names(dataset_root: Path) -> list:
    """Vraća listu naziva klasa na osnovu direktorijuma."""
    class_dirs = sorted([
        path for path in dataset_root.iterdir()
        if path.is_dir()
    ])
    return [path.name for path in class_dirs]