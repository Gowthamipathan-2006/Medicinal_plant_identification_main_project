"""
PyTorch Dataset and DataLoader Utilities for Medicinal Plant Leaves
Supports dynamic class discovery, botanical data augmentation, and custom splits.
"""

import os
from typing import Tuple, List, Dict, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_transforms(
    img_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns data transforms for training (with botanical augmentation)
    and validation/testing/inference.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, eval_transform


class MedicinalPlantDataset(Dataset):
    """
    PyTorch Dataset for medicinal plant leaf images.
    Auto-discovers classes from directory structure or accepts explicit class mappings.
    """
    def __init__(
        self,
        root_dir: str,
        split: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Resolve path
        if split:
            target_path = os.path.join(root_dir, split)
            if not os.path.exists(target_path):
                # Fallback to root if split directory does not exist
                target_path = root_dir
        else:
            target_path = root_dir
            
        self.target_path = target_path
        
        # Discover classes
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
            self.classes = sorted(list(class_to_idx.keys()))
        else:
            discovered = [
                d for d in os.listdir(target_path)
                if os.path.isdir(os.path.join(target_path, d)) and not d.startswith(".")
            ]
            discovered.sort()
            self.classes = discovered
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}
        
        # Index samples
        self.samples: List[Tuple[str, int]] = []
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = os.path.join(target_path, cls_name)
            if not os.path.exists(cls_dir):
                continue
            for root, _, files in os.walk(cls_dir):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in valid_exts:
                        full_path = os.path.join(root, f)
                        self.samples.append((full_path, cls_idx))
                        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label, path


def create_dataloaders(
    data_dir: str = "dataset",
    img_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Builds train, validation, and test dataloaders with automatic class alignment
    and stratified class distribution.
    """
    import random
    from collections import defaultdict
    
    train_tf, eval_tf = get_transforms(img_size=img_size)
    
    # Check if split subfolders exist
    has_splits = all(os.path.exists(os.path.join(data_dir, s)) for s in ["train", "val", "test"])
    
    if has_splits:
        train_ds = MedicinalPlantDataset(data_dir, split="train", transform=train_tf)
        class_to_idx = train_ds.class_to_idx
        val_ds = MedicinalPlantDataset(data_dir, split="val", transform=eval_tf, class_to_idx=class_to_idx)
        test_ds = MedicinalPlantDataset(data_dir, split="test", transform=eval_tf, class_to_idx=class_to_idx)
    else:
        # Load entire directory and perform stratified train/val/test split
        full_ds = MedicinalPlantDataset(data_dir, transform=None)
        class_to_idx = full_ds.class_to_idx
        
        # Group indices by class label
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(full_ds.samples):
            class_indices[label].append(idx)
            
        rng = random.Random(42)
        train_indices, val_indices, test_indices = [], [], []
        
        for label, idxs in class_indices.items():
            rng.shuffle(idxs)
            n = len(idxs)
            n_train = max(1, int(0.70 * n))
            n_val = max(1, int(0.15 * n))
            
            train_indices.extend(idxs[:n_train])
            val_indices.extend(idxs[n_train:n_train + n_val])
            test_indices.extend(idxs[n_train + n_val:])
            
        class TransformSubset(Dataset):
            def __init__(self, full_dataset, indices, transform):
                self.full_dataset = full_dataset
                self.indices = indices
                self.transform = transform
                
            def __len__(self):
                return len(self.indices)
                
            def __getitem__(self, i):
                orig_idx = self.indices[i]
                img, label, path = self.full_dataset[orig_idx]
                if self.transform is not None:
                    img = self.transform(img)
                return img, label, path
                
        train_ds = TransformSubset(full_ds, train_indices, train_tf)
        val_ds = TransformSubset(full_ds, val_indices, eval_tf)
        test_ds = TransformSubset(full_ds, test_indices, eval_tf)
        
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader, class_to_idx
