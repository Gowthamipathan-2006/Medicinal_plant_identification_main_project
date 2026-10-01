"""
Robust Domain-Generalization Training Pipeline for Tri-Branch Botanical Identification
Solves out-of-distribution / in-the-wild generalization for random smartphone and web photos by:
1. Multi-view domain-generalizing botanical augmentations (Scale, Aspect-Preserved Crop, Rotation, Color Jitter, Flips)
2. Aspect-ratio preserving tensor extraction across Swin-T, VMamba-T, and MaxViT-T
3. Label Smoothing Cross-Entropy regularization (epsilon = 0.1)
4. Cosine Annealing optimization of Tri-Cross-Attention Fusion & Dynamic Softmax Routing
"""

import os
import time
import json
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from src.utils.helpers import set_seed, get_device, save_checkpoint
from src.data.dataset import MedicinalPlantDataset
from src.models.backbones import SwinTransformerBackbone, VMambaBackbone, MaxViTBackbone
from src.models.fusion import build_fusion_module
from src.models.hybrid_classifier import TriHybridMedicinalClassifier


def get_robust_augmentation_transforms(
    img_size: int = 224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
):
    norm = transforms.Normalize(mean=mean, std=std)
    
    # 1. Canonical Aspect-Preserved Center View
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        norm
    ])
    
    # 2. In-the-Wild Foliar View A: Multi-Scale Crop + Jitter + Rotation
    aug_transform_a = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04),
        transforms.ToTensor(),
        norm
    ])
    
    # 3. In-the-Wild Foliar View B: Orientation Shift + Mild Perspective
    aug_transform_b = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        norm
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        norm
    ])
    
    return [base_transform, aug_transform_a, aug_transform_b], eval_transform


def extract_features_for_transform(dataset, transform, swin, mamba, maxvit, device, batch_size=64, desc="Extracting"):
    dataset.transform = transform
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    swin_sps, swin_ps = [], []
    mamba_sps, mamba_ps = [], []
    maxvit_sps, maxvit_ps = [], []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc=desc, leave=False):
            images = images.to(device)
            
            s_sp, s_p = swin(images)
            m_sp, m_p = mamba(images)
            v_sp, v_p = maxvit(images)
            
            swin_sps.append(s_sp.cpu())
            swin_ps.append(s_p.cpu())
            mamba_sps.append(m_sp.cpu())
            mamba_ps.append(m_p.cpu())
            maxvit_sps.append(v_sp.cpu())
            maxvit_ps.append(v_p.cpu())
            all_labels.append(labels)
            
    return {
        "swin_sp": torch.cat(swin_sps, dim=0),
        "swin_p": torch.cat(swin_ps, dim=0),
        "mamba_sp": torch.cat(mamba_sps, dim=0),
        "mamba_p": torch.cat(mamba_ps, dim=0),
        "maxvit_sp": torch.cat(maxvit_sps, dim=0),
        "maxvit_p": torch.cat(maxvit_ps, dim=0),
        "labels": torch.cat(all_labels, dim=0)
    }


def main():
    set_seed(42)
    device = get_device()
    torch.set_num_threads(min(16, os.cpu_count() or 4))
    
    print("=" * 80)
    print("  ROBUST DOMAIN-GENERALIZATION TRAINING: TRI-BRANCH BOTANICAL CLASSIFIER")
    print("  Models: Swin-T + VMamba-T + MaxViT-T | 40 Species Classes | In-The-Wild Robust")
    print("=" * 80)
    
    train_transforms, eval_transform = get_robust_augmentation_transforms(img_size=224)
    
    # Datasets
    train_dataset = MedicinalPlantDataset(root_dir="dataset", split="train")
    val_dataset = MedicinalPlantDataset(root_dir="dataset", split="val", transform=eval_transform, class_to_idx=train_dataset.class_to_idx)
    test_dataset = MedicinalPlantDataset(root_dir="dataset", split="test", transform=eval_transform, class_to_idx=train_dataset.class_to_idx)
    
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    
    os.makedirs("outputs/cache_robust", exist_ok=True)
    train_cache_file = os.path.join("outputs", "cache_robust", "features_train_augmented.pt")
    val_cache_file = os.path.join("outputs", "cache_robust", "features_val.pt")
    test_cache_file = os.path.join("outputs", "cache_robust", "features_test.pt")
    
    # Initialize Backbones
    swin = SwinTransformerBackbone(pretrained=True).to(device).eval()
    mamba = VMambaBackbone().to(device).eval()
    maxvit = MaxViTBackbone(pretrained=True).to(device).eval()
    
    # 1. Feature Extraction with Multi-View Augmentations
    if os.path.exists(train_cache_file):
        print(f"[CACHE] Loading cached robust train features from '{train_cache_file}'...")
        train_feats = torch.load(train_cache_file, map_location="cpu", weights_only=False)
    else:
        print(f"[CACHE] Extracting 3 Multi-View Augmented representations for Train ({len(train_dataset)} x 3 = {len(train_dataset)*3} samples)...")
        t0 = time.time()
        view_feats = []
        for i, tfm in enumerate(train_transforms):
            vf = extract_features_for_transform(
                train_dataset, tfm, swin, mamba, maxvit, device,
                desc=f"Extracting Train View {i+1}/3"
            )
            view_feats.append(vf)
            
        train_feats = {
            "swin_sp": torch.cat([v["swin_sp"] for v in view_feats], dim=0),
            "swin_p": torch.cat([v["swin_p"] for v in view_feats], dim=0),
            "mamba_sp": torch.cat([v["mamba_sp"] for v in view_feats], dim=0),
            "mamba_p": torch.cat([v["mamba_p"] for v in view_feats], dim=0),
            "maxvit_sp": torch.cat([v["maxvit_sp"] for v in view_feats], dim=0),
            "maxvit_p": torch.cat([v["maxvit_p"] for v in view_feats], dim=0),
            "labels": torch.cat([v["labels"] for v in view_feats], dim=0)
        }
        torch.save(train_feats, train_cache_file)
        print(f"[CACHE] Saved robust train features to '{train_cache_file}' in {time.time()-t0:.1f}s.")
        
    if os.path.exists(val_cache_file):
        val_feats = torch.load(val_cache_file, map_location="cpu", weights_only=False)
    else:
        val_feats = extract_features_for_transform(val_dataset, eval_transform, swin, mamba, maxvit, device, desc="Extracting Val")
        torch.save(val_feats, val_cache_file)
        
    if os.path.exists(test_cache_file):
        test_feats = torch.load(test_cache_file, map_location="cpu", weights_only=False)
    else:
        test_feats = extract_features_for_transform(test_dataset, eval_transform, swin, mamba, maxvit, device, desc="Extracting Test")
        torch.save(test_feats, test_cache_file)
        
    # 2. Build Head Trainer
    class RobustTriHeadTrainer(nn.Module):
        def __init__(self, num_classes=40, fusion_dim=512, dropout=0.35):
            super().__init__()
            self.fusion = build_fusion_module(
                fusion_type="tri_cross_attention",
                swin_dim=768,
                vmamba_dim=768,
                maxvit_dim=512,
                fusion_dim=fusion_dim
            )

            self.classifier = nn.Sequential(
                nn.LayerNorm(self.fusion.out_dim),
                nn.Dropout(p=dropout),
                nn.Linear(self.fusion.out_dim, 256),
                nn.GELU(),
                nn.Dropout(p=dropout * 0.5),
                nn.Linear(256, num_classes)
            )
            
        def forward(self, s_sp, s_p, m_sp, m_p, v_sp, v_p):
            fused, meta = self.fusion(s_sp, s_p, m_sp, m_p, v_sp, v_p)
            logits = self.classifier(fused)
            return logits, meta

    head_model = RobustTriHeadTrainer(num_classes=num_classes, fusion_dim=512, dropout=0.35).to(device)
    
    # 3. Optimization with Label Smoothing & Cosine Schedule
    epochs = 40
    batch_size = 128
    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
    optimizer = AdamW(head_model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    N_train = train_feats["labels"].size(0)
    print(f"\n[TRAINING] Training on {N_train} Augmented Foliar Representations for {epochs} epochs...")
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_head_state = None
    t_train_start = time.time()
    
    for epoch in range(epochs):
        head_model.train()
        indices = torch.randperm(N_train)
        total_loss, correct, total = 0.0, 0, 0
        
        for start_idx in range(0, N_train, batch_size):
            end_idx = min(start_idx + batch_size, N_train)
            batch_idx = indices[start_idx:end_idx]
            
            s_sp = train_feats["swin_sp"][batch_idx].to(device)
            s_p = train_feats["swin_p"][batch_idx].to(device)
            m_sp = train_feats["mamba_sp"][batch_idx].to(device)
            m_p = train_feats["mamba_p"][batch_idx].to(device)
            v_sp = train_feats["maxvit_sp"][batch_idx].to(device)
            v_p = train_feats["maxvit_p"][batch_idx].to(device)
            labels = train_feats["labels"][batch_idx].to(device)
            
            optimizer.zero_grad()
            logits, _ = head_model(s_sp, s_p, m_sp, m_p, v_sp, v_p)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * len(batch_idx)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(batch_idx)
            
        train_loss = total_loss / total
        train_acc = correct / total
        scheduler.step()
        
        # Validation
        head_model.eval()
        with torch.no_grad():
            v_s_sp = val_feats["swin_sp"].to(device)
            v_s_p = val_feats["swin_p"].to(device)
            v_m_sp = val_feats["mamba_sp"].to(device)
            v_m_p = val_feats["mamba_p"].to(device)
            v_v_sp = val_feats["maxvit_sp"].to(device)
            v_v_p = val_feats["maxvit_p"].to(device)
            v_labels = val_feats["labels"].to(device)
            
            v_logits, _ = head_model(v_s_sp, v_s_p, v_m_sp, v_m_p, v_v_sp, v_v_p)
            v_loss = criterion(v_logits, v_labels).item()
            v_preds = torch.argmax(v_logits, dim=1).cpu().numpy()
            v_targets = v_labels.cpu().numpy()
            
            val_acc = accuracy_score(v_targets, v_preds)
            val_f1 = f1_score(v_targets, v_preds, average="macro", zero_division=0)
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_head_state = {k: v.cpu() for k, v in head_model.state_dict().items()}
            
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch+1:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}% | "
                f"Val Loss: {v_loss:.4f} | Val Acc: {val_acc*100:5.2f}% | Val F1: {val_f1*100:5.2f}%"
            )
            
    print(f"[TRAINING] Completed in {time.time()-t_train_start:.1f}s. Best Val Acc: {best_val_acc*100:.2f}%")
    
    # 4. Save to Unified TriHybridMedicinalClassifier Checkpoint
    full_model = TriHybridMedicinalClassifier(
        num_classes=num_classes,
        fusion_type="tri_cross_attention",
        fusion_dim=512,
        dropout=0.35,
        pretrained=True
    ).to(device)
    
    head_model.load_state_dict({k: v.to(device) for k, v in best_head_state.items()})
    full_model.fusion.load_state_dict(head_model.fusion.state_dict())
    full_model.classifier.load_state_dict(head_model.classifier.state_dict())
    
    os.makedirs("checkpoints", exist_ok=True)
    save_checkpoint(
        state={
            "epoch": epochs,
            "model_state_dict": full_model.state_dict(),
            "best_val_acc": best_val_acc,
            "best_val_f1": best_val_f1,
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "architecture": "Tri-Branch (Swin-T + VMamba-T + MaxViT-T)"
        },
        is_best=True,
        checkpoint_dir="checkpoints"
    )
    print("\n[CHECKPOINT] Successfully saved robust domain-generalized weights to 'checkpoints/best_hybrid_model.pt'.")
    
    # 5. Evaluate on Hold-out Test Split
    print("\n[EVALUATION] Evaluating on Hold-Out Test Set (934 Images)...")
    head_model.eval()
    with torch.no_grad():
        t_s_sp = test_feats["swin_sp"].to(device)
        t_s_p = test_feats["swin_p"].to(device)
        t_m_sp = test_feats["mamba_sp"].to(device)
        t_m_p = test_feats["mamba_p"].to(device)
        t_v_sp = test_feats["maxvit_sp"].to(device)
        t_v_p = test_feats["maxvit_p"].to(device)
        t_labels = test_feats["labels"].to(device)
        
        t_logits, _ = head_model(t_s_sp, t_s_p, t_m_sp, t_m_p, t_v_sp, t_v_p)
        t_preds = torch.argmax(t_logits, dim=1).cpu().numpy()
        t_targets = t_labels.cpu().numpy()
        
    test_acc = accuracy_score(t_targets, t_preds)
    test_prec = precision_score(t_targets, t_preds, average="macro", zero_division=0)
    test_rec = recall_score(t_targets, t_preds, average="macro", zero_division=0)
    test_f1 = f1_score(t_targets, t_preds, average="macro", zero_division=0)
    test_wf1 = f1_score(t_targets, t_preds, average="weighted", zero_division=0)
    
    print("\n" + "=" * 70)
    print("           FINAL TEST SET EVALUATION RESULTS (934 Images)")
    print("=" * 70)
    print(f"  Test Top-1 Accuracy:  {test_acc*100:6.2f}%")
    print(f"  Macro Precision:      {test_prec*100:6.2f}%")
    print(f"  Macro Recall:         {test_rec*100:6.2f}%")
    print(f"  Macro F1-Score:       {test_f1*100:6.2f}%")
    print(f"  Weighted F1-Score:    {test_wf1*100:6.2f}%")
    print("=" * 70 + "\n")
    
    os.makedirs("outputs", exist_ok=True)
    report_path = os.path.join("outputs", "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "architecture": "Tri-Branch (Swin-T + VMamba-T + MaxViT-T)",
            "metrics": {
                "accuracy": test_acc,
                "macro_precision": test_prec,
                "macro_recall": test_rec,
                "macro_f1": test_f1,
                "weighted_f1": test_wf1,
                "total_evaluated_samples": len(t_targets)
            },
            "classes": class_names,
            "best_val_acc": best_val_acc,
            "best_val_f1": best_val_f1
        }, f, indent=2)


if __name__ == "__main__":
    main()
