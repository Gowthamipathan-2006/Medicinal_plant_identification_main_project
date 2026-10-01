"""
High-Accuracy End-to-End Training Pipeline for Tri-Branch Medicinal Plant Classifier
Architecture: Swin-T + VMamba-T (VSSM) + MaxViT-T fused with Tri-Cross-Attention (Tri-ACF)
Dataset: 40 Medicinal Plant Classes (5,945 total images: 4,151 train, 860 val, 934 test)
"""

import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

from src.utils.helpers import load_config, set_seed, get_device, save_checkpoint
from src.data.dataset import create_dataloaders
from src.data.plant_database import CLASS_NAMES
from src.models.hybrid_classifier import TriHybridMedicinalClassifier


def main():
    set_seed(42)
    device = get_device()
    threads = min(16, os.cpu_count() or 4)
    torch.set_num_threads(threads)
    print(f"[SYSTEM] PyTorch initialized with {threads} CPU worker threads. Device: {device}")
    
    # 1. Build Dataloaders
    print("[DATA] Loading 40-class medicinal plant leaf dataset...")
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        data_dir="dataset",
        img_size=224,
        batch_size=32,
        num_workers=0
    )
    num_classes = len(class_to_idx)
    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]
    print(f"[DATA] Total classes: {num_classes}")
    print(f"[DATA] Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    
    # 2. Instantiate Tri-Branch Hybrid Model
    print("\n[MODEL] Initializing Tri-Branch Architecture (Swin-T + VMamba-T + MaxViT-T)...")
    model = TriHybridMedicinalClassifier(
        num_classes=num_classes,
        fusion_type="tri_cross_attention",
        fusion_dim=512,
        dropout=0.3,
        pretrained=True
    ).to(device)
    
    # Freeze initial backbone layers for fast convergence and high generalization
    for p in model.swin_branch.features[:-1].parameters():
        p.requires_grad = False
    for p in model.maxvit_branch.stem.parameters():
        p.requires_grad = False
    for p in model.maxvit_branch.blocks[:-1].parameters():
        p.requires_grad = False
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Trainable Parameters: {trainable_params/1e6:.2f}M / Total: {total_params/1e6:.2f}M")
    
    # 3. Training Setup
    epochs = 8
    lr = 0.0004
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": [], "lr": []}
    
    print(f"\n[TRAINING] Starting {epochs} fine-tuning epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs:02d} [Train]", leave=False)
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total*100:.1f}%"})
            
        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)
        
        # Validation
        model.eval()
        v_loss, v_preds, v_labels, v_total = 0.0, [], [], 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                v_loss += loss.item() * images.size(0)
                preds = torch.argmax(logits, dim=1)
                v_preds.extend(preds.cpu().numpy().tolist())
                v_labels.extend(labels.cpu().numpy().tolist())
                v_total += images.size(0)
                
        val_loss = v_loss / max(1, v_total)
        val_acc = accuracy_score(v_labels, v_preds)
        val_f1 = f1_score(v_labels, v_preds, average="macro", zero_division=0)
        
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)
        
        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.2f}% | "
            f"Val F1: {val_f1*100:5.2f}% | LR: {current_lr:.6f}"
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            save_checkpoint(
                state={
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "best_val_f1": best_val_f1,
                    "class_names": class_names,
                    "class_to_idx": class_to_idx,
                    "history": history
                },
                is_best=True,
                checkpoint_dir="checkpoints"
            )
            print(f"  --> Saved new best checkpoint with Val Acc: {val_acc*100:.2f}% (Macro F1: {val_f1*100:.2f}%)")
            
    print(f"\n[INFO] Fine-tuning finished in {(time.time()-start_time)/60:.2f} mins.")
    
    # 4. Final Evaluation on Hold-Out Test Split (934 images)
    print("\n[EVAL] Running final evaluation on unseen test split...")
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            test_preds.extend(preds.cpu().numpy().tolist())
            test_labels.extend(labels.numpy().tolist())
            
    test_acc = accuracy_score(test_labels, test_preds)
    test_prec = precision_score(test_labels, test_preds, average="macro", zero_division=0)
    test_rec = recall_score(test_labels, test_preds, average="macro", zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    test_wf1 = f1_score(test_labels, test_preds, average="weighted", zero_division=0)
    
    print("=" * 65)
    print("           FINAL TEST SET EVALUATION RESULTS (934 Images)")
    print("=" * 65)
    print(f"  Test Accuracy:        {test_acc*100:6.2f}%")
    print(f"  Macro Precision:      {test_prec*100:6.2f}%")
    print(f"  Macro Recall:         {test_rec*100:6.2f}%")
    print(f"  Macro F1-Score:       {test_f1*100:6.2f}%")
    print(f"  Weighted F1-Score:    {test_wf1*100:6.2f}%")
    print("=" * 65)
    
    # Save Evaluation Report
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
                "total_evaluated_samples": len(test_labels)
            },
            "classes": class_names,
            "best_val_acc": best_val_acc,
            "best_val_f1": best_val_f1
        }, f, indent=2)
        
    print(f"[SUCCESS] Test report persisted to '{report_path}'.")


if __name__ == "__main__":
    main()
