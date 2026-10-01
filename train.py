"""
CLI Training Script for Novel Tri-Branch Explainable Medicinal Plant Identification
Runs end-to-end training of the Tri-Branch (Swin-T + VMamba-T + MaxViT-T) architecture
fused via Tri-Branch Cross-Attention & Dynamic Softmax Routing (Tri-ACF).
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.helpers import load_config, set_seed, get_device
from src.data.dataset import create_dataloaders
from src.data.plant_database import CLASS_NAMES
from src.models.hybrid_classifier import TriHybridMedicinalClassifier
from src.training.trainer import HybridTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Tri-Branch Swin-T + VMamba-T + MaxViT-T Classifier")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Root path to leaf dataset containing train/val/test")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--fusion", type=str, default="tri_cross_attention", choices=["tri_cross_attention", "tri_gated", "tri_concat"])
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("project", {}).get("seed", 42))
    device = get_device()
    
    if device.type == "cpu":
        threads = min(16, os.cpu_count() or 4)
        torch.set_num_threads(threads)
        print(f"[PERF] Configured PyTorch with {threads} CPU parallel worker threads.")
    
    data_dir = args.data_dir or config.get("data", {}).get("dataset_dir", "dataset")
    epochs = args.epochs or config.get("training", {}).get("epochs", 10)
    batch_size = args.batch_size or config.get("data", {}).get("batch_size", 32)
    lr = args.lr or config.get("training", {}).get("learning_rate", 0.0003)
    fusion_type = args.fusion or config.get("model", {}).get("fusion_type", "tri_cross_attention")
    
    print("=" * 75)
    print("  TRI-BRANCH EXPLAINABLE MEDICINAL PLANT IDENTIFICATION: TRAINING")
    print("  Architecture: Swin-T + VMamba-T (State Space) + MaxViT-T (Multi-Axis)")
    print("=" * 75)
    print(f"Device: {device}")
    print(f"Dataset directory: {data_dir}")
    print(f"Fusion strategy: {fusion_type}")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | Learning rate: {lr}\n")
    
    # Build Dataloaders
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        data_dir=data_dir,
        img_size=config.get("data", {}).get("img_size", 224),
        batch_size=batch_size
    )
    num_classes = len(class_to_idx)
    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]
    print(f"[INFO] Discovered {num_classes} classes.")
    print(f"[INFO] Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)} | Test samples: {len(test_loader.dataset)}")
    
    # Initialize Tri-Branch Hybrid Model
    model = TriHybridMedicinalClassifier(
        num_classes=num_classes,
        fusion_type=fusion_type,
        fusion_dim=config.get("model", {}).get("fusion_dim", 512),
        dropout=config.get("model", {}).get("dropout", 0.3),
        pretrained=config.get("model", {}).get("use_pretrained", True)
    ).to(device)
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config.get("training", {}).get("weight_decay", 0.01))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.get("training", {}).get("min_lr", 1e-5))
    criterion = nn.CrossEntropyLoss()
    
    # Train
    trainer = HybridTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        config=config,
        checkpoint_dir=config.get("training", {}).get("checkpoint_dir", "checkpoints")
    )
    
    results = trainer.fit(num_epochs=epochs, patience=config.get("training", {}).get("early_stopping_patience", 5))
    
    # Save training metadata & classes
    os.makedirs("outputs", exist_ok=True)
    summary_path = os.path.join("outputs", "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "architecture": "Tri-Branch (Swin-T + VMamba-T + MaxViT-T)",
            "fusion_type": fusion_type,
            "epochs_trained": results["total_epochs_trained"],
            "best_val_f1": results["best_val_f1"],
            "best_epoch": results["best_epoch"],
            "history": results["history"]
        }, f, indent=2)
        
    print(f"\n[SUCCESS] Training summary persisted to '{summary_path}'.")
    print(f"[SUCCESS] Best checkpoint ready at '{config.get('training', {}).get('checkpoint_dir', 'checkpoints')}/best_hybrid_model.pt'.")


if __name__ == "__main__":
    main()
