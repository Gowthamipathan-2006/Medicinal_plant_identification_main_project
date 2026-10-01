"""
CLI Evaluation & Benchmark Script for Tri-Branch Medicinal Plant Identification
Evaluates Tri-Branch model on test split, generates classification metrics,
confusion matrix, and hardware latency benchmark.
"""

import os
import json
import argparse
import torch
from typing import Dict, Any

from src.utils.helpers import load_config, get_device, load_checkpoint
from src.data.dataset import create_dataloaders
from src.models.hybrid_classifier import TriHybridMedicinalClassifier
from src.training.metrics import compute_classification_metrics, benchmark_model_speed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Tri-Branch Classifier")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_hybrid_model.pt", help="Model checkpoint path")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Root path to dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    if device.type == "cpu":
        torch.set_num_threads(min(16, os.cpu_count() or 4))
    data_dir = args.data_dir or config.get("data", {}).get("dataset_dir", "dataset")
    
    print("=" * 75)
    print("  TRI-BRANCH EXPLAINABLE MEDICINAL PLANT IDENTIFICATION: EVALUATION")
    print("  Backbones: Swin-T + VMamba-T (State Space) + MaxViT-T (Multi-Axis)")
    print("=" * 75)
    
    # Load dataset test split
    _, _, test_loader, class_to_idx = create_dataloaders(
        data_dir=data_dir,
        img_size=config.get("data", {}).get("img_size", 224),
        batch_size=config.get("data", {}).get("batch_size", 32)
    )
    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]
    
    # Instantiate model
    model = TriHybridMedicinalClassifier(
        num_classes=len(class_names),
        fusion_type=config.get("model", {}).get("fusion_type", "tri_cross_attention"),
        fusion_dim=config.get("model", {}).get("fusion_dim", 512),
        pretrained=False
    ).to(device)
    
    # Load checkpoint if exists
    if os.path.exists(args.checkpoint):
        print(f"[INFO] Loading checkpoint from: {args.checkpoint}")
        load_checkpoint(args.checkpoint, model)
    else:
        print(f"[WARNING] Checkpoint '{args.checkpoint}' not found. Evaluating with current weights.")
        
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.numpy().tolist())
            
    # Metrics
    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    speed = benchmark_model_speed(model, device=device)
    
    print("\n" + "=" * 60)
    print("                 ACADEMIC PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"  Test Accuracy:        {metrics['accuracy']*100:6.2f}%")
    print(f"  Macro Precision:      {metrics['macro_precision']*100:6.2f}%")
    print(f"  Macro Recall:         {metrics['macro_recall']*100:6.2f}%")
    print(f"  Macro F1-Score:       {metrics['macro_f1']*100:6.2f}%")
    print(f"  Weighted F1-Score:    {metrics['weighted_f1']*100:6.2f}%")
    print("-" * 60)
    print(f"  Inference Latency:    {speed['latency_ms']} ms / image")
    print(f"  Throughput:           {speed['throughput_fps']} FPS")
    print(f"  Total Parameters:     {speed['total_parameters_million']} Million")
    print("=" * 60 + "\n")
    
    # Save JSON report
    os.makedirs("outputs", exist_ok=True)
    report_path = os.path.join("outputs", "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "architecture": "Tri-Branch (Swin-T + VMamba-T + MaxViT-T)",
            "metrics": metrics,
            "benchmark": speed,
            "classes": class_names
        }, f, indent=2)
        
    print(f"[SUCCESS] Detailed evaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()
