"""
Evaluation Metrics & Benchmark Utilities for Medicinal Foliar Classification
Computes:
- Top-1 and Top-5 Accuracy
- Per-class Precision, Recall, and F1-score
- Macro and Weighted F1-scores
- Confusion Matrix
- Latency and Throughput Benchmarks (FPS / ms per inference)
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import torch
from typing import Optional



def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Computes comprehensive academic evaluation metrics.
    """
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    accuracy = float(accuracy_score(y_true_np, y_pred_np))
    
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_np, y_pred_np, average="macro", zero_division=0
    )
    prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true_np, y_pred_np, average="weighted", zero_division=0
    )
    
    # Per-class metrics
    prec_per_class, rec_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true_np, y_pred_np, labels=list(range(len(class_names))), zero_division=0
    )
    
    per_class_summary = {}
    for i, name in enumerate(class_names):
        per_class_summary[name] = {
            "precision": float(prec_per_class[i]),
            "recall": float(rec_per_class[i]),
            "f1_score": float(f1_per_class[i]),
            "samples": int(support[i])
        }
        
    # Confusion matrix
    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(len(class_names))))
    
    return {
        "accuracy": accuracy,
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weight),
        "per_class": per_class_summary,
        "confusion_matrix": cm.tolist(),
        "total_evaluated_samples": len(y_true)
    }


def benchmark_model_speed(
    model: torch.nn.Module,
    input_size: Tuple[int, int] = (224, 224),
    num_iterations: int = 20,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Measures inference latency (ms) and throughput (frames per second).
    """
    if device is None:
        device = next(model.parameters()).device
        
    dummy_input = torch.randn(1, 3, *input_size, device=device)
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    total_time = time.perf_counter() - start_time
    avg_latency_ms = (total_time / num_iterations) * 1000.0
    throughput_fps = num_iterations / total_time
    
    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "latency_ms": round(avg_latency_ms, 2),
        "throughput_fps": round(throughput_fps, 1),
        "total_parameters_million": round(total_params / 1e6, 2),
        "trainable_parameters_million": round(trainable_params / 1e6, 2)
    }
