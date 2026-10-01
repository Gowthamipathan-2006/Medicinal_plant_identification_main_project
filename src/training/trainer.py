"""
Modular Training Engine for Hybrid EfficientNetV2 + Swin Transformer
Features:
- AdamW optimizer with Cosine Annealing learning rate schedule
- Early stopping based on validation Macro F1 / Accuracy
- Automatic checkpoint serialization
- Training history logging for academic report generation and UI graphing
"""

import os
import time
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
from sklearn.metrics import f1_score

from src.utils.helpers import save_checkpoint


class HybridTrainer:
    """
    End-to-End Trainer for Hybrid Foliar Classifier.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_dir: str = "checkpoints"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device or next(model.parameters()).device
        self.config = config or {}
        self.checkpoint_dir = checkpoint_dir
        
        self.model.to(self.device)
        
        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "lr": []
        }
        
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Runs one full training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            # Gradient clipping for stable transformer training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += images.size(0)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total_samples*100:.1f}%"})
            
        avg_loss = total_loss / max(1, total_samples)
        avg_acc = correct / max(1, total_samples)
        return avg_loss, avg_acc
        
    def evaluate(self) -> Tuple[float, float, float]:
        """Evaluates model on validation loader."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        total_samples = 0
        
        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item() * images.size(0)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                total_samples += images.size(0)
                
        avg_loss = total_loss / max(1, total_samples)
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        avg_acc = correct / max(1, total_samples)
        macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
        
        return avg_loss, avg_acc, macro_f1
        
    def fit(self, num_epochs: int = 10, patience: int = 4) -> Dict[str, Any]:
        """
        Executes complete training loop with early stopping and checkpointing.
        """
        best_val_f1 = -1.0
        patience_counter = 0
        best_epoch = 0
        start_time = time.time()
        
        print(f"\n[INFO] Starting training: {num_epochs} epochs | Device: {self.device}")
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc, val_f1 = self.evaluate()
            
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                self.scheduler.step()
                
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["lr"].append(current_lr)
            
            print(
                f"Epoch {epoch+1:02d}/{num_epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.2f}% | "
                f"Val F1: {val_f1*100:5.2f}% | LR: {current_lr:.6f}"
            )
            
            # Checkpoint best model
            is_best = val_f1 > best_val_f1
            if is_best:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                patience_counter = 0
                save_checkpoint(
                    state={
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_val_f1": best_val_f1,
                        "val_acc": val_acc,
                        "history": self.history
                    },
                    is_best=True,
                    checkpoint_dir=self.checkpoint_dir
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[EARLY STOPPING] Validation F1 has not improved for {patience} epochs. Terminating at Epoch {epoch+1}.")
                    break
                    
        elapsed = time.time() - start_time
        print(f"\n[INFO] Training complete in {elapsed/60:.2f} mins. Best Val F1: {best_val_f1*100:.2f}% at Epoch {best_epoch}.\n")
        
        return {
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "total_epochs_trained": len(self.history["train_loss"]),
            "history": self.history
        }
