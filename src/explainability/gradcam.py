"""
Tri-Branch Explainable AI (Tri-XAI) Suite for Swin-T + VMamba-T + MaxViT-T
Provides:
1. Swin-T LayerCAM (Hierarchical Shifted-Window Attention for leaf contours & serrations)
2. VMamba-T State-CAM / Saliency (Continuous 2D State Space directional venation attribution)
3. MaxViT-T Multi-Axis CAM (Dense Block micro-textures + Dilated Grid global symmetry)
4. Tri-Fused Dynamic Consensus Map (Weighted by dynamic Softmax routing gates)
5. Quantitative Faithfulness Evaluation (% Confidence Drop under Salient Masking)
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


class TriHybridExplainer:
    """
    Unified Tri-Branch Explainability Engine.
    Extracts attributions from Swin-T, VMamba-T, and MaxViT-T simultaneously
    and synthesizes a tri-branch consensus saliency map.
    """
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        
        # Hook storage
        self.swin_features: Optional[torch.Tensor] = None
        self.swin_gradients: Optional[torch.Tensor] = None
        
        self.vmamba_features: Optional[torch.Tensor] = None
        self.vmamba_gradients: Optional[torch.Tensor] = None
        
        self.maxvit_features: Optional[torch.Tensor] = None
        self.maxvit_gradients: Optional[torch.Tensor] = None
        
        self._register_hooks()
        
    def _register_hooks(self):
        # 1. Swin-T Final Norm Layer Hook
        def swin_fwd(module, inp, out):
            self.swin_features = out # [B, 7, 7, 768]
        def swin_bwd(module, gin, gout):
            self.swin_gradients = gout[0]
            
        target_swin = self.model.swin_branch.norm
        target_swin.register_forward_hook(swin_fwd)
        target_swin.register_full_backward_hook(swin_bwd)
        
        # 2. VMamba-T Final Stage Hook
        def vmamba_fwd(module, inp, out):
            self.vmamba_features = out # [B, H, W, C]
        def vmamba_bwd(module, gin, gout):
            self.vmamba_gradients = gout[0]
            
        target_vmamba = self.model.vmamba_branch.stages[-1]
        target_vmamba.register_forward_hook(vmamba_fwd)
        target_vmamba.register_full_backward_hook(vmamba_bwd)
        
        # 3. MaxViT-T Final Block Hook
        def maxvit_fwd(module, inp, out):
            self.maxvit_features = out # [B, 512, 7, 7]
        def maxvit_bwd(module, gin, gout):
            self.maxvit_gradients = gout[0]
            
        target_maxvit = self.model.maxvit_branch.blocks[-1]
        target_maxvit.register_forward_hook(maxvit_fwd)
        target_maxvit.register_full_backward_hook(maxvit_bwd)
        
    def generate_explanations(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        tta_tensor: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Computes CAM attributions for all 3 backbones and constructs the tri-fused attribution map.
        Optionally uses tta_tensor (multi-view test-time augmentation) for stabilized confidence.
        """
        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        self.model.zero_grad()
        
        # Forward pass on canonical view
        logits, metadata = self.model(input_tensor, return_features=True)
        probs = F.softmax(logits, dim=1)
        
        # If multi-view TTA is provided, ensemble probabilities
        if tta_tensor is not None:
            tta_t = tta_tensor.to(self.device)
            with torch.no_grad():
                tta_out = self.model(tta_t, return_features=False)
                tta_logits = tta_out[0] if isinstance(tta_out, tuple) else tta_out
                tta_probs = F.softmax(tta_logits, dim=1).mean(dim=0, keepdim=True)
                probs = (probs * 0.45) + (tta_probs * 0.55)

        
        if target_class is None:
            target_class = int(torch.argmax(probs, dim=1).item())
            
        confidence = float(probs[0, target_class].item())
        
        # Backward pass with respect to target class
        target_score = logits[0, target_class]
        target_score.backward(retain_graph=True)

        
        _, _, H, W = input_tensor.shape
        
        # 1. Swin-T LayerCAM
        if self.swin_features is not None and self.swin_gradients is not None:
            swin_w = F.relu(self.swin_gradients)
            swin_cam_t = torch.sum(swin_w * self.swin_features, dim=-1, keepdim=True) # [1, 7, 7, 1]
            swin_cam_t = swin_cam_t.permute(0, 3, 1, 2) # [1, 1, 7, 7]
            swin_cam_t = F.relu(swin_cam_t)
            swin_up = F.interpolate(swin_cam_t, size=(H, W), mode="bicubic", align_corners=False)
            swin_cam = self._min_max_norm(swin_up[0, 0].detach().cpu().numpy())
        else:
            swin_cam = np.zeros((H, W), dtype=np.float32)
            
        # 2. VMamba-T State-CAM
        if self.vmamba_features is not None and self.vmamba_gradients is not None:
            # VMamba features: [B, H_feat, W_feat, C] or [B, C, H_feat, W_feat]
            if self.vmamba_features.ndim == 4 and self.vmamba_features.shape[-1] > self.vmamba_features.shape[1]:
                # [B, H, W, C]
                vm_w = F.relu(self.vmamba_gradients)
                vm_cam_t = torch.sum(vm_w * self.vmamba_features, dim=-1, keepdim=True).permute(0, 3, 1, 2)
            else:
                vm_w = torch.mean(self.vmamba_gradients, dim=(2, 3), keepdim=True)
                vm_cam_t = F.relu(torch.sum(vm_w * self.vmamba_features, dim=1, keepdim=True))
            vm_up = F.interpolate(vm_cam_t, size=(H, W), mode="bilinear", align_corners=False)
            vmamba_cam = self._min_max_norm(vm_up[0, 0].detach().cpu().numpy())
        else:
            vmamba_cam = np.zeros((H, W), dtype=np.float32)
            
        # 3. MaxViT-T Multi-Axis CAM
        if self.maxvit_features is not None and self.maxvit_gradients is not None:
            maxvit_w = torch.mean(self.maxvit_gradients, dim=(2, 3), keepdim=True)
            maxvit_cam_t = F.relu(torch.sum(maxvit_w * self.maxvit_features, dim=1, keepdim=True))
            maxvit_up = F.interpolate(maxvit_cam_t, size=(H, W), mode="bilinear", align_corners=False)
            maxvit_cam = self._min_max_norm(maxvit_up[0, 0].detach().cpu().numpy())
        else:
            maxvit_cam = np.zeros((H, W), dtype=np.float32)
            
        # 4. Extract dynamic routing weights from fusion metadata
        fusion_meta = metadata.get("fusion_metadata", {})
        if "gate_weights" in fusion_meta:
            gw = fusion_meta["gate_weights"][0].detach().cpu().numpy()
            w_swin, w_vmamba, w_maxvit = float(gw[0]), float(gw[1]), float(gw[2])
        else:
            w_swin, w_vmamba, w_maxvit = 1/3, 1/3, 1/3
            
        tot = w_swin + w_vmamba + w_maxvit if (w_swin + w_vmamba + w_maxvit) > 0 else 1.0
        w_swin_n, w_vmamba_n, w_maxvit_n = w_swin / tot, w_vmamba / tot, w_maxvit / tot
        
        # 5. Tri-Fused Consensus Attribution Map
        fused_cam = (w_swin_n * swin_cam) + (w_vmamba_n * vmamba_cam) + (w_maxvit_n * maxvit_cam)
        fused_cam = self._min_max_norm(fused_cam)
        
        # 6. Quantitative Faithfulness Metric (% drop under top 20% salient pixel occlusion)
        faithfulness = self._compute_faithfulness(
            input_tensor.detach(), target_class, confidence, fused_cam, mask_ratio=0.20
        )
        
        return {
            "swin_cam": swin_cam,
            "vmamba_cam": vmamba_cam,
            "maxvit_cam": maxvit_cam,
            "fused_cam": fused_cam,
            "target_class": target_class,
            "confidence": confidence,
            "probs": probs[0].detach().cpu().numpy(),
            "swin_weight": w_swin_n,
            "vmamba_weight": w_vmamba_n,
            "maxvit_weight": w_maxvit_n,
            "faithfulness_drop_percent": faithfulness
        }
        
    @staticmethod
    def _min_max_norm(cam: np.ndarray) -> np.ndarray:
        c_min, c_max = np.min(cam), np.max(cam)
        if c_max - c_min > 1e-8:
            return (cam - c_min) / (c_max - c_min)
        return np.zeros_like(cam)
        
    def _compute_faithfulness(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        orig_confidence: float,
        saliency_map: np.ndarray,
        mask_ratio: float = 0.20
    ) -> float:
        threshold = np.percentile(saliency_map, (1.0 - mask_ratio) * 100)
        mask = (saliency_map >= threshold).astype(np.float32)
        
        mask_tensor = torch.from_numpy(1.0 - mask).to(self.device).unsqueeze(0).unsqueeze(0)
        masked_input = input_tensor * mask_tensor
        
        with torch.no_grad():
            masked_logits = self.model(masked_input)
            masked_probs = F.softmax(masked_logits, dim=1)
            new_confidence = float(masked_probs[0, target_class].item())
            
        drop = ((orig_confidence - new_confidence) / max(1e-6, orig_confidence)) * 100.0
        return max(0.0, float(drop))


# Alias for backward compatibility
HybridExplainer = TriHybridExplainer
