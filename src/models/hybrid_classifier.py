"""
Tri-Branch Hybrid Deep Learning Classifier: Swin-T + VMamba-T + MaxViT-T
Synergizes:
- Shifted Window Multi-Head Attention (Swin-T) for boundary geometry & margin serrations
- 2D Selective State Space Directional Transitions (VMamba-T) for continuous venation propagation
- Multi-Axis Block & Dilated Grid Attention (MaxViT-T) for dual-scale trichome textures and leaf symmetry
Fused via Tri-Branch Multi-Axis Cross-Attention & Dynamic Softmax Routing (Tri-ACF).
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn

from src.models.backbones import SwinTransformerBackbone, VMambaBackbone, MaxViTBackbone
from src.models.fusion import build_fusion_module


class TriHybridMedicinalClassifier(nn.Module):
    """
    Novel Tri-Branch Architecture for Explainable Medicinal Foliar Identification.
    Unifies Swin-T, VMamba-T, and MaxViT-T via Tri-Cross-Attention Fusion.
    """
    def __init__(
        self,
        num_classes: int = 40,
        fusion_type: str = "tri_cross_attention",
        fusion_dim: int = 512,
        dropout: float = 0.3,
        pretrained: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        
        # 1. Tri-Backbones
        self.swin_branch = SwinTransformerBackbone(pretrained=pretrained)
        self.vmamba_branch = VMambaBackbone()
        self.maxvit_branch = MaxViTBackbone(pretrained=pretrained)
        
        # 2. Tri-Branch Feature Fusion
        self.fusion = build_fusion_module(
            fusion_type=fusion_type,
            swin_dim=self.swin_branch.out_dim,      # 768
            vmamba_dim=self.vmamba_branch.out_dim,  # 768
            maxvit_dim=self.maxvit_branch.out_dim,  # 512
            fusion_dim=fusion_dim                   # 512
        )
        
        # 3. Botanical Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.fusion.out_dim),
            nn.Dropout(p=dropout),
            nn.Linear(self.fusion.out_dim, 256),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # Intermediate activation hooks for Multi-Branch XAI
        self.swin_activations: Optional[torch.Tensor] = None
        self.vmamba_activations: Optional[torch.Tensor] = None
        self.maxvit_activations: Optional[torch.Tensor] = None
        
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through all 3 branches, cross-attention fusion, and classification head.
        """
        # Branch 1: Swin-T
        f_swin_spatial, f_swin_pooled = self.swin_branch(x)
        self.swin_activations = f_swin_spatial
        
        # Branch 2: VMamba-T (State Space)
        f_vmamba_spatial, f_vmamba_pooled = self.vmamba_branch(x)
        self.vmamba_activations = f_vmamba_spatial
        
        # Branch 3: MaxViT-T (Multi-Axis)
        f_maxvit_spatial, f_maxvit_pooled = self.maxvit_branch(x)
        self.maxvit_activations = f_maxvit_spatial
        
        # Tri-Branch Fusion
        fused_features, fusion_metadata = self.fusion(
            f_swin_spatial, f_swin_pooled,
            f_vmamba_spatial, f_vmamba_pooled,
            f_maxvit_spatial, f_maxvit_pooled
        )
        
        # Classification Logits
        logits = self.classifier(fused_features)
        
        if return_features:
            metadata = {
                "fusion_metadata": fusion_metadata,
                "swin_pooled": f_swin_pooled,
                "vmamba_pooled": f_vmamba_pooled,
                "maxvit_pooled": f_maxvit_pooled,
                "fused_features": fused_features
            }
            return logits, metadata
            
        return logits


# Aliases for backward compatibility
HybridMedicinalClassifier = TriHybridMedicinalClassifier


def create_model(
    num_classes: int = 40,
    config: Optional[Dict[str, Any]] = None,
    pretrained: bool = True
) -> TriHybridMedicinalClassifier:
    """Instantiate Tri-Hybrid model using configuration dictionary or defaults."""
    if config is not None:
        model_cfg = config.get("model", {})
        return TriHybridMedicinalClassifier(
            num_classes=num_classes,
            fusion_type=model_cfg.get("fusion_type", "tri_cross_attention"),
            fusion_dim=model_cfg.get("fusion_dim", 512),
            dropout=model_cfg.get("dropout", 0.3),
            pretrained=pretrained
        )
    return TriHybridMedicinalClassifier(num_classes=num_classes, pretrained=pretrained)
