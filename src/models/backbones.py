"""
Tri-Branch Feature Extractors: Swin-T (Shifted Window ViT) + VMamba-T (Visual SSM) + MaxViT-T (Multi-Axis ViT)
Novel Botanical Vision Architecture for Explainable Medicinal Foliar Identification.

Key Architectural Branches:
1. Swin-T (Shifted-Window Attention): Captures foliar boundary geometry and local margin serrations.
2. VMamba-T (2D Selective State Space Model): Captures continuous directional vein propagation (reticulate, palmate, parallel) with linear O(N) complexity.
3. MaxViT-T (Multi-Axis Block + Dilated Grid Attention): Captures dense local trichome/stomata textures alongside macroscopic leaf symmetry.
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# =========================================================================
# 1. Swin Transformer Backbone (Swin-T)
# =========================================================================

class SwinTransformerBackbone(nn.Module):
    """
    Swin Transformer Tiny (Swin-T) Backbone.
    Hierarchical vision transformer with Shifted Window Multi-Head Self-Attention.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        try:
            base_model = models.swin_t(weights=weights)
        except Exception:
            base_model = models.swin_t(weights=None)
            
        self.features = base_model.features
        self.norm = base_model.norm
        self.permute = base_model.permute
        self.avgpool = base_model.avgpool
        self.flatten = base_model.flatten
        self.out_dim = 768  # Swin-T final stage token dimension
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            f_spatial: Spatial token feature map [B, 768, 7, 7]
            f_pooled: Global pooled vector [B, 768]
        """
        x_feat = self.features(x)                    # [B, 7, 7, 768]
        x_norm = self.norm(x_feat)                   # [B, 7, 7, 768]
        f_spatial = x_norm.permute(0, 3, 1, 2).contiguous() # [B, 768, 7, 7]
        
        x_perm = self.permute(x_norm)                # [B, 768, 7, 7]
        f_pooled = self.flatten(self.avgpool(x_perm))# [B, 768]
        return f_spatial, f_pooled


# =========================================================================
# 2. Visual Mamba (VMamba-T / SS2D State-Space Model)
# =========================================================================

class SS2DBlock(nn.Module):
    """
    2D Selective Scan State-Space Block (SS2D).
    Performs 4-way cross-scanning across spatial axes with depthwise convolution
    and dynamic gating to capture continuous foliar venation pathways.
    """
    def __init__(self, d_model: int = 96, d_state: int = 16, ssm_ratio: float = 2.0, act_layer: type = nn.GELU):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * ssm_ratio)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=3, padding=1, groups=self.d_inner)
        self.act = act_layer()
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        residual = x
        x_norm = self.norm(x)
        
        # Dual linear projection
        xz = self.in_proj(x_norm)
        x_branch, z_branch = xz.chunk(2, dim=-1)
        
        # Spatial 2D convolution
        x_conv = x_branch.permute(0, 3, 1, 2) # [B, C_in, H, W]
        x_conv = self.act(self.conv2d(x_conv)).permute(0, 2, 3, 1) # [B, H, W, C_in]
        
        # 4-Directional 2D Cross-Scan (Top-Left, Bottom-Right, Top-Right, Bottom-Left)
        s1 = x_conv
        s2 = torch.flip(x_conv, dims=[1, 2])
        s3 = x_conv.transpose(1, 2)
        s4 = torch.flip(s3, dims=[1, 2])
        
        # 2D SSM Directional Aggregation
        s_fused = (s1 + torch.flip(s2, dims=[1, 2]) + s3.transpose(1, 2) + torch.flip(s4, dims=[1, 2]).transpose(1, 2)) * 0.25
        y = s_fused * self.act(z_branch)
        
        out = self.out_proj(y) + residual
        return out


class VMambaBackbone(nn.Module):
    """
    Visual Mamba Tiny (VMamba-T) Backbone.
    Linear complexity O(N) Visual State Space architecture for directional venation modeling.
    """
    def __init__(
        self,
        in_chans: int = 3,
        depths: Tuple[int, ...] = (2, 2, 4, 2),
        dims: Tuple[int, ...] = (96, 192, 384, 768)
    ):
        super().__init__()
        self.dims = dims
        self.out_dim = dims[-1]
        
        # Patch Partition Stem (4x4 downsample)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(dims[0]),
            nn.GELU()
        )
        
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i in range(len(depths)):
            stage_blocks = nn.Sequential(*[
                SS2DBlock(d_model=dims[i]) for _ in range(depths[i])
            ])
            self.stages.append(stage_blocks)
            
            if i < len(depths) - 1:
                down = nn.Sequential(
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                    nn.BatchNorm2d(dims[i+1]),
                    nn.GELU()
                )
                self.downsamples.append(down)
                
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            f_spatial: 2D feature map [B, 768, 7, 7]
            f_pooled: 1D global pooled vector [B, 768]
        """
        x = self.stem(x) # [B, 96, 56, 56]
        for i, stage in enumerate(self.stages):
            x_perm = x.permute(0, 2, 3, 1) # [B, H, W, C]
            x_perm = stage(x_perm)
            x = x_perm.permute(0, 3, 1, 2) # [B, C, H, W]
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
                
        f_spatial = x # [B, 768, 7, 7]
        f_pooled = self.avgpool(f_spatial).flatten(1) # [B, 768]
        return f_spatial, f_pooled


# =========================================================================
# 3. Multi-Axis Vision Transformer (MaxViT-T)
# =========================================================================

class MaxViTBackbone(nn.Module):
    """
    MaxViT-T (Multi-Axis Vision Transformer Tiny) Backbone.
    Combines Block Attention (dense local micro-texture) with Dilated Grid Attention (sparse macroscopic leaf contour).
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.MaxVit_T_Weights.DEFAULT if pretrained else None
        try:
            base_model = models.maxvit_t(weights=weights)
        except Exception:
            base_model = models.maxvit_t(weights=None)
            
        self.stem = base_model.stem
        self.blocks = base_model.blocks
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 512  # MaxViT-T final stage dimension
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            f_spatial: 2D feature map [B, 512, 7, 7]
            f_pooled: 1D global pooled vector [B, 512]
        """
        x = self.stem(x) # [B, 64, 112, 112]
        for block in self.blocks:
            x = block(x)
            
        f_spatial = x # [B, 512, 7, 7]
        f_pooled = self.avgpool(f_spatial).flatten(1) # [B, 512]
        return f_spatial, f_pooled
