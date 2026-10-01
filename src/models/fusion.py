"""
Tri-Branch Feature Fusion Modules for Hybrid Deep Learning Architectures
Synergizes:
1. Swin-T (Shifted Window ViT Tokens)
2. VMamba-T (2D Selective State Space Directional Context)
3. MaxViT-T (Multi-Axis Block + Dilated Grid Representations)

Includes:
1. TriCrossAttentionFusion (Tri-ACF: Tri-Branch Cross-Attention & Dynamic Softmax Routing) - Proposed Novelty
2. TriGatedAdaptiveFusion (Dynamic 3-Way Convex Combination)
3. TriConcatenationFusion (Baseline for ablation studies)
"""

import math
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriConcatenationFusion(nn.Module):
    """
    Baseline feature fusion: Concatenates pooled vectors [f_swin; f_vmamba; f_maxvit]
    and applies linear projection with layer normalization.
    """
    def __init__(
        self,
        swin_dim: int = 768,
        vmamba_dim: int = 768,
        maxvit_dim: int = 512,
        fusion_dim: int = 512
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(swin_dim + vmamba_dim + maxvit_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        self.out_dim = fusion_dim
        
    def forward(
        self,
        f_swin_sp: torch.Tensor,
        f_swin_p: torch.Tensor,
        f_vmamba_sp: torch.Tensor,
        f_vmamba_p: torch.Tensor,
        f_maxvit_sp: torch.Tensor,
        f_maxvit_p: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B = f_swin_p.size(0)
        concat = torch.cat([f_swin_p, f_vmamba_p, f_maxvit_p], dim=1)
        fused = self.proj(concat)
        
        one_third = torch.ones(B, 1, device=fused.device) / 3.0
        return fused, {
            "g_swin": one_third,
            "g_vmamba": one_third,
            "g_maxvit": one_third,
            "gate_weights": torch.cat([one_third, one_third, one_third], dim=1)
        }


class TriGatedAdaptiveFusion(nn.Module):
    """
    Dynamic 3-Way Softmax Gated Fusion:
    Projects all 3 pooled modalities into a shared embedding space and computes
    learnable sample-specific routing weights g = [g_swin, g_vmamba, g_maxvit] sum to 1.
    """
    def __init__(
        self,
        swin_dim: int = 768,
        vmamba_dim: int = 768,
        maxvit_dim: int = 512,
        fusion_dim: int = 512
    ):
        super().__init__()
        self.proj_swin = nn.Sequential(nn.Linear(swin_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.proj_vmamba = nn.Sequential(nn.Linear(vmamba_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.proj_maxvit = nn.Sequential(nn.Linear(maxvit_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        
        self.gate_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 3, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )
        self.out_norm = nn.LayerNorm(fusion_dim)
        self.out_dim = fusion_dim
        
    def forward(
        self,
        f_swin_sp: torch.Tensor,
        f_swin_p: torch.Tensor,
        f_vmamba_sp: torch.Tensor,
        f_vmamba_p: torch.Tensor,
        f_maxvit_sp: torch.Tensor,
        f_maxvit_p: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        h_swin = self.proj_swin(f_swin_p)
        h_vmamba = self.proj_vmamba(f_vmamba_p)
        h_maxvit = self.proj_maxvit(f_maxvit_p)
        
        gate_logits = self.gate_mlp(torch.cat([h_swin, h_vmamba, h_maxvit], dim=1))
        gate_weights = F.softmax(gate_logits, dim=-1) # [B, 3]
        
        g_swin = gate_weights[:, 0:1]
        g_vmamba = gate_weights[:, 1:2]
        g_maxvit = gate_weights[:, 2:3]
        
        fused = g_swin * h_swin + g_vmamba * h_vmamba + g_maxvit * h_maxvit
        fused = self.out_norm(fused)
        
        return fused, {
            "g_swin": g_swin,
            "g_vmamba": g_vmamba,
            "g_maxvit": g_maxvit,
            "gate_weights": gate_weights
        }


class TriCrossAttentionFusion(nn.Module):
    """
    Tri-Branch Cross-Attention & Dynamic Softmax Routing Fusion (Tri-ACF):
    Mathematical Formulation:
    1. Spatial Projections: P_swin, P_vmamba, P_maxvit in R^{B x 49 x d}
    2. Multi-Axis Cross Attention: Swin window tokens query combined VMamba state & MaxViT grid context:
       Attn = Softmax( Q_swin * K_{vmamba,maxvit}^T / sqrt(d) ) * V_{vmamba,maxvit}
    3. Dynamic 3-Way Softmax Gating:
       g = Softmax( W_g [h_swin; h_vmamba; h_maxvit] + b_g ) in R^3
    4. Consensus Synthesis:
       f_fused = LayerNorm( g_1 h_swin + g_2 h_vmamba + g_3 h_maxvit + gamma * h_cross )
    """
    def __init__(
        self,
        swin_dim: int = 768,
        vmamba_dim: int = 768,
        maxvit_dim: int = 512,
        fusion_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        # 1. 1x1 Conv Projections for 2D Spatial Token Maps
        self.conv_swin = nn.Conv2d(swin_dim, fusion_dim, kernel_size=1)
        self.conv_vmamba = nn.Conv2d(vmamba_dim, fusion_dim, kernel_size=1)
        self.conv_maxvit = nn.Conv2d(maxvit_dim, fusion_dim, kernel_size=1)
        
        # 2. Multi-head Cross-Attention Synergy
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_cross = nn.LayerNorm(fusion_dim)
        
        # 3. Pooled 1D Projections
        self.proj_swin = nn.Sequential(nn.Linear(swin_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.proj_vmamba = nn.Sequential(nn.Linear(vmamba_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.proj_maxvit = nn.Sequential(nn.Linear(maxvit_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        
        # 4. Tri-Branch Dynamic Softmax Gating Network
        self.gate_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 3, 128),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 3)
        )
        
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.out_norm = nn.LayerNorm(fusion_dim)
        self.out_dim = fusion_dim
        
    def forward(
        self,
        f_swin_sp: torch.Tensor,
        f_swin_p: torch.Tensor,
        f_vmamba_sp: torch.Tensor,
        f_vmamba_p: torch.Tensor,
        f_maxvit_sp: torch.Tensor,
        f_maxvit_p: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B = f_swin_sp.size(0)
        
        # 1. Spatial Token Projection [B, 49, fusion_dim]
        tokens_swin = self.conv_swin(f_swin_sp).flatten(2).permute(0, 2, 1)
        tokens_vmamba = self.conv_vmamba(f_vmamba_sp).flatten(2).permute(0, 2, 1)
        tokens_maxvit = self.conv_maxvit(f_maxvit_sp).flatten(2).permute(0, 2, 1)
        
        # Unified Cross-Branch Context (VMamba Directional SSM + MaxViT Multi-Axis Grid)
        kv_context = torch.cat([tokens_vmamba, tokens_maxvit], dim=1) # [B, 98, fusion_dim]
        
        # Cross-Attention: Swin queries the joint SSM + Multi-Axis context
        attn_out, attn_weights = self.cross_attn(
            query=tokens_swin,
            key=kv_context,
            value=kv_context
        )
        tokens_enhanced = self.norm_cross(tokens_swin + attn_out)
        h_cross = tokens_enhanced.mean(dim=1) # [B, fusion_dim]
        
        # 2. Pooled Representations
        h_swin = self.proj_swin(f_swin_p)
        h_vmamba = self.proj_vmamba(f_vmamba_p)
        h_maxvit = self.proj_maxvit(f_maxvit_p)
        
        # 3. Dynamic Softmax Routing Gates
        gate_logits = self.gate_mlp(torch.cat([h_swin, h_vmamba, h_maxvit], dim=1))
        gate_weights = F.softmax(gate_logits, dim=-1) # [B, 3]
        
        g_swin = gate_weights[:, 0:1]
        g_vmamba = gate_weights[:, 1:2]
        g_maxvit = gate_weights[:, 2:3]
        
        # 4. Synthesis with learnable cross-attention coefficient gamma
        fused = g_swin * h_swin + g_vmamba * h_vmamba + g_maxvit * h_maxvit + self.gamma * h_cross
        fused = self.out_norm(fused)
        
        metadata = {
            "g_swin": g_swin,
            "g_vmamba": g_vmamba,
            "g_maxvit": g_maxvit,
            "gate_weights": gate_weights,
            "attn_weights": attn_weights,
            "gamma": self.gamma.item()
        }
        return fused, metadata


def build_fusion_module(
    fusion_type: str = "tri_cross_attention",
    swin_dim: int = 768,
    vmamba_dim: int = 768,
    maxvit_dim: int = 512,
    fusion_dim: int = 512
) -> nn.Module:
    """Factory helper to instantiate desired fusion module."""
    if fusion_type in ["tri_cross_attention", "gated_cross_attention"]:
        return TriCrossAttentionFusion(swin_dim, vmamba_dim, maxvit_dim, fusion_dim)
    elif fusion_type in ["tri_gated", "gated"]:
        return TriGatedAdaptiveFusion(swin_dim, vmamba_dim, maxvit_dim, fusion_dim)
    elif fusion_type in ["tri_concat", "concat"]:
        return TriConcatenationFusion(swin_dim, vmamba_dim, maxvit_dim, fusion_dim)
    else:
        raise ValueError(f"Unknown fusion type '{fusion_type}'. Options: 'tri_cross_attention', 'tri_gated', 'tri_concat'")
