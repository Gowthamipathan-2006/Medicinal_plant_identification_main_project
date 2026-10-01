"""
Tri-Branch Explainable AI Heatmap Visualizer & Overlay Rendering Engine
Converts normalized CAM attributions to color heatmaps, performs alpha blending,
and generates side-by-side comparative 5-panel publication figures:
[Original Specimen | Swin-T CAM | VMamba SSM CAM | MaxViT CAM | Tri-Fused Consensus]
"""

import io
import base64
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def apply_colormap_overlay(
    image_rgb: np.ndarray,
    cam_heatmap: np.ndarray,
    colormap: str = "turbo",
    alpha: float = 0.55
) -> np.ndarray:
    """
    Blends a normalized 2D CAM heatmap [0, 1] onto an RGB image [0, 255].
    """
    H, W = image_rgb.shape[:2]
    
    # Ensure CAM matches image dimensions
    if cam_heatmap.shape[:2] != (H, W):
        cam_heatmap = cv2.resize(cam_heatmap, (W, H))
        
    cam_uint8 = np.uint8(255 * np.clip(cam_heatmap, 0.0, 1.0))
    
    cv_maps = {
        "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "inferno": cv2.COLORMAP_INFERNO
    }
    cv_cmap = cv_maps.get(colormap.lower(), cv2.COLORMAP_TURBO)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv_cmap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    blended = np.clip(
        alpha * heatmap_rgb.astype(np.float32) + (1.0 - alpha) * image_rgb.astype(np.float32),
        0, 255
    ).astype(np.uint8)
    
    return blended


def create_side_by_side_comparison(
    image_rgb: np.ndarray,
    explanations: Dict[str, Any],
    species_name: str,
    colormap: str = "turbo",
    alpha: float = 0.55,
    save_path: Optional[str] = None
) -> Image.Image:
    """
    Generates an academic 5-panel comparative figure:
    [1] Original Specimen | [2] Swin-T CAM | [3] VMamba SSM CAM | [4] MaxViT Multi-Axis CAM | [5] Tri-Fused Consensus
    """
    swin_overlay = apply_colormap_overlay(image_rgb, explanations["swin_cam"], colormap, alpha)
    vmamba_overlay = apply_colormap_overlay(image_rgb, explanations["vmamba_cam"], colormap, alpha)
    maxvit_overlay = apply_colormap_overlay(image_rgb, explanations["maxvit_cam"], colormap, alpha)
    fused_overlay = apply_colormap_overlay(image_rgb, explanations["fused_cam"], colormap, alpha)
    
    w_swin = explanations.get("swin_weight", 1/3) * 100
    w_vm = explanations.get("vmamba_weight", 1/3) * 100
    w_mv = explanations.get("maxvit_weight", 1/3) * 100
    f_drop = explanations.get("faithfulness_drop_percent", 0.0)
    
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8), dpi=150)
    fig.patch.set_facecolor("#0b1311")  # Rich dark botanical backdrop
    
    panels = [
        ("Original Specimen", image_rgb, "Natural Leaf Specimen"),
        (f"Swin-T LayerCAM ({w_swin:.1f}%)", swin_overlay, "Shifted Window Morphology"),
        (f"VMamba SSM CAM ({w_vm:.1f}%)", vmamba_overlay, "Continuous 2D Vein Scanning"),
        (f"MaxViT Multi-Axis ({w_mv:.1f}%)", maxvit_overlay, "Block Texture + Grid Symmetry"),
        ("Tri-Fused Consensus", fused_overlay, f"Faithfulness Drop: {f_drop:.1f}%")
    ]
    
    for ax, (title, img_data, subtitle) in zip(axes, panels):
        ax.imshow(img_data)
        ax.set_title(title, fontsize=11, fontweight="bold", color="#F3F4F6", pad=8)
        ax.set_xlabel(subtitle, fontsize=9, color="#9CA3AF", labelpad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#10B981" if "Tri-Fused" in title else "#1f2937")
            spine.set_linewidth(2.0 if "Tri-Fused" in title else 1.2)
            
    fig.suptitle(
        f"Botanical Diagnosis: {species_name.replace('_', ' ')}  |  Confidence: {explanations['confidence']*100:.2f}%  |  Tri-Branch Synergy",
        fontsize=13, fontweight="bold", color="#10B981", y=0.98
    )
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    
    return Image.open(buf)


def numpy_to_base64_png(image_rgb: np.ndarray) -> str:
    """Encodes a uint8 RGB numpy array to base64 PNG string."""
    pil_img = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"
