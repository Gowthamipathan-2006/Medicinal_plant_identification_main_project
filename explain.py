"""
CLI Explainability Script for Tri-Branch Foliar Identification
Generates 5-panel XAI heatmaps for any leaf image:
[Original Specimen | Swin-T CAM | VMamba SSM CAM | MaxViT Multi-Axis CAM | Tri-Fused Consensus]
"""

import os
import argparse
from PIL import Image
import torch

from src.utils.helpers import load_config, get_device, load_checkpoint, preprocess_image_for_inference
from src.models.hybrid_classifier import TriHybridMedicinalClassifier
from src.explainability.gradcam import TriHybridExplainer
from src.explainability.visualizer import create_side_by_side_comparison
from src.data.plant_database import get_plant_monograph, CLASS_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Tri-Branch XAI Visualizations for Leaf Image")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image file")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_hybrid_model.pt", help="Path to checkpoint")
    parser.add_argument("--colormap", type=str, default="turbo", choices=["turbo", "jet", "viridis", "inferno"])
    parser.add_argument("--alpha", type=float, default=0.55, help="Alpha transparency overlay")
    parser.add_argument("--output", type=str, default=None, help="Save path for 5-panel figure")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Input leaf image not found at: {args.image}")
        
    raw_img = Image.open(args.image)
    tensor, img_rgb = preprocess_image_for_inference(
        raw_img, img_size=config.get("data", {}).get("img_size", 224)
    )
    
    # Initialize Tri-Branch model
    model = TriHybridMedicinalClassifier(
        num_classes=len(CLASS_NAMES),
        fusion_type=config.get("model", {}).get("fusion_type", "tri_cross_attention"),
        fusion_dim=config.get("model", {}).get("fusion_dim", 512),
        pretrained=False
    ).to(device)
    
    if os.path.exists(args.checkpoint):
        load_checkpoint(args.checkpoint, model)
    else:
        print(f"[INFO] Using initialized weights (checkpoint '{args.checkpoint}' not found).")
        
    explainer = TriHybridExplainer(model, device=device)
    explanations = explainer.generate_explanations(tensor)
    
    pred_idx = explanations["target_class"]
    pred_species = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"Class_{pred_idx}"
    monograph = get_plant_monograph(pred_species)
    
    print("\n" + "=" * 70)
    print("            BOTANICAL TRI-BRANCH XAI DIAGNOSIS RESULT")
    print("=" * 70)
    print(f"  Identified Species:    {monograph['scientific_name']} ({monograph['common_name']})")
    print(f"  Botanical Family:      {monograph['botanical_family']}")
    print(f"  Diagnostic Confidence: {explanations['confidence']*100:.2f}%")
    print(f"  Swin-T Weight:         {explanations['swin_weight']*100:.1f}% (Boundary Morphology)")
    print(f"  VMamba Weight:         {explanations['vmamba_weight']*100:.1f}% (Directional Venation)")
    print(f"  MaxViT-T Weight:       {explanations['maxvit_weight']*100:.1f}% (Micro-Texture & Grid)")
    print(f"  Faithfulness Drop:     {explanations.get('faithfulness_drop_percent', 0.0):.1f}%")
    print("-" * 70)
    print(f"  Active Phytochemicals: {', '.join(monograph['active_phytochemicals'][:3])}")
    print(f"  Therapeutic Actions:   {', '.join(monograph['pharmacological_actions'][:2])}")
    print("=" * 70 + "\n")
    
    # Save 5-panel figure
    os.makedirs("outputs", exist_ok=True)
    out_path = args.output or os.path.join("outputs", f"xai_{pred_species}.png")
    fig_img = create_side_by_side_comparison(
        image_rgb=img_rgb,
        explanations=explanations,
        species_name=pred_species,
        colormap=args.colormap,
        alpha=args.alpha,
        save_path=out_path
    )
    print(f"[SUCCESS] 5-Panel Tri-Branch comparative visualization saved to: {out_path}\n")


if __name__ == "__main__":
    main()
