"""
Utility helpers for configuration loading, seed reproducibility, device selection,
and model checkpoint persistence.
"""

import os
import yaml
import random
import numpy as np
import torch
from PIL import Image
from typing import Dict, Any, Optional, Tuple
from torchvision import transforms


def load_config(config_path: str = "configs/default_config.yaml") -> Dict[str, Any]:
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """Sets random seeds across random, numpy, and torch for scientific reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Selects CUDA if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool = False,
    checkpoint_dir: str = "checkpoints",
    filename: str = "last_checkpoint.pt"
) -> str:
    """Saves model checkpoint and optionally marks best."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_hybrid_model.pt")
        torch.save(state, best_path)
    return filepath


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """Loads model weights and optimizer state from checkpoint with shape alignment."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=get_device(), weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    
    # Filter matching parameters only
    model_state = model.state_dict()
    filtered_state = {}
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v
            
    model.load_state_dict(filtered_state, strict=False)
    if optimizer and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception:
            pass
    return checkpoint


def auto_foliar_crop(image: Image.Image, padding_ratio: float = 0.04) -> Image.Image:
    """
    Detects pure white or uniform empty margins (common in stock/web photos)
    and tightly crops around the actual foliar specimen so that leaf margins
    and fine serrations maintain maximum resolution for the vision backbones.
    """
    img_rgb = image.convert("RGB")
    img_np = np.array(img_rgb)
    h, w, _ = img_np.shape
    
    # Fast background detection: near-white (> 235 in all channels) or near-black (< 18)
    is_white_bg = (img_np[:, :, 0] > 235) & (img_np[:, :, 1] > 235) & (img_np[:, :, 2] > 235)
    is_black_bg = (img_np[:, :, 0] < 18) & (img_np[:, :, 1] < 18) & (img_np[:, :, 2] < 18)
    is_uniform_bg = is_white_bg | is_black_bg
    
    foliar_mask = ~is_uniform_bg
    
    # If substantial foliar area is found and background is present
    if (np.sum(foliar_mask) > 0.015 * h * w) and (np.sum(is_uniform_bg) > 0.08 * h * w):
        y_indices, x_indices = np.where(foliar_mask)
        ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))
        xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))
        
        # Add padding
        pad_y = int((ymax - ymin) * padding_ratio)
        pad_x = int((xmax - xmin) * padding_ratio)
        ymin = max(0, ymin - pad_y)
        ymax = min(h, ymax + pad_y)
        xmin = max(0, xmin - pad_x)
        xmax = min(w, xmax + pad_x)
        
        if (xmax - xmin) > 20 and (ymax - ymin) > 20:
            return img_rgb.crop((xmin, ymin, xmax, ymax))
            
    return img_rgb


def preserve_aspect_ratio_resize(image: Image.Image, img_size: int = 224) -> Image.Image:
    """
    Intelligently resizes and crops image to (img_size, img_size) while strictly preserving
    the natural geometric aspect ratio of the leaf blade, margins, and venation angles.
    """
    image = auto_foliar_crop(image)
    w, h = image.size
    
    if w == h:
        return image.resize((img_size, img_size), Image.Resampling.BICUBIC)
        
    # Scale such that smaller dimension is img_size (aspect-ratio preserving)
    scale = img_size / min(w, h)
    new_w = max(img_size, int(round(w * scale)))
    new_h = max(img_size, int(round(h * scale)))
    scaled = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    
    # Center-crop to img_size x img_size
    left = (new_w - img_size) // 2
    top = (new_h - img_size) // 2
    return scaled.crop((left, top, left + img_size, top + img_size))


def preprocess_image_for_inference(
    image: Image.Image,
    img_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocesses a PIL Image into:
    1. A normalized torch tensor [1, 3, H, W] with preserved aspect ratio
    2. A clean RGB numpy array [H, W, 3] for overlay visualization.
    """
    img_processed = preserve_aspect_ratio_resize(image, img_size=img_size)
    img_rgb = np.array(img_processed)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    tensor = transform(img_processed).unsqueeze(0)
    
    return tensor, img_rgb


def create_tta_batch_for_inference(
    image: Image.Image,
    img_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Generates a 6-view Test-Time Augmentation (TTA) batch for robust outside/in-the-wild photos:
    1. Standard auto-cropped aspect-preserved view
    2. Horizontally mirrored view
    3. Vertically mirrored view
    4. 0.85x Zoom center-crop focusing on leaflet serrations
    5. Contrast-enhanced normalized view
    6. Brightness adjusted view
    """
    cropped_img = auto_foliar_crop(image)
    base_img = preserve_aspect_ratio_resize(cropped_img, img_size=img_size)
    img_rgb = np.array(base_img)
    
    norm = transforms.Normalize(mean=mean, std=std)
    to_tensor = transforms.ToTensor()
    
    # View 1: Base aspect-preserved
    t1 = norm(to_tensor(base_img))
    
    # View 2: Horizontal flip
    flip_h = base_img.transpose(Image.FLIP_LEFT_RIGHT)
    t2 = norm(to_tensor(flip_h))
    
    # View 3: Vertical flip
    flip_v = base_img.transpose(Image.FLIP_TOP_BOTTOM)
    t3 = norm(to_tensor(flip_v))
    
    # View 4: Serration zoom crop (0.82x crop of foliar region)
    w, h = cropped_img.size
    crop_w = int(w * 0.82)
    crop_h = int(h * 0.82)
    cl = max(0, (w - crop_w) // 2)
    ct = max(0, (h - crop_h) // 2)
    zoom_cropped = cropped_img.crop((cl, ct, cl + crop_w, ct + crop_h))
    zoom_resized = preserve_aspect_ratio_resize(zoom_cropped, img_size=img_size)
    t4 = norm(to_tensor(zoom_resized))
    
    # View 5: Contrast normalization
    from PIL import ImageEnhance
    enhancer_c = ImageEnhance.Contrast(base_img)
    contrast_img = enhancer_c.enhance(1.15)
    t5 = norm(to_tensor(contrast_img))
    
    # View 6: Brightness adjusted
    enhancer_b = ImageEnhance.Brightness(base_img)
    bright_img = enhancer_b.enhance(1.08)
    t6 = norm(to_tensor(bright_img))
    
    batch_tensor = torch.stack([t1, t2, t3, t4, t5, t6], dim=0) # [6, 3, 224, 224]
    return batch_tensor, img_rgb


def compute_uncertainty_metrics(probs: np.ndarray, min_confidence_threshold: float = 0.45) -> Dict[str, Any]:
    """
    Computes Out-of-Distribution (OOD) and uncertainty metrics:
    - Top-1 Confidence
    - Normalized Shannon Entropy [0, 1]
    - Margin between Top-1 and Top-2
    - Flag for uncertain / out-of-catalog specimen
    """
    num_classes = len(probs)
    top_indices = np.argsort(probs)[::-1]
    top1_prob = float(probs[top_indices[0]])
    top2_prob = float(probs[top_indices[1]]) if num_classes > 1 else 0.0
    margin = top1_prob - top2_prob
    
    # Normalized Shannon entropy in [0, 1]
    eps = 1e-12
    entropy = -float(np.sum(probs * np.log2(probs + eps))) / float(np.log2(max(num_classes, 2)))
    entropy = max(0.0, min(1.0, entropy))
    
    is_uncertain = bool((top1_prob < min_confidence_threshold) or (entropy > 0.65) or (margin < 0.10))
    
    if top1_prob >= 0.75 and margin >= 0.35:
        tier = "High Confidence"
        status_color = "success"
    elif top1_prob >= min_confidence_threshold and not is_uncertain:
        tier = "Moderate Confidence"
        status_color = "warning"
    else:
        tier = "Low / Out-of-Catalog Candidate"
        status_color = "danger"
        
    warning_msg = None
    if is_uncertain:
        warning_msg = (
            f"Specimen Uncertainty Advisory (Confidence: {top1_prob*100:.1f}%, Entropy: {entropy:.2f}): "
            "The model detected ambiguous botanical features. This specimen may belong to an uncataloged species "
            "(such as Turmeric / Curcuma longa) or contains heavy background clutter / non-standard lighting."
        )
        
    return {
        "is_uncertain": is_uncertain,
        "confidence_tier": tier,
        "status_color": status_color,
        "top1_confidence": round(top1_prob * 100, 2),
        "top2_confidence": round(top2_prob * 100, 2),
        "prediction_margin": round(margin * 100, 2),
        "normalized_entropy": round(entropy, 3),
        "advisory": warning_msg
    }


