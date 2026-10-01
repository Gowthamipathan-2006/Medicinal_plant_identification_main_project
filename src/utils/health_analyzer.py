"""
Foliar Health & Phytochemical Purity Analyzer
Computes computer vision tissue segmentation on medicinal foliar specimens:
- Isolates true foliar parenchyma across complex backgrounds (soil, tiles, studio white)
- Quantifies Healthy photosynthetic parenchyma (%)
- Detects Chlorosis / Chlorophyll depletion (%)
- Identifies Necrosis, fungal lesions, pest spots & rot (%)
- Computes Medicinal Harvest Suitability & Purity Grade (Grade A, B, C)
"""

import io
import base64
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import cv2


def analyze_foliar_health(image: Image.Image) -> Dict[str, Any]:
    """
    Performs robust computer vision foliar tissue segmentation and purity grading.
    Returns quantitative health breakdown, purity score, medicinal grade, and visual overlay.
    """
    img_rgb = np.array(image.convert("RGB"))
    h, w, _ = img_rgb.shape
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)
    h_c = hsv[:, :, 0]
    s_c = hsv[:, :, 1]
    v_c = hsv[:, :, 2]
    
    # 1. Accurate Foliar Vegetation Mask
    # Detect green/yellow-green foliage and discriminate against backgrounds (tiles, soil, white paper)
    is_green_veg = ((g >= r - 12) & (g > b + 20)) | ((h_c >= 25) & (h_c <= 95) & (s_c >= 30) & (v_c >= 30))
    # Exclude bright non-saturated studio background
    is_green_veg &= ~((v_c > 225) & (s_c < 30))
    
    # Clean up morphological noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(is_green_veg.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    
    # Fill internal contours to capture interior necrotic spots/lesions
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_leaf = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            cv2.drawContours(filled_leaf, [cnt], -1, 255, -1)
            
    leaf_mask = filled_leaf > 0
    total_leaf_pixels = int(np.sum(leaf_mask))
    if total_leaf_pixels < 100:
        total_leaf_pixels = max(1, int(np.sum(is_green_veg)))
        leaf_mask = is_green_veg if total_leaf_pixels > 50 else np.ones((h, w), dtype=bool)
        total_leaf_pixels = int(np.sum(leaf_mask))

    # 2. Tissue Health Categorization within True Leaf Region
    # A) Healthy Green Parenchyma
    healthy_mask = leaf_mask & is_green_veg & (h_c >= 28) & (h_c <= 92)
    # B) Chlorosis (Yellowing / Chlorophyll fading)
    chlorosis_mask = leaf_mask & ~healthy_mask & (((h_c >= 16) & (h_c < 28) & (s_c >= 25)) | ((g > b + 25) & (r > b + 25)))
    # C) Necrosis / Fungal Blight / Lesions / Pest Damage
    necrosis_mask = leaf_mask & ~healthy_mask & ~chlorosis_mask

    # 3. Calculate Percentages
    healthy_count = int(np.sum(healthy_mask))
    chlorosis_count = int(np.sum(chlorosis_mask))
    necrosis_count = int(np.sum(necrosis_mask))
    
    healthy_pct = (healthy_count / total_leaf_pixels) * 100.0
    chlorosis_pct = (chlorosis_count / total_leaf_pixels) * 100.0
    necrosis_pct = (necrosis_count / total_leaf_pixels) * 100.0
    
    tot_pct = healthy_pct + chlorosis_pct + necrosis_pct
    if tot_pct > 0:
        healthy_pct = round((healthy_pct / tot_pct) * 100.0, 1)
        chlorosis_pct = round((chlorosis_pct / tot_pct) * 100.0, 1)
        necrosis_pct = round((necrosis_pct / tot_pct) * 100.0, 1)
        # Ensure exact 100.0% sum
        diff = 100.0 - (healthy_pct + chlorosis_pct + necrosis_pct)
        healthy_pct = round(healthy_pct + diff, 1)

    # 4. Medicinal Quality Grading & Safety Clearance
    purity_score = max(0.0, min(100.0, healthy_pct - (0.35 * chlorosis_pct) - (1.2 * necrosis_pct)))
    purity_score = round(purity_score, 1)
    
    if healthy_pct >= 78.0 and necrosis_pct <= 10.0:
        grade = "Grade A (Optimal)"
        grade_code = "A"
        is_suitable = True
        status_label = "Optimal Therapeutic Quality"
        status_color = "success"
        recommendation = "Intact active foliar parenchyma with optimal active phytochemical density. Approved for Ayurvedic extracts, Kwatha decoctions, and Churna powders."
    elif healthy_pct >= 58.0 and necrosis_pct <= 24.0:
        grade = "Grade B (Acceptable)"
        grade_code = "B"
        is_suitable = True
        status_label = "Moderate Quality (Botanical Washing Required)"
        status_color = "warning"
        recommendation = "Minor foliar chlorosis or edge wear detected. Approved for hot-water decoctions and medicated oils after thorough botanical cleansing."
    else:
        grade = "Grade C (Unsuitable / Compromised)"
        grade_code = "C"
        is_suitable = False
        status_label = "UNSUITABLE FOR MEDICINE (Safety Lockout)"
        status_color = "danger"
        recommendation = "Severe foliar necrosis or fungal blight detected (>24% tissue damage). Risk of mycotoxin contamination and degraded bioactives. Formulation generation is locked for safety."

    # 5. Visual Health Overlay
    overlay = img_rgb.copy()
    overlay[healthy_mask] = (overlay[healthy_mask] * 0.70 + np.array([34, 197, 94]) * 0.30).astype(np.uint8)
    overlay[chlorosis_mask] = (overlay[chlorosis_mask] * 0.60 + np.array([234, 179, 8]) * 0.40).astype(np.uint8)
    overlay[necrosis_mask] = (overlay[necrosis_mask] * 0.50 + np.array([239, 68, 68]) * 0.50).astype(np.uint8)

    overlay_pil = Image.fromarray(overlay)
    buf = io.BytesIO()
    overlay_pil.save(buf, format="PNG")
    health_overlay_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "purity_score": purity_score,
        "grade": grade,
        "grade_code": grade_code,
        "is_suitable": is_suitable,
        "status_label": status_label,
        "status_color": status_color,
        "healthy_percent": healthy_pct,
        "chlorosis_percent": chlorosis_pct,
        "necrosis_percent": necrosis_pct,
        "recommendation": recommendation,
        "health_overlay_b64": health_overlay_b64,
        "active_parenchyma_index": round(healthy_pct / 100.0, 3)
    }
