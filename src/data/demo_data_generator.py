"""
Procedural Botanical Leaf Dataset Generator
Generates realistic, mathematically modelled foliar specimens for each medicinal plant species.
Models unique botanical morphology (shape, serration, venation architecture, aspect ratio, coloration)
allowing instant zero-shot demonstration, training, and benchmarking of the hybrid deep learning system.
"""

import os
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import Tuple, List, Dict
import cv2

from src.data.plant_database import MEDICINAL_PLANTS, CLASS_NAMES


def draw_leaf_contour(
    draw: ImageDraw.ImageDraw,
    center: Tuple[int, int],
    species: str,
    img_size: int = 224,
    leaf_color: Tuple[int, int, int] = (45, 115, 45)
) -> np.ndarray:
    """
    Renders realistic botanical contours specific to each plant species.
    Returns binary mask of the leaf.
    """
    cx, cy = center
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    
    # Species-specific morphological modeling
    points = []
    num_points = 180
    
    if species == "Centella_asiatica":
        # Reniform / Kidney-shaped leaf with cordate sinus
        for i in range(num_points):
            theta = 2 * math.pi * i / num_points
            r_base = img_size * 0.35
            # Cardioid / kidney formula with crenate edge
            r = r_base * (0.8 + 0.25 * math.cos(theta) - 0.15 * math.sin(theta)**2)
            # Crenate tooth indentation
            r += 4 * math.sin(18 * theta)
            x = int(cx + r * math.cos(theta))
            y = int(cy + r * math.sin(theta) * 0.75)
            points.append((x, y))
            
    elif species == "Aloe_barbadensis":
        # Succulent, elongated lance-like rosette leaves with spiny marginal teeth
        for i in range(num_points):
            theta = 2 * math.pi * i / num_points
            t = (i - num_points / 2) / (num_points / 2)
            if theta < math.pi:
                # Right edge tapering to sharp acute apex
                x = int(cx + (img_size * 0.18) * (1 - abs(t)**1.4) + 3 * math.sin(24 * t))
                y = int(cy + t * img_size * 0.42)
            else:
                # Left edge with spiny dentate teeth
                x = int(cx - (img_size * 0.18) * (1 - abs(t)**1.4) - 3 * math.sin(24 * t))
                y = int(cy + t * img_size * 0.42)
            points.append((x, y))

    elif species == "Azadirachta_indica":
        # Neem: Asymmetric oblique base, falcate curve, sharply serrate margin
        for i in range(num_points):
            t = (i - num_points / 2) / (num_points / 2)
            serration = 3.5 * math.sin(30 * t) if abs(t) < 0.85 else 0
            # Slight falcate curvature
            falcate = 12 * math.sin(math.pi * (t + 1) / 2)
            if i < num_points / 2:
                # Oblique asymmetrical bulge on one side
                width = (img_size * 0.16) * (1 - t**2) + falcate
                x = int(cx + width + serration)
                y = int(cy + t * img_size * 0.42)
            else:
                width = (img_size * 0.22) * (1 - t**2) - falcate
                x = int(cx - width - serration)
                y = int(cy + t * img_size * 0.42)
            points.append((x, y))

    elif species == "Curcuma_longa":
        # Turmeric: Large oblong-lanceolate blade, entire undulate margin
        for i in range(num_points):
            t = (i - num_points / 2) / (num_points / 2)
            undulate = 3.0 * math.sin(8 * t)
            width = (img_size * 0.28) * (1 - (t**2)**0.8) + undulate
            if i < num_points / 2:
                x = int(cx + width)
                y = int(cy + t * img_size * 0.44)
            else:
                x = int(cx - width)
                y = int(cy + t * img_size * 0.44)
            points.append((x, y))

    elif species == "Catharanthus_roseus":
        # Periwinkle: Glossy obovate blade, rounded apex with tiny mucronate tip, smooth margin
        for i in range(num_points):
            t = (i - num_points / 2) / (num_points / 2)
            # Obovate: broader towards upper third
            skew = (1.0 + 0.35 * t) if t < 0 else (1.0 - 0.25 * t)
            width = (img_size * 0.24) * math.sqrt(max(0.01, 1 - t**2)) * skew
            if i < num_points / 2:
                x = int(cx + width)
                y = int(cy + t * img_size * 0.36)
            else:
                x = int(cx - width)
                y = int(cy + t * img_size * 0.36)
            points.append((x, y))

    elif species == "Mentha_spicata":
        # Mint: Ovate-lanceolate, sharply serrated margins, sunken veins
        for i in range(num_points):
            t = (i - num_points / 2) / (num_points / 2)
            serrate = 4.0 * math.sin(36 * t) if abs(t) < 0.9 else 0
            width = (img_size * 0.23) * (1 - abs(t)**1.2) + serrate
            if i < num_points / 2:
                x = int(cx + width)
                y = int(cy + t * img_size * 0.38)
            else:
                x = int(cx - width)
                y = int(cy + t * img_size * 0.38)
            points.append((x, y))

    elif species == "Hibiscus_rosa_sinensis":
        # Hibiscus: Broadly ovate with coarse serration in upper half
        for i in range(num_points):
            t = (i - num_points / 2) / (num_points / 2)
            # Serration prominent towards apex (t < 0)
            dentate = 5.0 * math.sin(22 * t) if t < 0.1 else 1.0 * math.sin(10 * t)
            width = (img_size * 0.30) * (1 - t**2)**0.7 + dentate
            if i < num_points / 2:
                x = int(cx + width)
                y = int(cy + t * img_size * 0.38)
            else:
                x = int(cx - width)
                y = int(cy + t * img_size * 0.38)
            points.append((x, y))

    elif species == "Moringa_oleifera":
        # Moringa: Small delicate obovate leaflet with pale midrib
        for i in range(num_points):
            theta = 2 * math.pi * i / num_points
            r_x = img_size * 0.22
            r_y = img_size * 0.28
            # Slightly wider above center
            y_offset = -img_size * 0.05 * math.cos(theta)
            x = int(cx + r_x * math.sin(theta))
            y = int(cy + r_y * math.cos(theta) + y_offset)
            points.append((x, y))

    elif species == "Justicia_adhatoda":
        # Vasaka: Elliptic-lanceolate, long tapering apex, repand-undulate
        for i in range(num_points):
            t = (i - num_points / 2) / (num_points / 2)
            width = (img_size * 0.20) * (1 - abs(t)**1.5)
            if i < num_points / 2:
                x = int(cx + width)
                y = int(cy + t * img_size * 0.45)
            else:
                x = int(cx - width)
                y = int(cy + t * img_size * 0.45)
            points.append((x, y))

    else:
        # Tulsi (Ocimum tenuiflorum) & default: Ovate, slightly pubescent margin, distinct petiole
        for i in range(num_points):
            t = (i - num_points / 2) / (num_points / 2)
            serrate = 2.0 * math.sin(20 * t) if abs(t) < 0.8 else 0
            width = (img_size * 0.25) * (1 - t**2) + serrate
            if i < num_points / 2:
                x = int(cx + width)
                y = int(cy + t * img_size * 0.38)
            else:
                x = int(cx - width)
                y = int(cy + t * img_size * 0.38)
            points.append((x, y))

    # Clean bounds
    points = [(max(2, min(img_size - 3, x)), max(2, min(img_size - 3, y))) for x, y in points]
    
    # Fill leaf blade
    draw.polygon(points, fill=leaf_color)
    
    # Create mask for venation and texture overlay
    pts_np = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts_np], 255)
    
    return mask


def draw_venation(
    draw: ImageDraw.ImageDraw,
    center: Tuple[int, int],
    species: str,
    img_size: int = 224,
    vein_color: Tuple[int, int, int] = (80, 160, 80)
):
    """
    Renders realistic primary midrib and secondary lateral venation networks.
    Crucial for CNN (EfficientNetV2) feature learning and Grad-CAM saliency.
    """
    cx, cy = center
    
    if species == "Centella_asiatica":
        # Palmate venation radiating from cordate sinus
        sinus_y = cy + int(img_size * 0.15)
        for angle_deg in range(-120, 120, 25):
            rad = math.radians(angle_deg - 90)
            end_x = int(cx + (img_size * 0.32) * math.cos(rad))
            end_y = int(sinus_y + (img_size * 0.25) * math.sin(rad))
            draw.line([(cx, sinus_y), (end_x, end_y)], fill=vein_color, width=2)
            # Tertiary branchlets
            mid_x = (cx + end_x) // 2
            mid_y = (sinus_y + end_y) // 2
            bx1 = int(mid_x + 12 * math.cos(rad + 0.4))
            by1 = int(mid_y + 12 * math.sin(rad + 0.4))
            draw.line([(mid_x, mid_y), (bx1, by1)], fill=vein_color, width=1)
            
    elif species == "Aloe_barbadensis":
        # Succulent: Central fleshy ridge and fine longitudinal parallel striations
        draw.line([(cx, cy - int(img_size * 0.4)), (cx, cy + int(img_size * 0.4))], fill=vein_color, width=3)
        for offset in [-12, -6, 6, 12]:
            draw.line([(cx + offset, cy - int(img_size * 0.35)), 
                       (cx + offset, cy + int(img_size * 0.35))], fill=vein_color, width=1)
                       
    else:
        # Pinnate venation: Main central midrib
        top_y = cy - int(img_size * 0.40)
        bot_y = cy + int(img_size * 0.42)
        draw.line([(cx, top_y), (cx, bot_y)], fill=vein_color, width=3)
        
        # Lateral secondary veins curving gracefully towards margins
        num_veins = 7 if species in ["Moringa_oleifera", "Mentha_spicata"] else 11
        for v in range(num_veins):
            t = -0.32 + (v / (num_veins - 1)) * 0.65
            vy = int(cy + t * img_size)
            vein_len = int(img_size * 0.18 * (1 - abs(t*1.2)**2))
            
            # Left lateral vein
            ctrl_lx = cx - vein_len // 2
            ctrl_ly = vy - 6
            end_lx = cx - vein_len
            end_ly = vy - 14
            draw.line([(cx, vy), (ctrl_lx, ctrl_ly), (end_lx, end_ly)], fill=vein_color, width=2)
            
            # Right lateral vein
            ctrl_rx = cx + vein_len // 2
            ctrl_ry = vy - 6
            end_rx = cx + vein_len
            end_ry = vy - 14
            draw.line([(cx, vy), (ctrl_rx, ctrl_ry), (end_rx, end_ry)], fill=vein_color, width=2)


def generate_single_leaf_image(
    species: str,
    img_size: int = 224,
    randomize: bool = True
) -> Image.Image:
    """
    Synthesizes a single high-quality botanical leaf sample with realistic texture and shading.
    """
    # Create canvas: clean white or botanical herbarium studio background with subtle texture
    bg_brightness = random.randint(238, 252) if randomize else 248
    bg = np.full((img_size, img_size, 3), bg_brightness, dtype=np.uint8)
    
    # Add subtle photographic paper grain
    if randomize:
        noise = np.random.normal(0, 3, (img_size, img_size, 3)).astype(np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    img = Image.fromarray(bg)
    draw = ImageDraw.Draw(img)
    
    # Center with minor jitter
    cx = img_size // 2 + (random.randint(-4, 4) if randomize else 0)
    cy = img_size // 2 + (random.randint(-4, 4) if randomize else 0)
    
    # Color palette tailored to botanical leaf characteristics
    color_palette = {
        "Azadirachta_indica": (36, 110, 48),       # Deep serrated emerald
        "Ocimum_tenuiflorum": (40, 95, 42),        # Herbaceous warm green with purple undertones
        "Aloe_barbadensis": (75, 138, 105),        # Glaucous sea-green succulent
        "Catharanthus_roseus": (25, 105, 38),      # Highly glossy dark forest green
        "Moringa_oleifera": (55, 130, 50),         # Fresh spring green
        "Centella_asiatica": (48, 122, 56),        # Vibrant leafy fan green
        "Mentha_spicata": (35, 125, 45),           # Crisp bright spearmint green
        "Curcuma_longa": (50, 115, 40),            # Yellowish-green tropical blade
        "Justicia_adhatoda": (30, 98, 42),         # Dense olive-tinted lanceolate
        "Hibiscus_rosa_sinensis": (28, 108, 46)    # Rich tropical dark green
    }
    
    base_color = color_palette.get(species, (40, 110, 45))
    if randomize:
        r = np.clip(base_color[0] + random.randint(-10, 10), 10, 240)
        g = np.clip(base_color[1] + random.randint(-12, 12), 30, 240)
        b = np.clip(base_color[2] + random.randint(-10, 10), 10, 240)
        leaf_color = (int(r), int(g), int(b))
    else:
        leaf_color = base_color
        
    vein_color = (
        min(255, leaf_color[0] + 35),
        min(255, leaf_color[1] + 45),
        min(255, leaf_color[2] + 25)
    )
    
    # 1. Render leaf contour
    mask = draw_leaf_contour(draw, (cx, cy), species, img_size, leaf_color)
    
    # 2. Render venation architecture
    draw_venation(draw, (cx, cy), species, img_size, vein_color)
    
    # 3. Add organic leaf surface gradient / lighting effect
    img_np = np.array(img)
    if randomize:
        # Slight leaf shadow
        shadow_kernel = np.ones((5, 5), np.uint8)
        shadow_mask = cv2.dilate(mask, shadow_kernel, iterations=2) - mask
        img_np[shadow_mask > 0] = np.clip(img_np[shadow_mask > 0].astype(np.int16) - 18, 0, 255).astype(np.uint8)
        
        # Subtle internal highlights on upper blade
        highlight_y = cy - int(img_size * 0.15)
        for hy in range(max(0, highlight_y - 25), min(img_size, highlight_y + 25)):
            for hx in range(max(0, cx - 35), min(img_size, cx + 35)):
                if mask[hy, hx] > 0 and random.random() > 0.4:
                    img_np[hy, hx] = np.clip(img_np[hy, hx].astype(np.int16) + 12, 0, 255).astype(np.uint8)
                    
    final_img = Image.fromarray(img_np)
    
    # Slight antialiasing blur for photorealistic edge rendering
    final_img = final_img.filter(ImageFilter.SMOOTH_MORE)
    
    return final_img


def build_demo_dataset(
    output_dir: str = "dataset",
    samples_per_class: int = 15,
    splits: Tuple[float, float, float] = (0.70, 0.15, 0.15)
) -> Dict[str, int]:
    """
    Builds a complete, formatted PyTorch ImageFolder dataset containing
    all 10 medicinal species partitioned into train, val, and test directories.
    """
    train_ratio, val_ratio, test_ratio = splits
    n_train = max(1, int(samples_per_class * train_ratio))
    n_val = max(1, int(samples_per_class * val_ratio))
    n_test = max(1, samples_per_class - n_train - n_val)
    
    summary = {"total_images": 0, "classes": len(CLASS_NAMES)}
    
    for split_name, count in [("train", n_train), ("val", n_val), ("test", n_test)]:
        for species in CLASS_NAMES:
            target_folder = os.path.join(output_dir, split_name, species)
            os.makedirs(target_folder, exist_ok=True)
            
            for idx in range(count):
                leaf_img = generate_single_leaf_image(species, img_size=224, randomize=True)
                file_path = os.path.join(target_folder, f"{species}_{split_name}_{idx+1:03d}.jpg")
                leaf_img.save(file_path, "JPEG", quality=95)
                summary["total_images"] += 1
                
    # Also save one representative demo sample of each species in static/samples for the web app UI!
    samples_dir = os.path.join("static", "samples")
    os.makedirs(samples_dir, exist_ok=True)
    for species in CLASS_NAMES:
        sample_img = generate_single_leaf_image(species, img_size=224, randomize=False)
        sample_path = os.path.join(samples_dir, f"{species}.jpg")
        sample_img.save(sample_path, "JPEG", quality=95)
        
    print(f"Dataset generated successfully at '{output_dir}'. Total samples: {summary['total_images']}.")
    return summary


if __name__ == "__main__":
    build_demo_dataset(samples_per_class=12)
