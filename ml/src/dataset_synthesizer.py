"""Photographic Food Image Dataset Generator for Nutrition AI.

Generates diverse, realistic food images for 50 distinct classes with varied
angles, lighting, backgrounds, camera perspectives, and textures.
"""

from __future__ import annotations

import csv
import math
import os
import random
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import numpy as np


def create_background(bg_type: int, width: int = 224, height: int = 224, rng: random.Random | None = None) -> np.ndarray:
    """Create photographic background surface.
    0: Dark slate / granite
    1: Warm hardwood cutting board
    2: White marble with veins
    3: White ceramic plate on wooden table
    4: Woven linen fabric
    5: Brushed stainless steel
    """
    if rng is None:
        rng = random.Random()

    arr = np.zeros((height, width, 3), dtype=np.float32)

    if bg_type == 0:
        # Dark slate
        base_val = rng.uniform(35, 55)
        noise = np.random.normal(0, 8, (height, width, 1))
        arr[:, :] = np.clip(base_val + noise, 0, 255)
        # Add subtle slate bluish/gray tint
        arr[:, :, 2] += rng.uniform(2, 6)

    elif bg_type == 1:
        # Warm wood cutting board
        base_r = rng.uniform(150, 190)
        base_g = rng.uniform(100, 135)
        base_b = rng.uniform(60, 85)
        # Horizontal wood grain striations
        y_coords = np.arange(height)[:, None]
        freq = rng.uniform(0.1, 0.25)
        grain = np.sin(y_coords * freq) * rng.uniform(10, 25)
        noise = np.random.normal(0, 4, (height, width, 1))
        arr[:, :, 0] = base_r + grain + noise[:, :, 0]
        arr[:, :, 1] = base_g + grain * 0.7 + noise[:, :, 0]
        arr[:, :, 2] = base_b + grain * 0.4 + noise[:, :, 0]

    elif bg_type == 2:
        # White marble
        base_val = rng.uniform(220, 240)
        noise = np.random.normal(0, 3, (height, width, 1))
        arr[:, :] = base_val + noise
        # Marble vein
        x = np.linspace(-2, 2, width)
        y = np.linspace(-2, 2, height)[:, None]
        vein = np.sin(x * 2.5 + y * 1.8) ** 16 * rng.uniform(-30, -15)
        arr[:, :, 0] += vein
        arr[:, :, 1] += vein
        arr[:, :, 2] += vein * 0.9

    elif bg_type == 3:
        # White ceramic plate on tabletop
        arr[:, :, 0] = rng.uniform(120, 150)
        arr[:, :, 1] = rng.uniform(90, 120)
        arr[:, :, 2] = rng.uniform(70, 95)
        # Circular plate
        cy, cx = height / 2.0, width / 2.0
        y_grid, x_grid = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        plate_radius = rng.uniform(90, 105)
        plate_mask = dist_from_center <= plate_radius
        rim_mask = (dist_from_center > plate_radius - 8) & plate_mask

        arr[plate_mask, 0] = rng.uniform(235, 248)
        arr[plate_mask, 1] = rng.uniform(235, 248)
        arr[plate_mask, 2] = rng.uniform(238, 250)
        arr[rim_mask] -= rng.uniform(15, 25)  # Plate rim shadow

    elif bg_type == 4:
        # Woven linen
        base = rng.uniform(180, 210)
        grid_x = ((np.arange(width) % 4 < 2) * 8.0)[None, :, None]
        grid_y = ((np.arange(height) % 4 < 2) * 8.0)[:, None, None]
        arr[:, :] = base + grid_x + grid_y + np.random.normal(0, 4, (height, width, 1))

    else:
        # Brushed stainless steel
        base = rng.uniform(160, 185)
        x_coords = np.arange(width)
        streaks = (np.sin(x_coords * 0.5) * 6.0)[None, :, None]
        arr[:, :] = base + streaks + np.random.normal(0, 5, (height, width, 1))

    # Apply lighting gradient (key light from random quadrant)
    light_angle = rng.uniform(0, 2 * math.pi)
    gx = math.cos(light_angle)
    gy = math.sin(light_angle)
    x_lin = np.linspace(-1, 1, width)[None, :]
    y_lin = np.linspace(-1, 1, height)[:, None]
    lighting = (x_lin * gx + y_lin * gy) * rng.uniform(15, 30)
    arr += lighting[:, :, None]

    # Vignette
    y_grid, x_grid = np.ogrid[:height, :width]
    r_sq = ((x_grid - width / 2) / (width / 2)) ** 2 + ((y_grid - height / 2) / (height / 2)) ** 2
    vignette = 1.0 - np.clip(r_sq * 0.18, 0, 0.4)
    arr *= vignette[:, :, None]

    return np.clip(arr, 0, 255).astype(np.uint8)


def generate_food_image(
    food_name: str,
    class_config: dict,
    image_idx: int,
    width: int = 224,
    height: int = 224,
) -> Image.Image:
    """Generate a photo of a given food class with realistic features."""
    rng = random.Random(hash(f"{food_name}_{image_idx}"))
    np_rng = np.random.default_rng(abs(hash(f"{food_name}_{image_idx}")) % (2**32))

    # 1. Background
    bg_id = rng.randint(0, 5)
    bg_arr = create_background(bg_id, width, height, rng=rng)
    bg_img = Image.fromarray(bg_arr, mode="RGB")

    # 2. Food Item Overlay
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Center position with slight jitter
    cx = width / 2.0 + rng.uniform(-18, 18)
    cy = height / 2.0 + rng.uniform(-18, 18)

    # Angle / scale variation
    scale = rng.uniform(0.70, 1.15)
    shape_type = class_config.get("shape", "circle")
    primary_color = class_config.get("color1", (180, 50, 40))
    secondary_color = class_config.get("color2", (140, 30, 25))
    accent_color = class_config.get("accent", (50, 120, 40))

    # Jitter colors for lighting / natural variation
    def jitter_color(c, amt=15):
        return tuple(int(np.clip(v + rng.uniform(-amt, amt), 0, 255)) for v in c)

    c1 = jitter_color(primary_color, 18)
    c2 = jitter_color(secondary_color, 18)
    acc = jitter_color(accent_color, 15)

    # Render drop shadow first
    shadow_offset_x = rng.uniform(4, 10)
    shadow_offset_y = rng.uniform(5, 12)
    s_rad = 50 * scale

    shadow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(shadow_layer)
    s_draw.ellipse(
        [
            cx + shadow_offset_x - s_rad * 1.1,
            cy + shadow_offset_y - s_rad * 0.7,
            cx + shadow_offset_x + s_rad * 1.1,
            cy + shadow_offset_y + s_rad * 0.7,
        ],
        fill=(10, 10, 15, int(rng.uniform(70, 120))),
    )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=rng.uniform(7, 14)))
    bg_img.paste(shadow_layer, (0, 0), shadow_layer)

    # Draw food shape
    if shape_type == "circle":  # Oranges, apples, mangos, tomatoes
        r_x = rng.uniform(45, 60) * scale
        r_y = rng.uniform(42, 58) * scale
        bbox = [cx - r_x, cy - r_y, cx + r_x, cy + r_y]
        draw.ellipse(bbox, fill=c1 + (255,), outline=c2 + (255,), width=2)
        # Highlight
        hx = cx - r_x * 0.35
        hy = cy - r_y * 0.35
        draw.ellipse([hx - r_x * 0.25, hy - r_y * 0.25, hx + r_x * 0.25, hy + r_y * 0.25], fill=(255, 255, 255, 55))

    elif shape_type == "curved":  # Bananas
        points = []
        angle_rad = rng.uniform(-0.5, 0.5)
        for t in np.linspace(-1.2, 1.2, 25):
            px = cx + (t * 55 * scale) * math.cos(angle_rad) - (t * t * 25 * scale) * math.sin(angle_rad)
            py = cy + (t * 55 * scale) * math.sin(angle_rad) + (t * t * 25 * scale) * math.cos(angle_rad)
            points.append((px, py))
        for p in points:
            draw.ellipse([p[0] - 14 * scale, p[1] - 14 * scale, p[0] + 14 * scale, p[1] + 14 * scale], fill=c1 + (255,))
        # Tips
        draw.ellipse([points[0][0] - 8, points[0][1] - 8, points[0][0] + 8, points[0][1] + 8], fill=c2 + (255,))
        draw.ellipse([points[-1][0] - 8, points[-1][1] - 8, points[-1][0] + 8, points[-1][1] + 8], fill=c2 + (255,))

    elif shape_type == "florets":  # Broccoli, kale
        # Cluster of florets
        n_florets = rng.randint(9, 18)
        # Stalk
        draw.rectangle([cx - 10 * scale, cy, cx + 10 * scale, cy + 45 * scale], fill=acc + (255,))
        for _ in range(n_florets):
            fx = cx + rng.uniform(-40, 40) * scale
            fy = cy + rng.uniform(-35, 10) * scale
            fr = rng.uniform(12, 22) * scale
            draw.ellipse([fx - fr, fy - fr, fx + fr, fy + fr], fill=c1 + (255,), outline=c2 + (255,), width=1)

    elif shape_type == "tapered":  # Carrots, wasabi
        top_y = cy - 50 * scale
        bot_y = cy + 50 * scale
        w_top = 18 * scale
        poly = [
            (cx - w_top, top_y),
            (cx + w_top, top_y),
            (cx + 4, bot_y),
            (cx - 4, bot_y),
        ]
        draw.polygon(poly, fill=c1 + (255,), outline=c2 + (255,))
        # Greens on top
        for _ in range(4):
            gx = cx + rng.uniform(-15, 15)
            draw.line([(cx, top_y), (gx, top_y - rng.uniform(20, 35))], fill=acc + (255,), width=3)

    elif shape_type == "leaves":  # Spinach, greens
        n_leaves = rng.randint(6, 12)
        for _ in range(n_leaves):
            lx = cx + rng.uniform(-35, 35) * scale
            ly = cy + rng.uniform(-35, 35) * scale
            lw = rng.uniform(16, 28) * scale
            lh = rng.uniform(25, 45) * scale
            rot = rng.uniform(0, 360)
            leaf = Image.new("RGBA", (int(lw * 2), int(lh * 2)), (0, 0, 0, 0))
            ldraw = ImageDraw.Draw(leaf)
            ldraw.ellipse([2, 2, lw * 2 - 2, lh * 2 - 2], fill=c1 + (235,), outline=c2 + (255,))
            ldraw.line([lw, 2, lw, lh * 2 - 2], fill=acc + (180,), width=1)
            leaf_rot = leaf.rotate(rot, expand=True)
            overlay.paste(leaf_rot, (int(lx - leaf_rot.width / 2), int(ly - leaf_rot.height / 2)), leaf_rot)

    elif shape_type == "pie_slice":  # Pizza, pies
        r = 65 * scale
        bbox = [cx - r, cy - r, cx + r, cy + r]
        start_a = rng.uniform(0, 270)
        draw.pieslice(bbox, start=start_a, end=start_a + 65, fill=c1 + (255,), outline=c2 + (255,), width=3)
        # Pepperoni / topping flecks
        for _ in range(5):
            ta = math.radians(start_a + rng.uniform(10, 55))
            tr = rng.uniform(15, r * 0.75)
            tx = cx + tr * math.cos(ta)
            ty = cy + tr * math.sin(ta)
            draw.ellipse([tx - 6, ty - 6, tx + 6, ty + 6], fill=acc + (255,))

    elif shape_type == "bowl_dip":  # Hummus, yogurt, dips
        r = 55 * scale
        draw.ellipse([cx - r, cy - r * 0.8, cx + r, cy + r * 0.8], fill=(230, 230, 235, 255), outline=(180, 180, 190, 255), width=4)
        # Dip surface
        draw.ellipse([cx - r * 0.85, cy - r * 0.65, cx + r * 0.85, cy + r * 0.65], fill=c1 + (255,))
        # Swirl
        draw.arc([cx - r * 0.5, cy - r * 0.35, cx + r * 0.5, cy + r * 0.35], start=45, end=270, fill=c2 + (255,), width=3)
        # Garnish flecks
        for _ in range(12):
            fx = cx + rng.uniform(-r * 0.4, r * 0.4)
            fy = cy + rng.uniform(-r * 0.3, r * 0.3)
            draw.ellipse([fx - 2, fy - 2, fx + 2, fy + 2], fill=acc + (255,))

    else:  # Slices / rectangles / bread / nuggets
        rw = rng.uniform(45, 65) * scale
        rh = rng.uniform(35, 55) * scale
        bbox = [cx - rw, cy - rh, cx + rw, cy + rh]
        draw.rounded_rectangle(bbox, radius=int(8 * scale), fill=c1 + (255,), outline=c2 + (255,), width=2)
        # Texture lines
        for _ in range(4):
            tx = cx + rng.uniform(-rw * 0.6, rw * 0.6)
            draw.line([(tx, cy - rh * 0.6), (tx + rng.uniform(-5, 5), cy + rh * 0.6)], fill=acc + (120,), width=1)

    # Composite overlay onto background
    bg_img.paste(overlay, (0, 0), overlay)

    # Subtle camera lens blur for shallow depth of field
    if rng.random() > 0.6:
        bg_img = bg_img.filter(ImageFilter.SMOOTH_MORE)

    return bg_img
