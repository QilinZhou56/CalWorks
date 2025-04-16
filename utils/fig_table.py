import os
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path
from collections import defaultdict

# === Config ===
PDF_PATH = "./pdfs/(Cal-CSA)Orange.pdf 03-22-40-516.pdf"
GEOMETRY_JSON = "./formatted_outputs/page_block_text_with_geometry.json"
OUTPUT_DIR = "./figure_table_crops"
DRAW_OVERLAY = False  # Set to True if want to draw instead of crop
MARGIN = 50  # Pixel margin around the crop

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load geometry JSON ===
with open(GEOMETRY_JSON, "r") as f:
    pages = json.load(f)

# === Categories to match
CATEGORY_PAIRS = {
    "table": "table caption",
    "figure": "figure caption"
}

# === Convert PDF pages to images (150 DPI)
images = convert_from_path(PDF_PATH, dpi=150)

# === Metadata storage
crops_data = []

# === Process each page ===
for page_data in pages:
    page_num = page_data["page"] + 1
    page_image = images[page_num - 1]
    cv_image = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
    img_height, img_width = cv_image.shape[:2]

    # Organize blocks by category
    blocks_by_category = defaultdict(list)
    for block in page_data.get("blocks", []):
        blocks_by_category[block["category"]].append(block)

    for main_cat, caption_cat in CATEGORY_PAIRS.items():
        for main_block in blocks_by_category[main_cat]:
            main_coords = main_block.get("region_poly", [])
            if len(main_coords) != 8:
                continue

            # Find nearest caption (above or below) 
            best_caption = None
            min_distance = float("inf")
            main_center_y = sum(main_coords[1::2]) / 4

            for caption_block in blocks_by_category[caption_cat]:
                cap_coords = caption_block.get("region_poly", [])
                if len(cap_coords) != 8:
                    continue

                cap_center_y = sum(cap_coords[1::2]) / 4
                dist = abs(cap_center_y - main_center_y)

                if dist < min_distance:
                    min_distance = dist
                    best_caption = caption_block

            # Merge regions
            merged_blocks = [main_block]
            if best_caption:
                merged_blocks.append(best_caption)

            all_pts = []
            for blk in merged_blocks:
                coords = blk["region_poly"]
                pts = np.array([
                    [coords[0], coords[1]],
                    [coords[2], coords[3]],
                    [coords[4], coords[5]],
                    [coords[6], coords[7]]
                ], np.int32).reshape((-1, 1, 2))
                all_pts.append(pts)

            if not all_pts:
                continue

            # Create mask
            mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
            for pts in all_pts:
                cv2.fillPoly(mask, [pts], 255)
            masked = cv2.bitwise_and(cv_image, cv_image, mask=mask)

            # Bounding box and margin
            all_pts_stack = np.vstack(all_pts)
            x, y, w, h = cv2.boundingRect(all_pts_stack)

            x_margin = max(0, x - MARGIN)
            y_margin = max(0, y - MARGIN)
            x_max = min(img_width, x + w + MARGIN)
            y_max = min(img_height, y + h + MARGIN)

            if DRAW_OVERLAY:
                overlay = cv_image.copy()
                for pts in all_pts:
                    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 225, 0), thickness=4)
                label = f"{main_cat}_with_caption"
                cv2.putText(overlay, label, (x_margin, y_margin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                img_name = f"page_{page_num}_{main_cat}_overlay.png"
                cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), overlay)
            else:
                # Crop and save
                # Could use masked if want more precise or cleaner version
                cropped = cv_image[y_margin:y_max, x_margin:x_max]
                cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                img_name = f"page_{page_num}_{main_cat}_with_caption_crop.png"
                img_path = os.path.join(OUTPUT_DIR, img_name)
                cropped_pil.save(img_path)

                # Save metadata
                combined_text = " ".join([b.get("text", "") for b in merged_blocks])
                crops_data.append({
                    "page": page_num,
                    "category": main_cat,
                    "text": combined_text.strip(),
                    "image_path": img_path,
                    "bounding_box": [x_margin, y_margin, x_max, y_max]
                })

# === Save metadata
df = pd.DataFrame(crops_data)
df.to_csv(os.path.join(OUTPUT_DIR, "figure_table_crops.csv"), index=False)
print("Done: Cropped images and metadata saved to:", OUTPUT_DIR)
