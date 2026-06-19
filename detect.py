"""
Step 1: Automatically detect scratches and dust in images.

Output: {image_name}_mask.png  (grayscale, white = area to remove)

Usage:
    python detect.py --input ./input --output ./masks
    python detect.py --input ./input --output ./masks --threshold 15 --max-area 8000
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def detect_scratches(
    img_path,
    threshold=config.THRESHOLD,
    blur_sigma=config.BLUR_SIGMA,
    min_area=config.MIN_AREA,
    max_area=config.MAX_AREA,
    max_dim=config.MAX_DIM,
    mask_dilate=config.MASK_DILATE,
):
    """
    Detect scratches and dust in a single image.

    Returns:
        mask (np.ndarray): uint8 grayscale, 255 = area to remove
        n_regions (int): number of detected regions
        coverage_pct (float): percentage of image area covered
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    orig_h, orig_w = img.shape[:2]
    scale = 1.0

    # Downscale very large images for faster processing
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / max(orig_h, orig_w)
        img_work = cv2.resize(
            img, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        img_work = img

    # Convert to LAB and work on the L channel (luminance)
    L = cv2.cvtColor(img_work, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)

    # Blurred L = "ideal" smooth version; diff = anomaly map
    L_blur = cv2.GaussianBlur(L, (0, 0), blur_sigma)
    diff = np.abs(L - L_blur).astype(np.uint8)

    # Hard threshold → rough binary mask
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Morphological open 2×2: remove single-pixel noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # Morphological close 3×3: bridge small gaps in scratches
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Connected-component filtering by area
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    kept = 0
    for i in range(1, n_labels):          # 0 = background
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            filtered[labels == i] = 255
            kept += 1

    # Dilate mask so scratch/dust edges are fully covered before inpainting.
    # Done before upscaling so the kernel size stays consistent across image sizes.
    if mask_dilate > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (mask_dilate * 2 + 1, mask_dilate * 2 + 1)
        )
        filtered = cv2.dilate(filtered, kernel)

    # Scale mask back to original resolution
    if scale != 1.0:
        filtered = cv2.resize(filtered, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    coverage = filtered.sum() / 255 / (orig_w * orig_h) * 100
    return filtered, kept, coverage


def collect_images(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return [p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def main():
    parser = argparse.ArgumentParser(description="Detect scratches and dust in images")
    parser.add_argument("--input", required=True, help="Input folder or single image")
    parser.add_argument("--output", default=config.MASKS_DIR, help="Output folder for masks")
    parser.add_argument("--threshold", type=int, default=config.THRESHOLD,
                        help=f"Anomaly threshold (default {config.THRESHOLD})")
    parser.add_argument("--blur-sigma", type=float, default=config.BLUR_SIGMA)
    parser.add_argument("--min-area", type=int, default=config.MIN_AREA)
    parser.add_argument("--max-area", type=int, default=config.MAX_AREA)
    parser.add_argument("--max-dim", type=int, default=config.MAX_DIM,
                        help="Resize longest side to this before detection")
    parser.add_argument("--dilate", type=int, default=config.MASK_DILATE,
                        help="Pixels to expand mask outward (default %(default)s); increase if edges still visible")
    args = parser.parse_args()

    images = collect_images(args.input)
    if not images:
        print(f"No images found in {args.input}")
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        threshold=args.threshold,
        blur_sigma=args.blur_sigma,
        min_area=args.min_area,
        max_area=args.max_area,
        max_dim=args.max_dim,
        mask_dilate=args.dilate,
    )

    print(f"Detecting in {len(images)} image(s)  →  {out_dir}")
    for img_path in tqdm(images):
        try:
            mask, n_regions, pct = detect_scratches(img_path, **kwargs)
            mask_path = out_dir / (img_path.stem + "_mask.png")
            cv2.imwrite(str(mask_path), mask)
            tqdm.write(f"  {img_path.name}: {n_regions} regions, {pct:.2f}% affected")
        except Exception as exc:
            tqdm.write(f"  ERROR {img_path.name}: {exc}")

    print("Detection complete.")


if __name__ == "__main__":
    main()
