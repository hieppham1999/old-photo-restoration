"""
Step 3: Inpaint images to remove scratches and dust based on masks.

Three modes:
  opencv  — fast, good for small dust (no GPU required)
  lama    — high quality, good for long scratches (requires simple-lama-inpainting)
  hybrid  — small regions → opencv, large regions → lama  (best overall)

Usage:
    python inpaint.py --input ./input --masks ./masks_verified --output ./output
    python inpaint.py --input ./input --masks ./masks_verified --output ./output --mode lama
    python inpaint.py --input ./input --masks ./masks_verified --output ./output --mode hybrid
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Inpainting backends
# ---------------------------------------------------------------------------

def inpaint_opencv(img_bgr, mask):
    """Fast TELEA inpainting — best for dust and narrow scratches."""
    return cv2.inpaint(img_bgr, mask, config.INPAINT_RADIUS, cv2.INPAINT_TELEA)


def inpaint_lama(img_bgr, mask):
    """
    LaMa large-mask inpainting — high quality for wide/long scratches.
    Requires:  pip install simple-lama-inpainting
    """
    try:
        from simple_lama_inpainting import SimpleLama
        from PIL import Image
    except ImportError:
        raise ImportError(
            "LaMa backend requires:  pip install simple-lama-inpainting\n"
            "  (also needs torch — see requirements.txt)"
        )

    _lama_model = getattr(inpaint_lama, "_model", None)
    if _lama_model is None:
        print("Loading LaMa model…")
        inpaint_lama._model = SimpleLama()
        _lama_model = inpaint_lama._model

    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask)
    result_pil = _lama_model(img_pil, mask_pil)
    return cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)


def inpaint_hybrid(img_bgr, mask, small_threshold=config.SMALL_AREA_THRESHOLD):
    """
    Hybrid strategy:
      - Connected components with area < small_threshold  →  OpenCV TELEA (fast)
      - Larger components                                 →  LaMa (quality)
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    mask_small = np.zeros_like(mask)
    mask_large = np.zeros_like(mask)

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < small_threshold:
            mask_small[labels == i] = 255
        else:
            mask_large[labels == i] = 255

    result = img_bgr.copy()
    if mask_small.any():
        result = inpaint_opencv(result, mask_small)
    if mask_large.any():
        result = inpaint_lama(result, mask_large)
    return result


BACKENDS = {
    "opencv": inpaint_opencv,
    "lama": inpaint_lama,
    "hybrid": inpaint_hybrid,
}


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def find_mask(img_path, masks_dir):
    """Return verified mask if it exists, otherwise auto-generated mask."""
    masks_dir = Path(masks_dir)
    stem = Path(img_path).stem
    for suffix in ("_mask_verified.png", "_mask.png"):
        p = masks_dir / (stem + suffix)
        if p.exists():
            return p
    return None


def process_image(img_path, mask_path, output_dir, mode="opencv"):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")

    # Ensure mask is strictly binary
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    if not mask_bin.any():
        # Nothing to inpaint — copy as-is
        result = img
    else:
        result = BACKENDS[mode](img, mask_bin)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / Path(img_path).name
    cv2.imwrite(str(out_path), result)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_images(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return [p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def main():
    parser = argparse.ArgumentParser(description="Inpaint images to remove scratches and dust")
    parser.add_argument("--input", required=True, help="Input folder or single image")
    parser.add_argument("--masks", required=True,
                        help="Folder containing masks (verified or auto)")
    parser.add_argument("--output", default=config.OUTPUT_DIR, help="Output folder")
    parser.add_argument(
        "--mode", choices=["opencv", "lama", "hybrid"], default="opencv",
        help="Inpainting backend (default: opencv)"
    )
    args = parser.parse_args()

    images = collect_images(args.input)
    if not images:
        print(f"No images found in {args.input}")
        return

    out_dir = Path(args.output)
    print(f"Inpainting {len(images)} image(s) [mode={args.mode}]  →  {out_dir}")

    for img_path in tqdm(images):
        mask_path = find_mask(img_path, args.masks)
        if mask_path is None:
            tqdm.write(f"  SKIP {img_path.name}: no mask found in {args.masks}")
            continue
        try:
            out = process_image(img_path, mask_path, out_dir, mode=args.mode)
            tqdm.write(f"  {img_path.name}  [{mask_path.name}]  →  {out.name}")
        except Exception as exc:
            tqdm.write(f"  ERROR {img_path.name}: {exc}")

    print("Inpainting complete.")


if __name__ == "__main__":
    main()
