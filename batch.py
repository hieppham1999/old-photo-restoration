"""
Run the full scratch-removal pipeline for an entire folder.

Quick usage (no verification):
    python batch.py --no-verify

With mask verification:
    python batch.py                          # opens Gradio UI between steps
    python batch.py --ui opencv              # OpenCV UI instead

Advanced:
    python batch.py --input ./input --output ./output --mode hybrid --threshold 15
"""

import argparse
import subprocess
import sys
from pathlib import Path

import config


def run(cmd, label):
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"  {' '.join(str(c) for c in cmd)}")
    print(f"{'─' * 60}")
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full scratch & dust removal pipeline"
    )
    parser.add_argument("--input", default=config.INPUT_DIR)
    parser.add_argument("--output", default=config.OUTPUT_DIR)
    parser.add_argument("--masks", default=config.MASKS_DIR)
    parser.add_argument("--masks-verified", default=config.MASKS_VERIFIED_DIR)
    parser.add_argument("--threshold", type=int, default=config.THRESHOLD)
    parser.add_argument("--blur-sigma", type=float, default=config.BLUR_SIGMA)
    parser.add_argument("--min-area", type=int, default=config.MIN_AREA)
    parser.add_argument("--max-area", type=int, default=config.MAX_AREA)
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the manual verification step")
    parser.add_argument("--ui", choices=["gradio", "opencv"], default="gradio",
                        help="Verify UI backend (default: gradio)")
    parser.add_argument("--mode", choices=["opencv", "lama", "hybrid"], default="opencv",
                        help="Inpainting mode (default: opencv)")
    args = parser.parse_args()

    py = sys.executable

    # ── Step 1: Detect ──────────────────────────────────────────────────────
    run(
        [py, "detect.py",
         "--input", args.input,
         "--output", args.masks,
         "--threshold", args.threshold,
         "--blur-sigma", args.blur_sigma,
         "--min-area", args.min_area,
         "--max-area", args.max_area],
        "Step 1 — Detect scratches & dust",
    )

    # ── Step 2: Verify (optional) ────────────────────────────────────────────
    if args.no_verify:
        print("\nVerification skipped (--no-verify).  Using auto-generated masks.")
        masks_for_inpaint = args.masks
    else:
        run(
            [py, "verify_ui.py",
             "--input", args.input,
             "--masks", args.masks,
             "--output", args.masks_verified,
             "--ui", args.ui],
            "Step 2 — Verify masks  (close the UI when done)",
        )
        masks_for_inpaint = args.masks_verified

    # ── Step 3: Inpaint ──────────────────────────────────────────────────────
    run(
        [py, "inpaint.py",
         "--input", args.input,
         "--masks", masks_for_inpaint,
         "--output", args.output,
         "--mode", args.mode],
        f"Step 3 — Inpaint  [mode={args.mode}]",
    )

    print(f"\nAll done.  Results saved to:  {args.output}")


if __name__ == "__main__":
    main()
