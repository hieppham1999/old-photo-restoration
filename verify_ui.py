"""
Step 2: Interactive UI for reviewing and editing detection masks.

Two UI backends:
  --ui gradio  (default, recommended) — opens http://localhost:7860 in the browser
  --ui opencv  — pure OpenCV window, no extra dependencies

Usage:
    python verify_ui.py --input ./input --masks ./masks --output ./masks_verified
    python verify_ui.py --input ./input --masks ./masks --output ./masks_verified --ui opencv
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def collect_pairs(input_dir, masks_dir):
    """Return list of (img_path, mask_path) for every image that has a mask."""
    input_dir, masks_dir = Path(input_dir), Path(masks_dir)
    pairs = []
    for img_path in sorted(input_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        mask_path = masks_dir / (img_path.stem + "_mask.png")
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    if not pairs:
        print(f"No matching image/mask pairs found.\n"
              f"  Images: {input_dir}\n  Masks:  {masks_dir}")
    return pairs


def load_rgb(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
    return mask


def create_overlay(img_rgb, mask_gray, alpha=0.5):
    """Blend original image with a semi-transparent red mask overlay."""
    overlay = img_rgb.copy()
    roi = mask_gray > 127
    overlay[roi, 0] = np.clip(img_rgb[roi, 0] * (1 - alpha) + 255 * alpha, 0, 255).astype(np.uint8)
    overlay[roi, 1] = (img_rgb[roi, 1] * (1 - alpha)).astype(np.uint8)
    overlay[roi, 2] = (img_rgb[roi, 2] * (1 - alpha)).astype(np.uint8)
    return overlay


def save_verified_mask(mask_gray, img_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (img_path.stem + "_mask_verified.png")
    cv2.imwrite(str(out_path), mask_gray)
    return out_path


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def extract_mask_from_editor(editor_val):
    """
    Pull the binary mask from a gr.ImageEditor return value.
    Returns uint8 grayscale ndarray or None.
    """
    if editor_val is None:
        return None
    # Gradio 4.x returns a dict with "composite" key
    if isinstance(editor_val, dict):
        img = editor_val.get("composite")
        if img is None:
            img = editor_val.get("background")
    else:
        img = editor_val

    if img is None:
        return None

    # img may be PIL or numpy; convert to numpy
    try:
        img = np.array(img)
    except Exception:
        return None

    if img.ndim == 3:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary


def launch_gradio(input_dir, masks_dir, output_dir):
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run:  pip install gradio>=4.0.0")
        sys.exit(1)

    pairs = collect_pairs(input_dir, masks_dir)
    if not pairs:
        return

    # ---- helper: build all UI data for index i ----
    def load_idx(i):
        img_path, mask_path = pairs[i]
        img_rgb = load_rgb(img_path)
        mask_gray = load_mask(mask_path)
        overlay = create_overlay(img_rgb, mask_gray)
        mask_rgb = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2RGB)
        editor_val = {"background": mask_rgb, "layers": [], "composite": mask_rgb}
        status = f"Image {i + 1} / {len(pairs)}:  {img_path.name}"
        return img_rgb, overlay, editor_val, status

    def do_save(editor_val, idx):
        mask = extract_mask_from_editor(editor_val)
        if mask is None:
            return "Error: could not read mask from editor."
        img_path, _ = pairs[idx]
        out = save_verified_mask(mask, img_path, output_dir)
        return f"Saved → {out.name}"

    def on_save_next(editor_val, idx):
        msg = do_save(editor_val, idx)
        new_idx = min(idx + 1, len(pairs) - 1)
        orig, ov, ed, status = load_idx(new_idx)
        if new_idx == idx:
            status = "All images done!  " + status
        return orig, ov, ed, status, new_idx, msg

    def on_prev(editor_val, idx):
        new_idx = max(idx - 1, 0)
        orig, ov, ed, status = load_idx(new_idx)
        return orig, ov, ed, status, new_idx

    def on_save_only(editor_val, idx):
        return do_save(editor_val, idx)

    with gr.Blocks(title="Scratch & Dust Removal — Verify Masks") as demo:
        gr.Markdown("## Scratch & Dust Removal — Verify Masks")
        gr.Markdown(
            "**Left**: original image.  **Centre**: mask overlay (red = detected).  "
            "**Right**: editable mask — paint **white** to add regions, **black** to erase false positives."
        )

        idx_state = gr.State(0)

        with gr.Row():
            orig_img = gr.Image(label="Original", interactive=False, height=500)
            overlay_img = gr.Image(label="Mask Overlay", interactive=False, height=500)
            mask_editor = gr.ImageEditor(
                label="Edit Mask  (white = remove  /  black = keep)",
                brush=gr.Brush(default_size=20, colors=["#ffffff", "#000000"], color_mode="fixed"),
                height=500,
            )

        status_box = gr.Textbox(label="Progress", interactive=False)
        save_msg = gr.Textbox(label="Save status", interactive=False)

        with gr.Row():
            prev_btn = gr.Button("← Previous")
            save_btn = gr.Button("Save Only", variant="secondary")
            next_btn = gr.Button("Save & Next →", variant="primary")

        outputs_nav = [orig_img, overlay_img, mask_editor, status_box, idx_state]

        demo.load(
            fn=lambda: list(load_idx(0)) + [0],
            outputs=outputs_nav,
        )

        next_btn.click(
            fn=on_save_next,
            inputs=[mask_editor, idx_state],
            outputs=outputs_nav + [save_msg],
        )
        prev_btn.click(
            fn=on_prev,
            inputs=[mask_editor, idx_state],
            outputs=outputs_nav,
        )
        save_btn.click(
            fn=on_save_only,
            inputs=[mask_editor, idx_state],
            outputs=[save_msg],
        )

    print(f"Opening Gradio UI at http://localhost:{config.GRADIO_PORT}")
    demo.launch(server_port=config.GRADIO_PORT)


# ---------------------------------------------------------------------------
# OpenCV UI
# ---------------------------------------------------------------------------

class _OpenCVEditor:
    """Brush-based mask editor using OpenCV imshow."""

    HELP = (
        "B=erase mask  D=draw mask  +/-=zoom  scroll=brush size  "
        "S=save&next  Q=quit"
    )

    def __init__(self, img_bgr, mask_gray, title="Verify mask"):
        self.img = img_bgr
        self.mask = mask_gray.copy()
        self.title = title
        self.mode = "erase"   # "erase" removes white, "draw" adds white
        self.brush = 20
        self.zoom = 1.0
        self.drawing = False

    # -- rendering --

    def _frame(self):
        h, w = self.img.shape[:2]
        disp = self.img.copy()
        red = np.zeros_like(disp)
        red[:, :, 2] = 255
        roi = self.mask > 127
        disp[roi] = (0.5 * red[roi] + 0.5 * disp[roi]).astype(np.uint8)

        info = (f"Mode: {'ERASE (B)' if self.mode == 'erase' else 'DRAW (D)'}  "
                f"| Brush: {self.brush}px  | Zoom: {self.zoom:.1f}x  | {self.HELP}")
        cv2.putText(disp, info, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

        if self.zoom != 1.0:
            nw, nh = int(w * self.zoom), int(h * self.zoom)
            disp = cv2.resize(disp, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return disp

    # -- mouse --

    def _to_img_coords(self, x, y):
        return int(x / self.zoom), int(y / self.zoom)

    def _on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self._paint(*self._to_img_coords(x, y))
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self._paint(*self._to_img_coords(x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.brush = max(1, min(200, self.brush + (5 if flags > 0 else -5)))

    def _paint(self, x, y):
        color = 255 if self.mode == "draw" else 0
        cv2.circle(self.mask, (x, y), self.brush, color, -1)

    # -- main loop --

    def run(self):
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.title, self._on_mouse)

        while True:
            cv2.imshow(self.title, self._frame())
            key = cv2.waitKey(30) & 0xFF

            if key in (ord('q'), ord('Q')):
                cv2.destroyWindow(self.title)
                return None                    # signal: user quit
            if key in (ord('s'), ord('S')):
                cv2.destroyWindow(self.title)
                return self.mask               # signal: save & next
            if key in (ord('b'), ord('B')):
                self.mode = "erase"
            if key in (ord('d'), ord('D')):
                self.mode = "draw"
            if key in (ord('+'), ord('=')):
                self.zoom = min(self.zoom * 1.25, 8.0)
            if key == ord('-'):
                self.zoom = max(self.zoom / 1.25, 0.1)


def launch_opencv(input_dir, masks_dir, output_dir):
    pairs = collect_pairs(input_dir, masks_dir)
    if not pairs:
        return

    idx = 0
    while idx < len(pairs):
        img_path, mask_path = pairs[idx]
        img_bgr = cv2.imread(str(img_path))
        mask_gray = load_mask(mask_path)

        title = f"[{idx+1}/{len(pairs)}] {img_path.name}  — S=save&next  Q=quit"
        editor = _OpenCVEditor(img_bgr, mask_gray, title=title)
        result = editor.run()

        if result is None:
            print("Quit.")
            break

        out = save_verified_mask(result, img_path, output_dir)
        print(f"Saved → {out}")
        idx += 1

    print("Verification complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify and edit detection masks")
    parser.add_argument("--input", required=True, help="Folder with original images")
    parser.add_argument("--masks", required=True, help="Folder with auto-generated masks")
    parser.add_argument("--output", default=config.MASKS_VERIFIED_DIR,
                        help="Folder to save verified masks")
    parser.add_argument("--ui", choices=["gradio", "opencv"], default="gradio",
                        help="UI backend (default: gradio)")
    args = parser.parse_args()

    if args.ui == "gradio":
        launch_gradio(args.input, args.masks, args.output)
    else:
        launch_opencv(args.input, args.masks, args.output)


if __name__ == "__main__":
    main()
