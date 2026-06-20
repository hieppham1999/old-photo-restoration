"""
Complete scratch-removal pipeline — single Gradio web UI.

Usage:
    python app.py
Then open http://localhost:7860

Workflow:
    Tab 1 · Upload & Detect  — upload images, run auto-detection, preview overlays
    Tab 2 · Verify Masks     — review/edit each mask with a brush tool
    Tab 3 · Inpaint & Download — run inpainting, download cleaned images
"""

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed.  Run:  pip install 'gradio>=4.0.0'")
    sys.exit(1)

import config
from detect import detect_scratches
from inpaint import process_image
from verify_ui import create_overlay, load_mask, load_rgb

# ---------------------------------------------------------------------------
# Session directories (isolated from the project's input/output folders)
# ---------------------------------------------------------------------------

_SESSION = Path(tempfile.mkdtemp(prefix="scratch_ui_"))
DIR_INPUT  = _SESSION / "input"
DIR_MASKS  = _SESSION / "masks"
DIR_OUTPUT = _SESSION / "output"
for _d in (DIR_INPUT, DIR_MASKS, DIR_OUTPUT):
    _d.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

_PLACEHOLDER = np.zeros((200, 320, 3), dtype=np.uint8)

# Multi-megapixel images crash the browser's canvas editor. Show/edit a
# downscaled version, then upscale the edited mask back to full resolution.
DISPLAY_MAX_DIM = 1280


def _downscale(img, max_dim=DISPLAY_MAX_DIM, interp=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    s = max_dim / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=interp)


# ---------------------------------------------------------------------------
# Tab 1 — Upload & Detect
# ---------------------------------------------------------------------------

def run_detection(
    uploaded_files,
    threshold,
    blur_sigma,
    min_area,
    max_area,
    mask_dilate,
    progress=gr.Progress(),
):
    if not uploaded_files:
        return "No files uploaded.", [], [], 0

    # Clear previous session data
    for p in list(DIR_INPUT.iterdir()) + list(DIR_MASKS.iterdir()) + list(DIR_OUTPUT.iterdir()):
        p.unlink(missing_ok=True)

    # Copy uploaded files into session input dir
    saved = []
    for f in uploaded_files:
        src = Path(f.name)
        dst = DIR_INPUT / src.name
        shutil.copy2(src, dst)
        saved.append(dst)

    params = {
        "threshold":   int(threshold),
        "blur_sigma":  float(blur_sigma),
        "min_area":    int(min_area),
        "max_area":    int(max_area),
        "mask_dilate": int(mask_dilate),
    }

    pairs = []
    gallery = []
    errors = []
    n = len(saved)

    for i, img_path in enumerate(saved):
        progress((i + 0.3) / n, desc=f"Detecting {img_path.name}")
        try:
            mask, n_regions, pct = detect_scratches(img_path, **params)
            mask_path = DIR_MASKS / (img_path.stem + "_mask.png")
            cv2.imwrite(str(mask_path), mask)

            img_rgb = _downscale(load_rgb(img_path))
            overlay = create_overlay(img_rgb, _downscale(mask, interp=cv2.INTER_NEAREST))
            gallery.append((overlay, f"{img_path.name} — {n_regions} regions, {pct:.1f}%"))

            pairs.append({
                "img_path":     str(img_path),
                "mask_path":    str(mask_path),
                "stem":         img_path.stem,
                "n_regions":    n_regions,
                "coverage_pct": round(pct, 2),
                "params":       dict(params),
            })
        except Exception as exc:
            errors.append(f"ERROR {img_path.name}: {exc}")
        progress((i + 1) / n)

    if errors:
        log = f"Done: {len(pairs)}/{n} images. Errors:\n" + "\n".join(errors)
    else:
        log = (
            f"Detection complete: {len(pairs)} image(s).\n"
            "Switch to the 'Verify Masks' tab to review, or go straight to 'Inpaint'."
        )
    return log, gallery, pairs, 0


# ---------------------------------------------------------------------------
# Tab 2 — Verify / Re-tune masks (no canvas; uses the live red overlay)
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS = {
    "threshold":   config.THRESHOLD,
    "blur_sigma":  config.BLUR_SIGMA,
    "min_area":    config.MIN_AREA,
    "max_area":    config.MAX_AREA,
    "mask_dilate": config.MASK_DILATE,
}


def _overlay_for(p):
    """Build (downscaled original, red overlay) for one pair."""
    img_rgb = _downscale(load_rgb(p["img_path"]))
    mask    = _downscale(load_mask(p["mask_path"]), interp=cv2.INTER_NEAREST)
    return img_rgb, create_overlay(img_rgb, mask)


def load_verify_idx(pairs, idx):
    """Return (orig, overlay, status, threshold, blur, min, max, dilate) for idx."""
    d = _DEFAULT_PARAMS
    defaults = (d["threshold"], d["blur_sigma"], d["min_area"], d["max_area"], d["mask_dilate"])
    if not pairs:
        ph = _PLACEHOLDER.copy()
        return (ph, ph, "Run detection first (Tab 1).", *defaults)

    idx = max(0, min(idx, len(pairs) - 1))
    p = pairs[idx]
    img_rgb, overlay = _overlay_for(p)
    pr = p.get("params", dict(d))
    status = (f"Image {idx + 1} / {len(pairs)}:  {Path(p['img_path']).name}  "
              f"— {p.get('n_regions', '?')} regions, {p.get('coverage_pct', 0):.1f}%")
    return (img_rgb, overlay, status,
            pr["threshold"], pr["blur_sigma"], pr["min_area"], pr["max_area"], pr["mask_dilate"])


def redetect_current(pairs, idx, threshold, blur_sigma, min_area, max_area, mask_dilate):
    """Re-run detection on the current image with new params; overwrite its mask."""
    if not pairs:
        ph = _PLACEHOLDER.copy()
        return pairs, ph, "Run detection first (Tab 1)."

    idx = max(0, min(idx, len(pairs) - 1))
    p = pairs[idx]
    params = {
        "threshold":   int(threshold),
        "blur_sigma":  float(blur_sigma),
        "min_area":    int(min_area),
        "max_area":    int(max_area),
        "mask_dilate": int(mask_dilate),
    }
    try:
        mask, n_regions, pct = detect_scratches(p["img_path"], **params)
        cv2.imwrite(p["mask_path"], mask)
    except Exception as exc:
        _, overlay = _overlay_for(p)
        return pairs, overlay, f"Re-detect failed: {exc}"

    new_pairs = list(pairs)
    new_pairs[idx] = {**p, "params": params, "n_regions": n_regions, "coverage_pct": round(pct, 2)}
    _, overlay = _overlay_for(new_pairs[idx])
    status = (f"Re-detected image {idx + 1} / {len(new_pairs)}:  {Path(p['img_path']).name}  "
              f"— {n_regions} regions, {pct:.1f}%")
    return new_pairs, overlay, status


def go_to(pairs, idx, delta, label_end):
    """Navigate by delta and return full UI state for the new index."""
    if not pairs:
        return (idx, *load_verify_idx(pairs, idx))
    new_idx = max(0, min(idx + delta, len(pairs) - 1))
    out = load_verify_idx(pairs, new_idx)
    if delta > 0 and new_idx == len(pairs) - 1 and new_idx == idx:
        out = (out[0], out[1], "All images reviewed — switch to the Inpaint tab.", *out[3:])
    return (new_idx, *out)


# ---------------------------------------------------------------------------
# Tab 3 — Inpaint & Download
# ---------------------------------------------------------------------------

def run_inpainting(pairs, mode, progress=gr.Progress()):
    if not pairs:
        return "Run detection first (Tab 1).", [], []

    gallery = []
    out_paths = []
    errors = []
    n = len(pairs)

    for i, p in enumerate(pairs):
        progress((i + 0.3) / n, desc=f"Inpainting {p['stem']}")
        try:
            out = process_image(p["img_path"], p["mask_path"], str(DIR_OUTPUT), mode=mode)
            result_rgb = _downscale(load_rgb(out))
            gallery.append((result_rgb, out.name))
            out_paths.append(str(out))
        except Exception as exc:
            errors.append(f"ERROR {p['stem']}: {exc}")
        progress((i + 1) / n)

    if errors:
        log = f"Done: {n - len(errors)}/{n} images. Errors:\n" + "\n".join(errors)
    else:
        log = f"Inpainting complete: {n} image(s). Download below."

    return log, gallery, out_paths


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

_CSS = """
.gradio-container { max-width: 100% !important; padding: 8px 16px !important; }
footer { display: none !important; }
.compact-status textarea { font-size: 0.85em; }
"""

with gr.Blocks(title="Scratch & Dust Removal", fill_width=True) as demo:
    session_pairs = gr.State([])
    verify_idx    = gr.State(0)

    # ── Tab 1 ────────────────────────────────────────────────────────────────
    with gr.Tab("1 · Upload & Detect"):
        # Image frame is the focus; controls sit in a narrow side column.
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=240):
                upload_files = gr.Files(
                    label="Images",
                    file_types=["image"],
                    file_count="multiple",
                    height=180,
                )
                detect_btn    = gr.Button("Run Detection", variant="primary")
                detect_status = gr.Textbox(label="Status", interactive=False,
                                           lines=4, elem_classes="compact-status")
            with gr.Column(scale=4):
                det_gallery = gr.Gallery(
                    label="Detection overlays (red = detected regions)",
                    columns=3, height=620, object_fit="contain",
                )

        # Detection parameters live below the image frame.
        with gr.Accordion("Detection parameters", open=False):
            with gr.Row():
                sl_thresh = gr.Slider(1, 60,      value=config.THRESHOLD,   step=1,   label="Threshold — higher = fewer false positives")
                sl_blur   = gr.Slider(0.5, 10,    value=config.BLUR_SIGMA,  step=0.5, label="Blur Sigma — higher detects wider scratches")
                sl_min    = gr.Slider(1, 200,     value=config.MIN_AREA,    step=1,   label="Min Area (px)")
                sl_max    = gr.Slider(100, 30000, value=config.MAX_AREA,    step=100, label="Max Area (px)")
                sl_dil    = gr.Slider(0, 10,      value=config.MASK_DILATE, step=1,   label="Dilate (px) — expands mask edges")

        detect_btn.click(
            fn=run_detection,
            inputs=[upload_files, sl_thresh, sl_blur, sl_min, sl_max, sl_dil],
            outputs=[detect_status, det_gallery, session_pairs, verify_idx],
        )

    # ── Tab 2 ────────────────────────────────────────────────────────────────
    with gr.Tab("2 · Verify Masks") as tab_verify:
        # Two read-only image panels are the focus (no WebGL canvas).
        with gr.Row(equal_height=True):
            orig_img = gr.Image(label="Original",                         interactive=False, height=620)
            ov_img   = gr.Image(label="Mask Overlay (red = will remove)", interactive=False, height=620)

        v_status = gr.Textbox(label="Progress", interactive=False, elem_classes="compact-status")

        # Navigation row.
        with gr.Row():
            v_prev = gr.Button("← Previous", scale=1)
            v_next = gr.Button("Accept & Next →", variant="primary", scale=1)

        # Per-image re-tuning lives below the frames. Adjust → Re-detect to
        # fix over-/under-detection; the overlay updates in place.
        with gr.Accordion("Adjust detection for this image", open=True):
            with gr.Row():
                v_thresh = gr.Slider(1, 60,      value=config.THRESHOLD,   step=1,   label="Threshold — higher = fewer false positives")
                v_blur   = gr.Slider(0.5, 10,    value=config.BLUR_SIGMA,  step=0.5, label="Blur Sigma — higher detects wider scratches")
                v_min    = gr.Slider(1, 200,     value=config.MIN_AREA,    step=1,   label="Min Area (px)")
                v_max    = gr.Slider(100, 30000, value=config.MAX_AREA,    step=100, label="Max Area (px)")
                v_dil    = gr.Slider(0, 10,      value=config.MASK_DILATE, step=1,   label="Dilate (px) — expands mask edges")
            v_redetect = gr.Button("Re-detect this image", variant="secondary")

        sliders   = [v_thresh, v_blur, v_min, v_max, v_dil]
        load_outs = [orig_img, ov_img, v_status, *sliders]
        nav_outs  = [verify_idx, *load_outs]

        # Reload current image each time the tab becomes active.
        tab_verify.select(
            fn=load_verify_idx,
            inputs=[session_pairs, verify_idx],
            outputs=load_outs,
        )

        v_redetect.click(
            fn=redetect_current,
            inputs=[session_pairs, verify_idx, *sliders],
            outputs=[session_pairs, ov_img, v_status],
        )
        v_next.click(
            fn=lambda pairs, idx: go_to(pairs, idx, +1, True),
            inputs=[session_pairs, verify_idx],
            outputs=nav_outs,
        )
        v_prev.click(
            fn=lambda pairs, idx: go_to(pairs, idx, -1, False),
            inputs=[session_pairs, verify_idx],
            outputs=nav_outs,
        )

    # ── Tab 3 ────────────────────────────────────────────────────────────────
    with gr.Tab("3 · Inpaint & Download"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=240):
                mode_dd = gr.Dropdown(
                    choices=["opencv", "lama", "hybrid"],
                    value="opencv",
                    label="Inpainting mode",
                )
                inpaint_btn    = gr.Button("Run Inpainting", variant="primary")
                inpaint_status = gr.Textbox(label="Status", interactive=False,
                                            lines=4, elem_classes="compact-status")
                inp_files      = gr.Files(label="Download results", interactive=False, height=180)
            with gr.Column(scale=4):
                inp_gallery = gr.Gallery(
                    label="Results", columns=3, height=620, object_fit="contain",
                )

        with gr.Accordion("About the modes", open=False):
            gr.Markdown(
                "- **opencv** — fast, good for dust and narrow scratches (no GPU needed)\n"
                "- **lama** — high quality for long/wide scratches (requires `simple-lama-inpainting`)\n"
                "- **hybrid** — opencv for small regions, lama for large ones"
            )

        inpaint_btn.click(
            fn=run_inpainting,
            inputs=[session_pairs, mode_dd],
            outputs=[inpaint_status, inp_gallery, inp_files],
        )

# ---------------------------------------------------------------------------

demo.queue()

if __name__ == "__main__":
    print(f"Starting Scratch & Dust Removal UI at http://localhost:{config.GRADIO_PORT}")
    demo.launch(server_port=config.GRADIO_PORT, css=_CSS)
