"""
Complete scratch-removal pipeline — single Gradio web UI.

Usage:
    python app.py
Then open http://localhost:7860

Workflow:
    Tab 1 · Detect & Edit      — upload images, auto-detect, then drag-brush to
                                 add/erase mask regions (brush = add, eraser = remove)
    Tab 2 · Inpaint & Download — run inpainting, download cleaned images
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
from verify_ui import load_mask, load_rgb

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
# Mask <-> ImageEditor-layer conversion
#
# The editor shows the photo as the background and the mask as an editable red
# layer. The user paints (brush = add red) or erases (eraser = remove). On save
# we read the layer's alpha channel back into a binary mask.
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS = {
    "threshold":   config.THRESHOLD,
    "blur_sigma":  config.BLUR_SIGMA,
    "min_area":    config.MIN_AREA,
    "max_area":    config.MAX_AREA,
    "mask_dilate": config.MASK_DILATE,
}
_DEFAULTS_TUPLE = tuple(_DEFAULT_PARAMS.values())

RED = (255, 0, 0, 255)


def mask_to_layer(mask_gray):
    """Binary mask -> RGBA layer: opaque red where mask, transparent elsewhere."""
    h, w = mask_gray.shape[:2]
    layer = np.zeros((h, w, 4), np.uint8)
    layer[mask_gray > 127] = RED
    return layer


def editor_value(img_path, mask_path):
    """Build the ImageEditor value: photo background + red mask layer (downscaled)."""
    photo = _downscale(load_rgb(img_path))
    mask  = _downscale(load_mask(mask_path), interp=cv2.INTER_NEAREST)
    return {"background": photo, "layers": [mask_to_layer(mask)], "composite": None}


def layer_to_mask(value):
    """Union of all layers' alpha -> binary mask (at editor/display resolution)."""
    layers = (value or {}).get("layers") or []
    if not layers:
        return None
    base = np.array(layers[0])
    acc = np.zeros(base.shape[:2], np.uint8)
    for layer in layers:
        arr = np.array(layer)
        if arr.ndim == 3 and arr.shape[2] == 4:
            acc[arr[:, :, 3] > 0] = 255
    return acc


def _params_tuple(p):
    pr = p.get("params", dict(_DEFAULT_PARAMS))
    return (pr["threshold"], pr["blur_sigma"], pr["min_area"], pr["max_area"], pr["mask_dilate"])


def _status(idx, pairs, p, prefix="Image"):
    return (f"{prefix} {idx + 1} / {len(pairs)}:  {Path(p['img_path']).name}  "
            f"— {p.get('n_regions', '?')} regions, {p.get('coverage_pct', 0):.1f}% affected")


# ---------------------------------------------------------------------------
# Tab 1 — Detect & Edit
# ---------------------------------------------------------------------------

def run_detection(uploaded_files, threshold, blur_sigma, min_area, max_area,
                  mask_dilate, progress=gr.Progress()):
    """Detect on all uploads, then load the first image into the editor."""
    if not uploaded_files:
        return None, "No files uploaded.", [], 0, *_DEFAULTS_TUPLE

    # Clear previous session data
    for p in list(DIR_INPUT.iterdir()) + list(DIR_MASKS.iterdir()) + list(DIR_OUTPUT.iterdir()):
        p.unlink(missing_ok=True)

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

    pairs, errors = [], []
    n = len(saved)
    for i, img_path in enumerate(saved):
        progress((i + 0.3) / n, desc=f"Detecting {img_path.name}")
        try:
            mask, n_regions, pct = detect_scratches(img_path, **params)
            mask_path = DIR_MASKS / (img_path.stem + "_mask.png")
            cv2.imwrite(str(mask_path), mask)
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

    if not pairs:
        msg = "Detection failed.\n" + "\n".join(errors)
        return None, msg, [], 0, *_DEFAULTS_TUPLE

    val = editor_value(pairs[0]["img_path"], pairs[0]["mask_path"])
    status = _status(0, pairs, pairs[0])
    if errors:
        status += "\n" + "\n".join(errors)
    return val, status, pairs, 0, *_params_tuple(pairs[0])


def load_editor(pairs, idx):
    """Return (editor_value, status, *sliders) for index idx."""
    if not pairs:
        return None, "Upload images and click 'Run Detection'.", *_DEFAULTS_TUPLE
    idx = max(0, min(idx, len(pairs) - 1))
    p = pairs[idx]
    return editor_value(p["img_path"], p["mask_path"]), _status(idx, pairs, p), *_params_tuple(p)


def _save_edits(value, pairs, idx):
    """Read the edited layer, upscale to full resolution, overwrite the mask."""
    if not pairs:
        return pairs, "Nothing to save."
    idx = max(0, min(idx, len(pairs) - 1))
    p = pairs[idx]
    mask_disp = layer_to_mask(value)
    if mask_disp is None:
        return pairs, "No mask layer found — nothing changed."

    full = load_mask(p["mask_path"])
    fh, fw = full.shape[:2]
    if mask_disp.shape[:2] != (fh, fw):
        mask_disp = cv2.resize(mask_disp, (fw, fh), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(p["mask_path"], mask_disp)

    n_regions = cv2.connectedComponents(mask_disp, connectivity=8)[0] - 1
    pct = float(mask_disp.sum()) / 255 / (fh * fw) * 100
    new_pairs = list(pairs)
    new_pairs[idx] = {**p, "n_regions": n_regions, "coverage_pct": round(pct, 2)}
    return new_pairs, n_regions


def apply_edits(value, pairs, idx):
    """Persist the current edits without navigating."""
    new_pairs, info = _save_edits(value, pairs, idx)
    if isinstance(info, str):
        return new_pairs, info
    return new_pairs, "Applied edits — " + _status(idx, new_pairs, new_pairs[idx])


def nav(value, pairs, idx, delta):
    """Save current edits, then move by delta and load that image."""
    if not pairs:
        return pairs, idx, None, "Upload images and click 'Run Detection'.", *_DEFAULTS_TUPLE
    new_pairs, _ = _save_edits(value, pairs, idx)
    new_idx = max(0, min(idx + delta, len(new_pairs) - 1))
    val, status, *sl = load_editor(new_pairs, new_idx)
    if delta > 0 and new_idx == idx:
        status = "Last image — all edits saved. Switch to 'Inpaint & Download'."
    return new_pairs, new_idx, val, status, *sl


def redetect(pairs, idx, threshold, blur_sigma, min_area, max_area, mask_dilate):
    """Re-run detection on the current image; reseed the mask layer (discards brush edits)."""
    if not pairs:
        return pairs, None, "Upload images and click 'Run Detection'."
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
        return pairs, editor_value(p["img_path"], p["mask_path"]), f"Re-detect failed: {exc}"

    new_pairs = list(pairs)
    new_pairs[idx] = {**p, "params": params, "n_regions": n_regions, "coverage_pct": round(pct, 2)}
    return (new_pairs,
            editor_value(p["img_path"], p["mask_path"]),
            _status(idx, new_pairs, new_pairs[idx], prefix="Re-detected"))


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
    cur_idx       = gr.State(0)

    # ── Tab 1 · Detect & Edit ────────────────────────────────────────────────
    with gr.Tab("1 · Detect & Edit"):
        with gr.Row(equal_height=False):
            # Left: upload + run + status.
            with gr.Column(scale=1, min_width=240):
                upload_files = gr.Files(
                    label="Images",
                    file_types=["image"],
                    file_count="multiple",
                    height=180,
                )
                detect_btn = gr.Button("Run Detection", variant="primary")
                status_box = gr.Textbox(label="Status", interactive=False,
                                        lines=5, elem_classes="compact-status")
            # Right: the brush workspace (focus).
            with gr.Column(scale=4):
                editor = gr.ImageEditor(
                    label="Edit mask — brush ✏️ = add (red), eraser = remove. Drag to paint.",
                    type="numpy",
                    image_mode="RGBA",
                    brush=gr.Brush(colors=["#FF0000"], color_mode="fixed", default_size=25),
                    eraser=gr.Eraser(),
                    sources=(),          # image comes from the pipeline, not direct upload
                    height=620,
                )

        # Navigation.
        with gr.Row():
            prev_btn  = gr.Button("← Previous", scale=1)
            apply_btn = gr.Button("Apply edits", variant="secondary", scale=1)
            next_btn  = gr.Button("Apply edits & Next →", variant="primary", scale=1)

        # Per-image re-detection (seeds the mask layer; brush refines it).
        with gr.Accordion("Adjust detection for this image", open=True):
            with gr.Row():
                sl_thresh = gr.Slider(1, 60,      value=config.THRESHOLD,   step=1,   label="Threshold — higher = fewer false positives")
                sl_blur   = gr.Slider(0.5, 10,    value=config.BLUR_SIGMA,  step=0.5, label="Blur Sigma — higher detects wider scratches")
                sl_min    = gr.Slider(1, 200,     value=config.MIN_AREA,    step=1,   label="Min Area (px)")
                sl_max    = gr.Slider(100, 30000, value=config.MAX_AREA,    step=100, label="Max Area (px)")
                sl_dil    = gr.Slider(0, 10,      value=config.MASK_DILATE, step=1,   label="Dilate (px) — expands mask edges")
            redetect_btn = gr.Button("Re-detect this image", variant="secondary")

        sliders = [sl_thresh, sl_blur, sl_min, sl_max, sl_dil]

        detect_btn.click(
            fn=run_detection,
            inputs=[upload_files, *sliders],
            outputs=[editor, status_box, session_pairs, cur_idx, *sliders],
        )
        next_btn.click(
            fn=lambda v, pairs, idx: nav(v, pairs, idx, +1),
            inputs=[editor, session_pairs, cur_idx],
            outputs=[session_pairs, cur_idx, editor, status_box, *sliders],
        )
        prev_btn.click(
            fn=lambda v, pairs, idx: nav(v, pairs, idx, -1),
            inputs=[editor, session_pairs, cur_idx],
            outputs=[session_pairs, cur_idx, editor, status_box, *sliders],
        )
        apply_btn.click(
            fn=apply_edits,
            inputs=[editor, session_pairs, cur_idx],
            outputs=[session_pairs, status_box],
        )
        redetect_btn.click(
            fn=redetect,
            inputs=[session_pairs, cur_idx, *sliders],
            outputs=[session_pairs, editor, status_box],
        )

    # ── Tab 2 · Inpaint & Download ───────────────────────────────────────────
    with gr.Tab("2 · Inpaint & Download"):
        gr.Markdown(
            "Tip: on Tab 1, click **Apply edits** (or *Apply edits & Next*) on the "
            "last image you brushed so its changes are saved before inpainting."
        )
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
