# Detection parameters
THRESHOLD = 18      # Higher → fewer false positives; Lower → catch more (and more noise)
BLUR_SIGMA = 3      # Higher → detect wider scratches
MIN_AREA = 3        # Minimum connected-component area (px) to keep
MAX_AREA = 5000     # Maximum area to keep (larger regions are likely real detail)

# Resize limit for large images during detection
MAX_DIM = 4000      # Longest side; image is downscaled if larger, mask is upscaled back

# Folder layout (relative to project root)
INPUT_DIR = "input"
MASKS_DIR = "masks"
MASKS_VERIFIED_DIR = "masks_verified"
OUTPUT_DIR = "output"

# Detection — mask expansion
MASK_DILATE = 2             # Pixels to grow mask outward after detection.
                            # Ensures scratch edges are fully covered before inpainting.
                            # 0 = no dilation; increase if bụi/xước edges still visible.

# Inpainting
INPAINT_RADIUS = 5          # OpenCV TELEA inpaint radius (was 3; higher = smoother fill)
SMALL_AREA_THRESHOLD = 50   # Below this → OpenCV; above → LaMa (hybrid mode)

# Gradio UI
GRADIO_PORT = 7860
