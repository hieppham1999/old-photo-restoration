# Scratch & Dust Removal Tool

Tool tự động phát hiện và xoá vết xước, bụi trên ảnh cũ (bọc plastic, chụp lại).  
Chạy hoàn toàn local, không cần GPU (trừ khi dùng LaMa mode).

---

## Kiến trúc

```
Input ảnh
    │
    ▼
detect.py ──────► masks/          (mask PNG tự động)
    │
    ▼
verify_ui.py ───► masks_verified/ (mask sau khi người dùng sửa)
    │
    ▼
inpaint.py ─────► output/         (ảnh đã xử lý)
```

Mỗi bước là script độc lập, có thể chạy riêng hoặc chạy liên tiếp qua `batch.py`.

---

## Cài đặt

```bash
pip install -r requirements.txt
```

**Yêu cầu tối thiểu:** `opencv-python`, `numpy`, `Pillow`, `tqdm`  
**Verify UI:** thêm `gradio>=4.0.0`  
**Inpaint chất lượng cao (LaMa):** thêm `torch`, `torchvision`, `simple-lama-inpainting`

---

## Cách chạy

### Chạy nhanh — không verify

```bash
python batch.py --no-verify
```

### Chạy đầy đủ — có verify mask qua Gradio UI

```bash
python batch.py
```

Sau bước detect, trình duyệt tự mở tại `http://localhost:7860`.  
Xem, sửa mask, nhấn **Save & Next** cho từng ảnh. Đóng tab khi xong.

### Chạy từng bước thủ công

```bash
# Bước 1 — Detect
python detect.py --input ./input --output ./masks

# Bước 2 — Verify mask (Gradio)
python verify_ui.py --input ./input --masks ./masks --output ./masks_verified

# Bước 2 — Verify mask (OpenCV, không cần Gradio)
python verify_ui.py --input ./input --masks ./masks --output ./masks_verified --ui opencv

# Bước 3 — Inpaint
python inpaint.py --input ./input --masks ./masks_verified --output ./output
```

---

## Tham số quan trọng

### detect.py

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--threshold` | 18 | Ngưỡng phát hiện anomaly. Giảm → bắt nhiều hơn (thêm false positive). Tăng → bỏ sót scratch mờ. |
| `--blur-sigma` | 3 | Sigma Gaussian blur. Tăng → detect scratch rộng hơn. |
| `--min-area` | 3 | Bỏ qua vùng nhỏ hơn N pixel (noise). |
| `--max-area` | 5000 | Bỏ qua vùng lớn hơn N pixel (có thể là chi tiết thật). |
| `--dilate` | 2 | Mở rộng mask ra N pixel sau detect. Tăng nếu edge bụi/xước vẫn còn thấy sau inpaint. |

```bash
# Ví dụ: bắt nhiều hơn, mask rộng hơn
python detect.py --input ./input --output ./masks --threshold 14 --dilate 4
```

### inpaint.py

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--mode` | opencv | `opencv` = nhanh; `lama` = chất lượng cao; `hybrid` = tự động chọn theo kích thước vùng |

```bash
# Ví dụ: dùng LaMa cho xước dài
python inpaint.py --input ./input --masks ./masks_verified --output ./output --mode lama

# Hybrid: bụi nhỏ → opencv, xước lớn → lama
python inpaint.py --input ./input --masks ./masks_verified --output ./output --mode hybrid
```

### batch.py

```bash
python batch.py [options]

  --input          ./input          Folder ảnh gốc
  --output         ./output         Folder ảnh kết quả
  --masks          ./masks          Folder mask tự động
  --masks-verified ./masks_verified Folder mask đã verify
  --threshold      18               Ngưỡng detect
  --no-verify                       Bỏ qua bước verify
  --ui             gradio|opencv    UI backend cho verify
  --mode           opencv|lama|hybrid  Inpaint mode
```

---

## Verify UI — hướng dẫn

### Gradio (mặc định)

Chạy `verify_ui.py`, trình duyệt mở tại `http://localhost:7860`.

- **Ảnh trái:** ảnh gốc để tham khảo
- **Ảnh giữa:** overlay đỏ = vùng sẽ bị xoá
- **Ảnh phải:** mask có thể chỉnh sửa

Dùng brush để sửa mask:
- Vẽ **trắng** → thêm vùng cần xoá
- Vẽ **đen** → xoá vùng nhận nhầm (false positive)

### OpenCV (không cần Gradio)

| Phím | Tác dụng |
|---|---|
| `B` | Chế độ xoá mask (erase false positive) |
| `D` | Chế độ vẽ thêm mask |
| Scroll | Thay đổi kích thước brush |
| `+` / `-` | Zoom in / out |
| `S` | Save mask và qua ảnh tiếp theo |
| `Q` | Thoát |

---

## Inpaint modes

| Mode | Tốc độ | Chất lượng | Yêu cầu |
|---|---|---|---|
| `opencv` | Rất nhanh | Tốt cho bụi nhỏ, xước < 5px | Không cần thêm |
| `lama` | Chậm hơn (~5s/ảnh GPU, ~30s CPU) | Tốt cho xước dài, vùng lớn | `torch` + `simple-lama-inpainting` |
| `hybrid` | Trung bình | Tốt nhất tổng thể | Như LaMa |

Cài LaMa:
```bash
pip install torch torchvision simple-lama-inpainting
```

---

## Cấu trúc thư mục

```
scratch-removal/
├── config.py           # Tham số chung (threshold, paths, ...)
├── detect.py           # Bước 1: tự động detect
├── verify_ui.py        # Bước 2: UI verify/sửa mask
├── inpaint.py          # Bước 3: xoá theo mask
├── batch.py            # Chạy cả pipeline
├── requirements.txt
│
├── input/              # Ảnh gốc đưa vào đây
├── masks/              # Mask tự động (output của detect.py)
├── masks_verified/     # Mask sau khi verify
└── output/             # Ảnh đã xử lý
```

---

## Xử lý ảnh lớn

- Ảnh > 20MP nên xử lý theo tiles
- Detect chạy trên ảnh resize nhỏ hơn (max 4000px cạnh dài), mask tự scale ngược lại
- Inpaint chạy trên full resolution để giữ chất lượng
- RAM khuyến nghị: 16GB+ cho ảnh scan 20MP+

---

## Tuning khi kết quả chưa đạt

| Vấn đề | Cách xử lý |
|---|---|
| Bỏ sót xước mờ | Giảm `--threshold` (thử 14–16) |
| Quá nhiều false positive | Tăng `--threshold` hoặc dùng verify UI |
| Edge bụi vẫn còn thấy sau inpaint | Tăng `--dilate` (thử 3–5) |
| Vùng inpaint bị blur/không khớp màu | Dùng `--mode lama` hoặc `hybrid` |
| Vùng texture phức tạp (đất, đá) bị nhận nhầm | Dùng verify UI để xoá false positive |
