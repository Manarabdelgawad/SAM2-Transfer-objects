from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import numpy as np
import cv2
from PIL import Image
from ultralytics import SAM

app = FastAPI(title="SAM2")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"])

model = SAM("sam2.1_b.pt")
print("SAM2 model loaded successfully!\n")


def transfer_object(source_pil: Image.Image, bg_pil: Image.Image) -> Image.Image:
    source_np = np.array(source_pil)
    bg_np = np.array(bg_pil)

    results = model(source_pil)
    if not results[0].masks:
        raise ValueError("No object detected!")

    masks = results[0].masks.data.cpu().numpy()
    img_area = masks.shape[1] * masks.shape[2]
    areas = masks.sum(axis=(1, 2))
    valid = [i for i, a in enumerate(areas) if a < 0.70 * img_area]
    best_idx = max(valid, key=lambda i: areas[i]) if valid else 0

    mask = masks[best_idx]
    binary_mask = (mask > 0.5).astype(np.uint8)

    kernel = np.ones((5,5), np.uint8)
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    binary_mask = cv2.GaussianBlur(binary_mask.astype(np.float32), (5,5), 0)
    binary_mask = (binary_mask > 0.3).astype(np.uint8)

    extracted = source_np.copy()
    extracted[binary_mask == 0] = 0

    obj_h, obj_w = extracted.shape[:2]
    bg_h, bg_w = bg_np.shape[:2]
    scale = min(bg_w * 0.85 / obj_w, bg_h * 0.85 / obj_h, 1.0)

    if scale < 1.0:
        new_w = int(obj_w * scale)
        new_h = int(obj_h * scale)
        extracted = cv2.resize(extracted, (new_w, new_h), interpolation=cv2.INTER_AREA)
        binary_mask = cv2.resize(binary_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    x = (bg_w - extracted.shape[1]) // 2
    y = (bg_h - extracted.shape[0]) // 2

    result = bg_np.copy()
    h, w = extracted.shape[:2]
    roi = result[y:y+h, x:x+w]
    mask_3ch = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
    result[y:y+h, x:x+w] = np.where(mask_3ch == 1, extracted, roi)

    return Image.fromarray(result)

@app.get("/")
async def chech_api():
    return {"message": "welcome check post method"}



@app.post("/transfer")
async def transfer(source: UploadFile = File(...), background: UploadFile = File(...)):
    source_pil = Image.open(io.BytesIO(await source.read())).convert("RGB")
    bg_pil = Image.open(io.BytesIO(await background.read())).convert("RGB")

    result_image = transfer_object(source_pil, bg_pil)

    buf = io.BytesIO()
    result_image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


