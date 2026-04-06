from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import io, base64, numpy as np

app = FastAPI()

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

model = YOLO("best.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # Run YOLOv11 + SAM2 inference
    results = model(img)
    
    detections = []
    for result in results:
        boxes = result.boxes
        masks = result.masks  # SAM2 segmentation masks

        for i, box in enumerate(boxes):
            det = {
                "label": model.names[int(box.cls)],
                "confidence": round(float(box.conf), 2),
                "bbox": box.xyxy[0].tolist()
            }
            # Add mask if available
            if masks is not None and i < len(masks):
                mask_array = masks.data[i].cpu().numpy()
                det["has_mask"] = True
            else:
                det["has_mask"] = False
            detections.append(det)

    # Generate annotated image with masks + boxes
    annotated = results[0].plot(
        masks=True,    # show SAM2 masks
        boxes=True,    # show bounding boxes
        labels=True,   # show labels
        conf=True      # show confidence
    )

    pil_img = Image.fromarray(annotated)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    img_base64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "detections": detections,
        "count": len(detections),
        "annotated_image": img_base64
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")