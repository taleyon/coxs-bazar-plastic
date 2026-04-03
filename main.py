from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
from PIL import Image
import io, base64, json

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
    img = Image.open(io.BytesIO(img_bytes))
    results = model(img)
    detections = []
    for box in results[0].boxes:
        detections.append({
            "label": model.names[int(box.cls)],
            "confidence": round(float(box.conf), 2),
            "bbox": box.xyxy[0].tolist()
        })
    annotated = results[0].plot()
    from PIL import Image as PILImage
    import numpy as np
    pil_img = PILImage.fromarray(annotated)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    return {
        "detections": detections,
        "count": len(detections),
        "annotated_image": img_base64
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")