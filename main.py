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

try:
    model = YOLO("best.pt")
    print("✅ Model loaded successfully")
    print(f"✅ Classes: {model.names}")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        if model is None:
            return {"error": "Model not loaded", "detections": [], "count": 0}

        results = model(img)
        
        detections = []
        for result in results:
            boxes = result.boxes
            masks = result.masks

            for i, box in enumerate(boxes):
                det = {
                    "label": model.names[int(box.cls)],
                    "confidence": round(float(box.conf), 2),
                    "bbox": box.xyxy[0].tolist()
                }
                if masks is not None and i < len(masks):
                    det["has_mask"] = True
                else:
                    det["has_mask"] = False
                detections.append(det)

        annotated = results[0].plot()
        pil_img = Image.fromarray(annotated)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        img_base64 = base64.b64encode(buf.getvalue()).decode()

        return {
            "detections": detections,
            "count": len(detections),
            "annotated_image": img_base64
        }

    except Exception as e:
        print(f"❌ Detection error: {e}")
        return {
            "error": str(e),
            "detections": [],
            "count": 0,
            "annotated_image": None
        }

app.mount("/", StaticFiles(directory="static", html=True), name="static")