
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn
import traceback
from pydantic import BaseModel
from typing import Optional

class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class PredictResponse(BaseModel):
    status: str
    confidence: Optional[float] = None
    kualitas: Optional[str] = None
    penyakit: Optional[str] = None
    deskripsi: Optional[str] = None
    bounding_box: Optional[BoundingBox] = None
    message: Optional[str] = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model_kualitas = YOLO("train21/weights/best.pt")        # T1–T4
model_penyakit_1 = YOLO("train6/weights/best.pt")       # Penyakit lama
model_penyakit_2 = YOLO("train-tea-obb2/weights/best.pt")  # Penyakit OBB terbaru

CONFIDENCE_THRESHOLD = 0.3

penyakit_deskripsi = {
    "Blister Blight": "Si perusak daun muda. Jamur Exobasidium vexans ...",
    "Brown Blight": "Jamur Colletotrichum camelliae ...",
    "Gray blight": "Bercak abu-abu kehitaman ...",
    "Red rust": "Bercak jingga kemerahan mencolok ..."
}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            return {"status": "Unknown", "message": "File yang diunggah bukan gambar"}

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        semua_prediksi = []

        # Model penyakit 1
        results_p1 = model_penyakit_1(image)
        if results_p1[0].boxes:
            for box in results_p1[0].boxes:
                c = float(box.conf)
                if c >= CONFIDENCE_THRESHOLD:
                    label = results_p1[0].names[int(box.cls)]
                    semua_prediksi.append({"label": label, "conf": c, "box": box})

        # Model penyakit 2
        results_p2 = model_penyakit_2(image)
        if results_p2[0].boxes:
            for box in results_p2[0].boxes:
                c = float(box.conf)
                if c >= CONFIDENCE_THRESHOLD:
                    label = results_p2[0].names[int(box.cls)]
                    semua_prediksi.append({"label": label, "conf": c, "box": box})

        # Model kualitas
        results_k = model_kualitas(image)
        if results_k[0].boxes:
            for box in results_k[0].boxes:
                c = float(box.conf)
                if c >= CONFIDENCE_THRESHOLD:
                    label = results_k[0].names[int(box.cls)]
                    semua_prediksi.append({"label": label, "conf": c, "box": box})

        if not semua_prediksi:
            return {"status": "Unknown", "message": "Gambar bukan daun teh atau tidak terdeteksi."}

        # Pilih prediksi dengan confidence tertinggi
        best = max(semua_prediksi, key=lambda x: x["conf"])
        label, conf, box = best["label"], best["conf"], best["box"]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Tentukan hasil berdasarkan label
        if label in ["T1", "T2", "T3", "T4"]:
            return {
                "status": "Healthy",
                "kualitas": label,
                "confidence": round(conf, 2),
                "bounding_box": {
                    "x": round(x1), "y": round(y1),
                    "width": round(x2 - x1), "height": round(y2 - y1)
                }
            }
        else:
            return {
                "status": "Sick",
                "penyakit": label,
                "confidence": round(conf, 2),
                "deskripsi": penyakit_deskripsi.get(label, "Deskripsi tidak ditemukan."),
                "bounding_box": {
                    "x": round(x1), "y": round(y1),
                    "width": round(x2 - x1), "height": round(y2 - y1)
                }
            }

    except Exception as e:
        traceback.print_exc()
        return {"status": "Error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("API:app", host="127.0.0.1", port=8000, reload=True)
