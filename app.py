from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import os
import shutil
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],




    
    allow_headers=["*"],
)

# ==========================
# Load YOLO Model
# ==========================

MODEL_PATH = os.path.abspath("best.pt")

print("=" * 60)
print("Current Working Directory :", os.getcwd())
print("Loading Model From        :", MODEL_PATH)
print("=" * 60)

model = YOLO(MODEL_PATH)

print("=" * 60)
print("Classes :", model.names)
print("=" * 60)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "Currency Detection API Running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print("\n")
    print("=" * 60)
    print("Uploaded Image :", file_path)

    img = cv2.imread(file_path)

    if img is None:
        return {
            "success": False,
            "predictions": [],
            "message": "Image could not be loaded."
        }

    print("Image Shape :", img.shape)

    results = model.predict(
        source=file_path,
        conf=0.85,
        verbose=False
    )

    predictions = []

    for result in results:

        print("Detected Boxes :", len(result.boxes))

        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = model.names[cls]

            print(f"Label : {label}")
            print(f"Confidence : {conf:.2f}")

            predictions.append({
                "label": label,
                "confidence": round(conf * 100, 2)
            })

    if len(predictions) == 0:

        print("No Currency Detected")

        return {
            "success": False,
            "predictions": [],
            "message": "No Currency Detected"
        }

    print("=" * 60)

    return {
        "success": True,
        "predictions": predictions
    }