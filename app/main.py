from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from app.yolov8_handler import predict_image
from PIL import Image
import io


app = FastAPI(title="YOLOv8 Tumour Detection API")


@app.get("/")
def root():
    return {"message": "Shivansh YOLOv8 tumour detection API is live "}


# ✅ GET endpoint for browser users
@app.get("/predict")
def predict_info():
    return {
        "message": "Use POST /predict with an image file.",
        "usage": "Go to /docs and upload an image."
    }


# ✅ POST endpoint for actual prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = predict_image(image)

    boxes = result.boxes.xyxy.cpu().tolist()
    scores = result.boxes.conf.cpu().tolist()
    classes = result.boxes.cls.cpu().tolist()

    return JSONResponse(content={
        "boxes": boxes,
        "scores": scores,
        "classes": classes
    })
