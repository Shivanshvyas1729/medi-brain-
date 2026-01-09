from ultralytics import YOLO
from PIL import Image
import torch

model = YOLO("weights/best.pt")  # path to your trained model

def predict_image(image: Image.Image):
    results = model(image)
    return results[0]
