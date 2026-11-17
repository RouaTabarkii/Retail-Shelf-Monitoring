import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
import cv2
from ultralytics import YOLO
from contextlib import asynccontextmanager
import numpy as np


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Application starting...")
    app.state.model = YOLO("C:/Users/LENOVO/Retail-Shelf-Monitoring/runs/baseline/weights/best.pt")

    yield

    print("Application shutting down...")


app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Not an image")
    contents = await file.read()
    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image"
    )

    result = request.app.state.model(
        image,
        max_det=1000,
        conf=0.25,
        save=True,
        project="C:/Users/LENOVO/Retail-Shelf-Monitoring/predictions",
        name="test_0",)
    
    return {"filename": file.filename , "num_boxes" : len(result[0].boxes)}




if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)