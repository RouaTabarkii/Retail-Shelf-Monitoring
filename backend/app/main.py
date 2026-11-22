import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
import cv2
from ultralytics import YOLO
from contextlib import asynccontextmanager
import numpy as np
from sklearn.cluster import DBSCAN
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Application starting...")
    app.state.model = YOLO("C:/Users/LENOVO/Retail-Shelf-Monitoring/runs/baseline/weights/best.pt")

    yield

    print("Application shutting down...")


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    
    bounding_box = result[0].boxes.xyxy.cpu().numpy()                      
    bounding_box_sorted = np.array(sorted(bounding_box, key=lambda box: box[3]))  
    H = image.shape[0]

    y2_normalized = np.array([
        box[3] / H for box in bounding_box_sorted
    ]).reshape(-1, 1)

    dbscan = DBSCAN(
        eps=0.04,
        min_samples=2
    )

    labels = dbscan.fit_predict(y2_normalized)


    sorted_shelves = {}

    for i in set(labels):
        if i == -1:
            continue

        cluster = bounding_box_sorted[labels == i]

        sorted_shelves[i] = sorted(cluster, key=lambda box: box[0])


    SEUIL = 100
    gaps = []                                                              

    for shelf_id, products in sorted_shelves.items():

        for j in range(len(products) - 1):

            current_product = products[j]
            next_product = products[j + 1]

            gap = next_product[0] - current_product[2]

            if gap > SEUIL:                                               
                gaps.append({
                    "etagere": int(shelf_id),
                    "ecart_px": round(float(gap), 2),
                    "gap_end" : float(next_product[0]),
                    "gap_start" : float(current_product [2]),
                    "y_haut" : float(min(p[1] for p in products)),
                    "y_bas" : float(max(p[3] for p in products))
                })

    return {                                                               
        "filename": file.filename,
        "num_boxes": len(bounding_box),
        "gap_detected": len(gaps) > 0,
        "gaps": gaps,
        "etageres": {
            int(k): [p.tolist() for p in v] for k, v in sorted_shelves.items()
        },
        "image_height" : image.shape[0],
        "image_width" : image.shape[1]

    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)