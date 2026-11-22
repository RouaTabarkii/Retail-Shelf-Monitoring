import numpy as np
import cv2
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

MODEL_PATH = "C:/Users/LENOVO/Retail-Shelf-Monitoring/runs/baseline/weights/best.pt"
PERSON_MODEL_PATH = "yolo11n.pt"   

SEUIL = 100            
PERSON_CONF = 0.4      
PERSON_MARGIN = 20     
HIDDEN_RATIO = 0.2     


def hidden_by_person(zone, persons):
    """zone = (x1, y1, x2, y2). Vrai si une personne en recouvre assez."""
    zx1, zy1, zx2, zy2 = zone
    zone_area = max(1.0, (zx2 - zx1) * (zy2 - zy1))

    for px1, py1, px2, py2 in persons:
        px1 -= PERSON_MARGIN
        py1 -= PERSON_MARGIN
        px2 += PERSON_MARGIN
        py2 += PERSON_MARGIN

        ix = max(0.0, min(zx2, px2) - max(zx1, px1))
        iy = max(0.0, min(zy2, py2) - max(zy1, py1))
        if (ix * iy) / zone_area >= HIDDEN_RATIO:
            return True
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application starting...")
    app.state.model = YOLO(MODEL_PATH)
    app.state.person_model = YOLO(PERSON_MODEL_PATH)
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
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image")

    result = request.app.state.model(image, max_det=1000, conf=0.25, verbose=False)
    bounding_box = result[0].boxes.xyxy.cpu().numpy()

    person_result = request.app.state.person_model(
        image, classes=[0], conf=PERSON_CONF, verbose=False
    )
    person_boxes = person_result[0].boxes.xyxy.cpu().numpy()

    H, W = image.shape[0], image.shape[1]

    if len(bounding_box) == 0:
        return {
            "filename": file.filename, "num_boxes": 0, "gap_detected": False,
            "gaps": [], "etageres": {}, "personnes": person_boxes.tolist(),
            "gaps_ignores": 0, "image_height": H, "image_width": W,
        }

    bounding_box_sorted = np.array(sorted(bounding_box, key=lambda box: box[3]))
    y2_normalized = (bounding_box_sorted[:, 3] / H).reshape(-1, 1)

    labels = DBSCAN(eps=0.04, min_samples=2).fit_predict(y2_normalized)

    sorted_shelves = {}
    for i in set(labels):
        if i == -1:
            continue
        cluster = bounding_box_sorted[labels == i]
        sorted_shelves[i] = sorted(cluster, key=lambda box: box[0])

    gaps = []
    ignored = 0

    for shelf_id, products in sorted_shelves.items():
        for j in range(len(products) - 1):
            current_product = products[j]
            next_product = products[j + 1]

            gap = next_product[0] - current_product[2]
            if gap <= SEUIL:
                continue

            # Zone entre les deux produits voisins, pour tester la présence d'une personne
            zone = (
                float(current_product[2]),
                float(min(current_product[1], next_product[1])),
                float(next_product[0]),
                float(max(current_product[3], next_product[3])),
            )
            if hidden_by_person(zone, person_boxes):
                ignored += 1
                continue

            gaps.append({
                "etagere": int(shelf_id),
                "ecart_px": round(float(gap), 2),
                "gap_end": float(next_product[0]),
                "gap_start": float(current_product[2]),
                "y_haut": float(min(p[1] for p in products)),
                "y_bas": float(max(p[3] for p in products)),
            })

    return {
        "filename": file.filename,
        "num_boxes": len(bounding_box),
        "gap_detected": len(gaps) > 0,
        "gaps": gaps,
        "etageres": {
            int(k): [p.tolist() for p in v] for k, v in sorted_shelves.items()
        },
        "personnes": person_boxes.tolist(),
        "gaps_ignores": ignored,
        "image_height": H,
        "image_width": W,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)