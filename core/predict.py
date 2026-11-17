import numpy as np
from sklearn.cluster import DBSCAN
import cv2
from ultralytics import YOLO
from backend.app.main import model

model =  YOLO("C:/Users/LENOVO/Retail-Shelf-Monitoring/runs/baseline/weights/best.pt")



source = "C:/Users/LENOVO/Retail-Shelf-Monitoring/data/samples/test_0.jpg"  
results = model(
    source,
    max_det=1000,
    conf=0.25,
    save=True,
    project="C:/Users/LENOVO/Retail-Shelf-Monitoring/predictions",
    name="test_0",
)

bounding_box = results[0].boxes.xyxy
bounding_box_sorted = sorted(bounding_box, key=lambda box: box[3])



image = cv2.imread(source)
if image is None:
    raise ValueError(f"Unable to read image: {source}")

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

for shelf_id, products in sorted_shelves.items():

    for j in range(len(products) - 1):

        current_product = products[j]
        next_product = products[j + 1]

        gap = next_product[0] - current_product[2]

        if gap > SEUIL:
            print(f"Gap détecté dans l'étagère {shelf_id} : {gap:.2f} pixels")