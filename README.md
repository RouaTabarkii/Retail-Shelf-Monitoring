# Retail-Shelf-Monitoring

## 1. Overview :

A computer vision application that detects missing products on retail shelves and automatically notifies the stock manager to facilitate timely restocking.

## 2. Features : 

- Use computer vision to monitor retail shelves.
  
- Apply machine learning models to detect potentially missing products.
  
- Notify the stock manager when product shortages are detected.
  
- Provide a user-friendly interface to view, track, and manage alerts.
  
- Allow the stock manager to access captured shelf images to identify and verify detected product gaps.

## 3. Technologies :

- Fine tuninig Yolov26n

- Python

- React

- DVC

- MLflow

- Docker compose

- FastAPI

- ONNX

- DBSCAN

## 4. Installation :

- git clone http://RouaTabarkii/Retail-Shelf-Monitoring

- cd Retail-Shelf-Monitoring

- pip install -r requirements.txt

- dvc pull

## 5. DataSet : 

- Source : https://www.kaggle.com/datasets/thedatasith/sku110k-annotations

- Purpose: Used to fine-tune YOLOv11.

## 6. Project structure : 


```
Retail-Intelligence/
│
├── backend/
│   ├── app/
│   │   ├── inference.py       
│   │   ├── main.py            
│   │   └── schema.py          
│   │
│   ├── tests/
│   │   └── test_api.py         
│   │
│   └── requirements.txt        # Backend dependencies
│
├── core/
│   ├── gap_analysis.py         # Shelf gap detection logic
│   └── test_gab_rule.ipynb     # Notebook for testing gap detection rules
│
├── model/
│   └── best.pt                 # Trained YOLO model
│
├── training/
│   └── train.py                # Model training script
│
├── .gitignore
└── README.md
