import yaml
import pandas as pd
import mlflow
from pathlib import Path
from mlflow import MlflowClient

client = MlflowClient()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Retail-shelf-Monitoring")

freeze11 = "C:/Users/LENOVO/Retail-Shelf-Monitoring/runs/freeze11"
baseline = "C:/Users/LENOVO/Retail-Shelf-Monitoring/runs/baseline"
yolov11 = "C:/Users/LENOVO/Retail-Shelf-Monitoring/runs/yolov11"


models = ["freeze11", "baseline", "yolov11"]
model_path = [freeze11, baseline, yolov11]


columns_to_keep = [
    "epoch",
    "train/box_loss",
    "train/cls_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
]

columns_to_log = [
    "train/box_loss",
    "train/cls_loss",
    "metrics/precision",
    "metrics/recall",
    "metrics/mAP50",
    "metrics/mAP50-95",
    "val/box_loss",
    "val/cls_loss",
]


def run_dirs(run):

    Yaml_file = run + "/args.yaml"
    result_file = run + "/results.csv"

    artifacts = {
        run + "/weights/best.pt": "weights",
        run + "/results.png": "plots",
        run + "/args.yaml": "config",
        run + "/results.csv": "config",
    }

    with open(Yaml_file, "r") as file:
        var = yaml.safe_load(file)
        print(len(var))
    results = pd.read_csv(result_file)

    results = results[columns_to_keep]

    results = results.rename(
        columns={
            "metrics/precision(B)": "metrics/precision",
            "metrics/recall(B)": "metrics/recall",
            "metrics/mAP50(B)": "metrics/mAP50",
            "metrics/mAP50-95(B)": "metrics/mAP50-95",
        }
    )

    r_name = Path(run)
    with mlflow.start_run(
        run_name=r_name.name,
    ):
        # lehne n7otou l code li bch yrani ki l experiment tebda
        mlflow.log_params(var)

        for index, row in results.iterrows():
            step = int(row["epoch"])
            metrics = {}

            for column in columns_to_log:
                metrics[column] = float(row[column])
            mlflow.log_metrics(metrics, step=step)

        for file_path, folder in artifacts.items():
            mlflow.log_artifact(
                file_path, artifact_path=folder
            )  # the artifact_path y9lk esm l var li bch ytsajel bih l artifact f mlflow

        mlflow.log_artifact("C:/Users/LENOVO/Retail-Shelf-Monitoring/requirements.txt")

        for path in model_path:
            for model in models:
                mlflow.register_model(model_uri=path, name=model)
        alias = client.set_registered_model_alias(
            {model[0]: "low", model[2]: "champion", model[1]: "production"}
        )
        mlflow.pyfunc.load_model("models:/{model}@{alias}")
        client.set_model_version_tag(name=model, key="dataset", value="SKU-110k")


for model in models:
    run_dirs(model)
