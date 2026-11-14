import yaml
import pandas as pd
import mlflow


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Retail-Shelf-Monitoring")

freeze11 = "C:/Users/LENOVO/Retail-Shelf-Monitoring/runs/freeze11"
baseline = "C:/Users/LENOVO/Retail-Shelf-Monitoring/runs/baseline"




Yaml_11 = freeze11 + "/args.yaml"
result_11 = freeze11 + "/results.csv"
Yaml_baseline = baseline +"/args.yaml"
result_baseline = baseline + "/results.csv"



artifacts = {
    freeze11 + "/weights/best.pt": "weights",
    freeze11 + "/results.png": "plots",
    freeze11 + "/args.yaml": "config",
    freeze11 + "/results.csv": "config",
}

with open(Yaml_11, "r") as file:
    var = yaml.safe_load(file)
    print(var)


results = pd.read_csv(result_file) 


columns_to_keep = ["epoch","train/box_loss", 'train/cls_loss',"train/l1_loss","metrics/precision(B)","metrics/recall(B)","metrics/mAP50(B)", "metrics/mAP50-95(B)",
                   "val/box_loss" , "val/cls_loss" , "val/l1_loss"]
results = results[columns_to_keep]

results = results.rename(columns={
    "metrics/precision(B)": "metrics/precision",
    "metrics/recall(B)": "metrics/recall",
    "metrics/mAP50(B)": "metrics/mAP50",
    "metrics/mAP50-95(B)": "metrics/mAP50-95"
})

columns_to_log = ["train/box_loss", "train/cls_loss", "train/l1_loss", "metrics/precision", "metrics/recall", "metrics/mAP50", "metrics/mAP50-95",
                  "val/box_loss", "val/cls_loss", "val/l1_loss"]


with mlflow.start_run(
    run_name = "freeze11",
) :
    #lehne n7otou l code li bch yrani ki l experiment tebda
    mlflow.log_params(var)

    for index , row in results.iterrows():
        step = int(row["epoch"])
        metrics = {}

        for column in columns_to_log:
            metrics[column] = float(row[column])
        mlflow.log_metrics(metrics, step=step)

    for file_path, folder in artifacts.items():
        mlflow.log_artifact(file_path, artifact_path=folder) #the artifact_path y9lk esm l var li bch ytsajel bih l artifact f mlflow