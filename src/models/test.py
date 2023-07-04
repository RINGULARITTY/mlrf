import click
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import DMatrix
from seaborn import heatmap
import os
import json
import gzip
from joblib import load
from datetime import datetime
from time import time
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib')))
import display_lib as dl


@click.command()
@click.option("--delete_hist", default=False, help="Remove old results history")
def main(delete_hist):
    delete_hist = bool(delete_hist)
    time_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    def load_results(model, model_name, feature, first):
        model_params = f"{time_str}"
        results_output = f"./reports/results/{model_name}.json"
        if not os.path.exists(results_output) or (delete_hist and first):
            results = {model_params: {}}
        else:
            with open(results_output, "r") as f:
                results = json.load(f)
            if model_params not in results:
                results[model_params] = {}
            if not feature in results[model_params]:
                results[model_params][feature] = {}

        results[model_params][feature] = {}
        return results, model_params
    
    def save_results(model_name, results):
        results_output = f"./reports/results/{model_name}.json"
        with open(results_output, "w") as f:
            json.dump(results, f, indent=2)
    
    def compute_acc(y_test, y_pred, local_results):
        accuracy = accuracy_score(y_test, y_pred)
        local_results["acc"] = accuracy

    def compute_cm(y_test, y_pred, model, feature, local_results):
        cm = confusion_matrix(y_test, y_pred)
        cm = 100 * cm / np.sum(np.sum(cm))
        local_results["cm"] = cm.tolist()
        
        plt.figure(figsize=(10, 10), dpi=80)
        plt.title(f"{model.upper()} Confusion Matrix for {feature}")
        heatmap(cm, annot=True)
        plt.savefig(f"./reports/figures/{model}_{feature}_cm.png")
        plt.close()
    
    def compute_auc(n_classes, y_test_bin, y_pred_bin, model, feature, local_results):
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        local_results["auc"] = roc_auc
        plt.figure(figsize=(10, 10), dpi=80)
        plt.plot([0, 1], [0, 1], 'k--')
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f'Class {i}' + 'ROC curve (area = %0.2f)' % roc_auc[i])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model.upper()} ROC Curve for {feature}')
        plt.legend(loc="lower right")
        plt.savefig(f"./reports/figures/{model}_{feature}_auc.png")
        plt.close()

    def test(model, model_name, feature, x_test, y_test, first):
        results, model_params = load_results(model, model_name, feature, first)
        local_results = results[model_params][feature]

        classes = np.unique(y_test)
        
        landmark = time()
        if model_name == "xgboost":
            y_pred = model.predict(DMatrix(x_test))
        else:
            y_pred = model.predict(x_test)
        local_results["predict_duration"] = time() - landmark

        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        y_pred_bin = label_binarize(y_pred, classes=classes)

        compute_acc(y_test, y_pred, local_results)
        compute_cm(y_test, y_pred, model_name, feature, local_results)
        compute_auc(y_test_bin.shape[1], y_test_bin, y_pred_bin, model_name, feature, local_results)

        results[model_params][feature] = local_results
        save_results(model_name, results)

    dl.init()
    data_folder = "./data/processed"

    dl.group_task("Import data")
    dl.sub_task("Labels")

    y_path = os.path.join(data_folder, "y_test.npy.gz")
    if not dl.check_file(y_path, "data/make"):
        return
    with gzip.GzipFile(y_path, 'r') as f:
        y_test = np.load(f, allow_pickle=True)

    dl.ok()

    for model_name in ["svm", "kmeans", "xgboost"]:
        first = True
        for feature in ["flatten", "hog", "lbp"]:
            dl.group_task(f"Tests {model_name} for {feature}")

            t = dl.Task("Load data")
            x_path = os.path.join(data_folder, f"x_{feature}_test.npy.gz")
            if not dl.check_file(x_path, "features/build"):
                return
            with gzip.GzipFile(x_path, 'r') as f:
                x_test = np.load(f, allow_pickle=True)
            t.stop()
            dl.ok()
            
            t = dl.Task("Load model")
            model_path = os.path.join(f"./models/", f"{model_name}_{feature}.joblib")
            if not dl.check_file(model_path, "model/train"):
                return
            model = load(model_path)
            t.stop()
            dl.ok()

            t = dl.Task("Test")
            test(model, model_name, feature, x_test, y_test, first)
            t.stop()
            dl.ok()
            
            first = False

if __name__ == "__main__":
    main()