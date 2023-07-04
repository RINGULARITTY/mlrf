import click
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from joblib import dump
from sklearn.neighbors import KNeighborsClassifier
from xgboost import DMatrix, train
from colorama import init
from display_lib import *
import gzip
import numpy as np
import json
import warnings
import matplotlib.pyplot as plt
from PIL import Image

@click.command()
@click.option("--hyper_params", default='{"svm": {"tol": 1e-4, "C": 1.0, "max_iter": 50}, "k-means": {"n_neighbors": 10, "leaf_size": 100}, "xg-boost": {"max_depth": 20, "epochs": 50, "learning_rate": 0.1}}', help="Choose hyper-parameters")
@click.option("--override", default=False, help="Override model, else doesn't compute it")
def main(hyper_params, override):
    hyper_params = json.loads(hyper_params)
    epochs = hyper_params["xg-boost"].pop("epochs")
    hyper_params["xg-boost"]["tree_method"] = "gpu_hist"

    data_folder = "./data/processed"
    
    def SVM(x_train, y_train):
        svc = OneVsRestClassifier(svm.LinearSVC(random_state=42, **hyper_params["svm"]))
        svc.fit(x_train, y_train)
        return svc, None

    def KMeans(x_train, y_train):
        knn = KNeighborsClassifier(n_jobs=-1, **hyper_params["k-means"])
        knn.fit(x_train, y_train)
        return knn, None
    
    def XGBoost(x_train, y_train):
        dtrain = DMatrix(x_train, label=y_train)
        evals_result = {}
        xgboost = train(hyper_params["xg-boost"], dtrain, epochs, verbose_eval=False, evals=[(dtrain, 'train')], evals_result=evals_result)
        return xgboost, evals_result['train']['rmse']

    init()
    data_folder = "./data/processed"
    
    group_task("Import data")
    sub_task("Labels")

    y_path = os.path.join(data_folder, "y_train.npy.gz")
    if not check_file(y_path, "make_dataset"):
        return
    with gzip.GzipFile(y_path, 'r') as f:
        y_train = np.load(f, allow_pickle=True)

    ok()
    
    warnings.filterwarnings("ignore")
    losses = {}
    for model in [SVM, KMeans, XGBoost]:
        for feature in ["flatten", "hog", "lbp"]:
            model_path = f'models/{model.__name__.lower()}_{feature}.joblib'
            
            if not override and os.path.exists(model_path):
                continue
            
            group_task(f"Processing {model.__name__} for {feature}")
            
            t = Task("Load data")
            x_path = os.path.join(data_folder, f"x_{feature}_train.npy.gz")
            if not check_file(x_path, "make_dataset & build_features"):
                return
            with gzip.GzipFile(x_path, 'r') as f:
                x_train = np.load(f, allow_pickle=True)
            t.stop()
            ok()
            
            t = Task("Train model")
            trained_model, loss = model(x_train, y_train)
            t.stop()
            ok()
            
            sub_task("Save model")
            dump(trained_model, model_path)
            losses[feature] = loss
            ok()

    try:
        for feature in ["flatten", "hog", "lbp"]:
            plt.plot(losses[feature], label=feature)
            plt.plot(losses[feature], 'o')

        plt.title(f"XG-Boost RSME")
        plt.legend()
        plt.savefig(f"./reports/figures/xgboost_train.png")
        plt.show()
    except:
        img = Image.open(f"./reports/figures/xgboost_train.png")
        img.show()

if __name__ == "__main__":
    main()