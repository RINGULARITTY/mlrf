import click
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from joblib import dump
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import differential_evolution
from sklearn.model_selection import train_test_split
from xgboost import DMatrix, train
import os
import gzip
import numpy as np
import json
import warnings
import matplotlib.pyplot as plt
from PIL import Image
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib')))
import display_lib as dl


@click.command()
@click.option("--hyper_params", default='{"svm": {"tol": 1e-4, "C": 1.0, "max_iter": 50}, "kmeans": {"n_neighbors": 10}, "xgboost": {"max_depth": 35, "epochs": 50, "learning_rate": 0.1}}', help="Choose hyper-parameters")
@click.option("--override", default=False, help="Override model, else doesn't compute it")
def main(hyper_params, override):
    hyper_params = json.loads(hyper_params)
    epochs = hyper_params["xgboost"].pop("epochs")
    hyper_params["xgboost"]["tree_method"] = "gpu_hist"

    data_folder = "./data/processed"

    def SVM(x_train, y_train):
        svc = OneVsRestClassifier(svm.LinearSVC(random_state=42, **hyper_params["svm"]))
        svc.fit(x_train, y_train)
        return svc, None

    def KMeans(x_train, y_train):
        bounds = [(15, 150)]
        x_train_subset, x_test_subset, y_train_subset, y_test_subset = train_test_split(x_train, y_train, test_size=0.05, random_state=42)

        best_values = []
        def callback(xk, convergence):
            value = objective(xk)
            best_values.append(value)
            print(f'Iteration: {len(best_values)}, Best Value: {value}, Convergence: {convergence}')
            return True

        def objective(hyperparameters):
            hyperparameters = [int(x) for x in hyperparameters]
            knn = KNeighborsClassifier(leaf_size=hyperparameters[0], n_jobs=-1, **hyper_params["kmeans"])
            knn.fit(x_train_subset, y_train_subset)
            return -np.mean(knn.predict(x_test_subset) == y_test_subset)

        result = differential_evolution(objective, bounds, popsize=100, maxiter=3, updating='deferred', callback=callback, tol=1e-18, polish=False)
        leaf_size = int(result.x[0])
        print(f"Best leaf_size : {leaf_size}")
        
        model = KNeighborsClassifier(leaf_size=leaf_size, n_jobs=-1, **hyper_params["kmeans"])
        model.fit(x_train, y_train)
        return model, None

    def XGBoost(x_train, y_train, x_test, y_test):
        hyper_params["xgboost"]["objective"] = "multi:softmax"
        hyper_params["xgboost"]["num_class"] = len(np.unique(y_train))
        
        dtrain = DMatrix(x_train, label=y_train)
        dtest = DMatrix(x_test, label=y_test)
        evals_result = {}
        xgboost = train(hyper_params["xgboost"], dtrain, epochs, verbose_eval=5, evals=[(dtrain, 'train'), (dtest, 'test')], evals_result=evals_result)
        return xgboost, evals_result

    dl.init()
    data_folder = "./data/processed"
    
    dl.group_task("Import data")
    dl.sub_task("Labels")

    y_path = os.path.join(data_folder, "y_train.npy.gz")
    if not dl.check_file(y_path, "make_dataset"):
        return
    with gzip.GzipFile(y_path, 'r') as f:
        y_train = np.load(f, allow_pickle=True)

    y_path = os.path.join(data_folder, "y_test.npy.gz")
    if not dl.check_file(y_path, "make_dataset"):
        return
    with gzip.GzipFile(y_path, 'r') as f:
        y_test = np.load(f, allow_pickle=True)

    dl.ok()
    
    warnings.filterwarnings("ignore")
    losses = {}
    for model in [SVM, KMeans, XGBoost]:
        for feature in ["flatten", "hog", "lbp"]:
            model_path = f'models/{model.__name__.lower()}_{feature}.joblib'
            
            if not override and os.path.exists(model_path):
                continue
            
            dl.group_task(f"Processing {model.__name__} for {feature}")
            
            t = dl.Task("Load data")
            x_path = os.path.join(data_folder, f"x_{feature}_train.npy.gz")
            if not dl.check_file(x_path, "features/build"):
                return
            with gzip.GzipFile(x_path, 'r') as f:
                x_train = np.load(f, allow_pickle=True)
            t.stop()
            dl.ok()
            
            if model == XGBoost:
                x_path = os.path.join(data_folder, f"x_{feature}_test.npy.gz")
                if not dl.check_file(x_path, "features/build"):
                    return
                with gzip.GzipFile(x_path, 'r') as f:
                    x_test = np.load(f, allow_pickle=True)
            
            t = dl.Task("Train model")
            if model == XGBoost:
                trained_model, loss = model(x_train, y_train, x_test, y_test)
            else:
                trained_model, loss = model(x_train, y_train)
            t.stop()
            dl.ok()
            
            dl.sub_task("Save model")
            dump(trained_model, model_path)
            losses[feature] = loss
            dl.ok()

    try:
        colors = {
            "flatten_train": 'red',
            "flatten_test": 'green',
            "hog_train": 'orange',
            "hog_test": 'blue',
            "lbp_train": 'purple',
            "lbp_test": 'black',
        }

        for feature in ["flatten", "hog", "lbp"]:
            plt.plot(losses[feature]['train']['mlogloss'], '-o', color=colors[f"{feature}_train"], label=f"train {feature}")
            plt.plot(losses[feature]['test']['mlogloss'], '-o', color=colors[f"{feature}_test"], label=f"test {feature}")

        plt.title(f"XG-Boost MLogLoss")
        plt.legend()
        plt.savefig(f"./reports/figures/xgboost_train.png")
        plt.show()
    except:
        img = Image.open(f"./reports/figures/xgboost_train.png")
        img.show()

if __name__ == "__main__":
    main()