import click
import numpy as np
from skimage import color
from skimage.feature import hog
from skimage import feature
from sklearn.preprocessing import StandardScaler
import gzip
import pickle
import warnings
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib')))
import display_lib as dl

@click.command()
@click.option("--features", default="all", help="Choose features to build ex : flatten,hog,lbp")
def main(features):
    if features == "all":
        features = ["flatten", "hog", "lbp"]
    else:
        features = features.split(",")

    def extract_hog_features(images):
        hog_features_list = []

        for img in images:
            img = np.array(img)
            img_gray = color.rgb2gray(img)

            hog_features = hog(img_gray, orientations=16, pixels_per_cell=(8, 8),
                            cells_per_block=(4, 4), visualize=False, transform_sqrt=True)
            hog_features_list.append(hog_features)

        return hog_features_list
    
    def extract_lbp_features(X, P=32, R=4):
        lbp_features = []
        for img in X:
            img_gray = img.convert('L')
            img_arr = np.array(img_gray)
            lbp = feature.local_binary_pattern(img_arr, P, R)
            hist, _ = np.histogram(lbp, density=True, bins=P, range=(0, P - 1))
            hist[np.isnan(hist)] = 0
            lbp_features.append(hist)
        return lbp_features
    
    def flatten_images(images):
        flattened_images = []
        for img in images:
            flattened_img = np.array(img).flatten()
            flattened_images.append(flattened_img)
        return flattened_images
    
    dl.init()
    dl.group_task("Import data")
    dl.sub_task("Train set")
    
    data_folder = "./data/processed"
    
    x_train_path = os.path.join("./data/interim", "x_train.pkl.gz")
    if not dl.check_file(x_train_path, "data/make"):
        return
    with gzip.open(x_train_path, 'rb') as f:
        x_train = pickle.load(f)

    dl.ok()
    dl.sub_task("Test set")

    x_test_path = os.path.join("./data/interim", "x_test.pkl.gz")
    if not dl.check_file(x_test_path, "data/make"):
        return
    with gzip.open(x_test_path, 'rb') as f:
        x_test = pickle.load(f)

    dl.ok()
    
    warnings.filterwarnings("ignore")
    for f in features:
        dl.group_task(f"Processing for {f}")
        
        dl.sub_task("Calculate for train and test")
        if f == "flatten":
            x_f_train = flatten_images(x_train)
            x_f_test = flatten_images(x_test)
        elif f == "hog":
            x_f_train = extract_hog_features(x_train)
            x_f_test = extract_hog_features(x_test)
        elif f == "lbp":
            x_f_train = extract_lbp_features(x_train)
            x_f_test = extract_lbp_features(x_test)
        dl.ok()
        
        dl.sub_task("Normalization")

        scaler = StandardScaler()
        scaler.fit(x_f_train)
        x_f_train = scaler.transform(x_f_train)
        x_f_test = scaler.transform(x_f_test)
        
        dl.ok()
        
        t = dl.Task("Export data")
        with gzip.GzipFile(os.path.join(data_folder, f'x_{f}_train.npy.gz'), 'w') as file:
            np.save(file=file, arr=x_f_train)
        with gzip.GzipFile(os.path.join(data_folder, f'x_{f}_test.npy.gz'), 'w') as file:
            np.save(file=file, arr=x_f_test)
        t.stop()
        dl.ok()

if __name__ == '__main__':
    main()