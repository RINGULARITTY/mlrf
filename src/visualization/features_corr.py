import click
import os
import gzip
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib')))
import display_lib as dl

@click.command()
def main():
    """Visualize correlation of features
    """
    
    dl.init()
    data_folder = "./data/processed"

    for feature in ["hog", "lbp"]:
        dl.group_task(f"Processing for {feature}")

        dl.sub_task("Load data")
        x_path = os.path.join(data_folder, f"x_{feature}_train.npy.gz")
        if not dl.check_file(x_path, "make_dataset & build_features"):
            return
        with gzip.GzipFile(x_path, 'r') as f:
            x = np.load(f, allow_pickle=True)
        dl.ok()
        
        t = dl.Task("Create figure")

        corr = pd.DataFrame(x).corr()

        f, _ = plt.subplots(figsize=(11, 9), dpi=180)
        f.suptitle(f"Correlation of {feature}")
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, center=0, square=True, cbar_kws={"shrink": .5})

        t.stop()
        dl.ok()

        plt.show()

if __name__ == "__main__":
    main()