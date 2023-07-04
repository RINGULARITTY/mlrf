import click
import os
import gzip
from display_lib import *
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from colorama import init

@click.command()
def main():
    """Visualize correlation of features
    """
    
    init()
    data_folder = "./data/processed"

    for feature in ["hog", "lbp"]:
        group_task(f"Processing for {feature}")

        sub_task("Load data")
        x_path = os.path.join(data_folder, f"x_{feature}_train.npy.gz")
        if not check_file(x_path, "make_dataset & build_features"):
            return
        with gzip.GzipFile(x_path, 'r') as f:
            x = np.load(f, allow_pickle=True)
        ok()
        
        t = Task("Create figure")

        corr = pd.DataFrame(x).corr()

        f, _ = plt.subplots(figsize=(11, 9), dpi=180)
        f.suptitle(f"Correlation of {feature}")
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, center=0, square=True, cbar_kws={"shrink": .5})

        t.stop()
        ok()

        plt.show()

if __name__ == "__main__":
    main()