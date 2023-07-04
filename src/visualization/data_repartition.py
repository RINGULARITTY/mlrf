import click
from colorama import init
from display_lib import *
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import gzip
import os

@click.command()
@click.option('--dataset', default="train", help='Pick train or test set')
def main(train):
    """Visualize data repartition
    """
    init()
    
    group_task("Data repartition")
    sub_task("Load data")

    data_folder = "./data/processed"
    dataset_type = "train" if train == "train" else "test"

    y_path = os.path.join(data_folder, f"y_{dataset_type}.npy.gz")
    if not check_file(y_path, "make_dataset"):
        return
    with gzip.GzipFile(y_path, 'r') as f:
        y = np.load(f, allow_pickle=True)

    ok()
    sub_task("Making figure")

    label_freq = Counter(y)
    label_perc = {i: count/len(y)*100 for i, count in label_freq.items()}
    labels = label_perc.keys()
    sizes = label_perc.values()

    fig, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

    ax1.axis('equal')

    ax1.set_title(f'Labels repartition of {dataset_type} set')
    fig.savefig(f"./reports/figures/repartition_{dataset_type}_set.png")

    ok()
    plt.show()

if __name__ == '__main__':
    main()