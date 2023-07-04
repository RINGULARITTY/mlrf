import click
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import gzip
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib')))
import display_lib as dl

@click.command()
@click.option('--dataset', default="train", help='Pick train or test set')
def main(dataset):
    """Visualize data repartition
    """
    dl.init()
    
    dl.group_task("Data repartition")
    dl.sub_task("Load data")

    data_folder = "./data/processed"

    y_path = os.path.join(data_folder, f"y_{dataset}.npy.gz")
    if not dl.check_file(y_path, "make_dataset"):
        return
    with gzip.GzipFile(y_path, 'r') as f:
        y = np.load(f, allow_pickle=True)

    dl.ok()
    dl.sub_task("Making figure")

    label_freq = Counter(y)
    label_perc = {i: count/len(y)*100 for i, count in label_freq.items()}
    labels = label_perc.keys()
    sizes = label_perc.values()

    fig, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

    ax1.axis('equal')

    ax1.set_title(f'Labels repartition of {dataset} set')
    fig.savefig(f"./reports/figures/repartition_{dataset}_set.png")

    dl.ok()
    plt.show()

if __name__ == '__main__':
    main()