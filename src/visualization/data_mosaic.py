import click
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
from display_lib import *
from colorama import init
from PIL import Image
import pickle

@click.command()
@click.option('--label', prompt='Pick a label ', help='Label to display')
@click.option('--amount', default=100, help='Amount to display')
def main(label, amount):
    label = int(label)
    if label < 0 or label > 9:
        error("Must have 0<=label<=9")
        return

    init()
    group_task("Data mosaic")
    sub_task("Load data")

    data_folder = "./data/processed"

    x_path = os.path.join(data_folder, "x_train.pkl.gz")
    if not check_file(x_path, "make_dataset"):
        return
    with gzip.open(x_path, 'rb') as f:
        x = pickle.load(f)

    y_path = os.path.join(data_folder, "y_train.npy.gz")
    if not check_file(y_path, "make_dataset"):
        return
    with gzip.GzipFile(y_path, 'r') as f:
        y = np.load(f, allow_pickle=True)

    ok()

    sub_task("Making figure")

    images = x[y == 1]
    images = images[np.random.choice(len(images), size=amount, replace=False)]

    mosaic_dim = int(np.sqrt(len(images))) + 1

    width, height = images[0].size
    mosaic_image = Image.new('RGB', (width * mosaic_dim, height * mosaic_dim))

    # Parcourir chaque image et la copier dans la mosaïque
    for i, img in enumerate(images):
        x = i % mosaic_dim
        y = i // mosaic_dim
        mosaic_image.paste(img, (x * width, y * height))

    # Afficher la mosaïque
    plt.title(f"Mosaic of {amount} images where label={label}")
    plt.imshow(mosaic_image)
    plt.axis('off')
    ok()
    
    plt.show()
if __name__ == '__main__':
    main()