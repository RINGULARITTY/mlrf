import click
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import warnings
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib')))
import display_lib as dl

@click.command()
@click.option('--label', default=0, help='Images with this label will be displayed')
@click.option('--size', default=9, help='Size of tile (9 means 81 images)')
def main(label, size):
    label = int(label)
    if label < 0 or label > 9:
        dl.error("Must have 0<=label<=9")
        return

    dl.init()
    dl.group_task("Data mosaic")
    dl.sub_task("Load data")

    data_folder = "./data/processed"

    x_path = os.path.join("./data/interim", "x_train.pkl.gz")
    if not dl.check_file(x_path, "data/make"):
        return
    with gzip.open(x_path, 'rb') as f:
        x = pickle.load(f)

    y_path = os.path.join(data_folder, "y_train.npy.gz")
    if not dl.check_file(y_path, "data/make"):
        return
    with gzip.GzipFile(y_path, 'r') as f:
        y = np.load(f, allow_pickle=True)

    dl.ok()

    dl.sub_task("Making figure")


    warnings.filterwarnings("ignore")
    images = np.array(x, dtype=object)[y == label]
    images = images[np.random.choice(len(images), size=size*size, replace=False)]

    width, height = images[0].size
    mosaic_image = Image.new('RGB', (width * size, height * size))

    for i, img in enumerate(images):
        x = i % size
        y = i // size
        mosaic_image.paste(img, (x * width, y * height))

    plt.title(f"Mosaic of {size}x{size} images where label={label}")
    plt.imshow(mosaic_image)
    plt.axis('off')
    plt.savefig(f"./reports/figures/mosaic_{label}_{size}x{size}.png")
    dl.ok()
    plt.show()

if __name__ == '__main__':
    main()