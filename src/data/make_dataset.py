import click
import numpy as np
import gzip
import os
from termcolor import colored
from PIL import Image
import pickle
from sklearn.preprocessing import StandardScaler
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib')))
import display_lib as dl

@click.command()
def main():
    """ Merge datasets, extract X and y and export them.
    """
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def convert_to_image(data, index):
        single_image_data = data[index]
        single_image_data = single_image_data.reshape((3, 32, 32))
        single_image_data = single_image_data.transpose((1, 2, 0))

        img = Image.fromarray(single_image_data, 'RGB')
        return img

    dl.init()
    dl.group_task("Making train set")

    input_folder = "./data/raw"
    output_folder = "./data/processed"

    dl.sub_task("Read batches")
    data = []
    labels = []
    for i in range(5):
        batch = unpickle(f"{input_folder}/data_batch_{i+1}")
        data.append(batch[b"data"])
        labels.extend(batch[b"labels"])

    x_tmp, y = np.concatenate(data), np.array(labels)
    dl.ok()
    
    dl.sub_task("Convert to images")
    x = []
    for i in range(len(x_tmp)):
        x.append(convert_to_image(x_tmp, i))
    dl.ok()
    
    
    dl.sub_task("Export data")
    with gzip.open(os.path.join("./data/interim", 'x_train.pkl.gz'), 'wb') as f:
        pickle.dump(x, f)
    with gzip.GzipFile(os.path.join(output_folder, 'y_train.npy.gz'), 'w') as f:
        np.save(file=f, arr=y)
    dl.ok()

    dl.group_task("Making test set")

    dl.sub_task("Read batches")
    datatest = unpickle(f"{input_folder}/test_batch")
    x_temp_test, y_test = datatest[b"data"], datatest[b"labels"]
    dl.ok()
    
    dl.sub_task("Convert to images")
    x_test = []
    for i in range(len(x_temp_test)):
        x_test.append(convert_to_image(x_temp_test, i))
    dl.ok()

    dl.sub_task("Export data")
    with gzip.open(os.path.join("./data/interim", 'x_test.pkl.gz'), 'wb') as f:
        pickle.dump(x_test, f)
    with gzip.GzipFile(os.path.join(output_folder, 'y_test.npy.gz'), 'w') as f:
        np.save(file=f, arr=y_test)
    dl.ok()

if __name__ == '__main__':
    main()