import click
import os
import requests
import tarfile
import shutil
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib')))
import display_lib as dl


@click.command()
def main():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    dl.init()
    dl.group_task("Get data")
    
    t = dl.Task("Downloading")
    response = requests.get(url)
    filename = url.split("/")[-1]
    with open(filename, 'wb') as f:
        f.write(response.content)
    t.stop()
    dl.ok()

    dl.sub_task("Extracting")
    with tarfile.open(filename) as tar:
        tar.extractall()
    dl.ok()

    folder_name = "cifar-10-batches-py"
    file_prefixes = ["data_batch_", "test_batch"]
    output_folder = "./data/raw"

    dl.sub_task("Copying files")
    for file in os.listdir(folder_name):
        for prefix in file_prefixes:
            if file.startswith(prefix):
                shutil.move(os.path.join(folder_name, file), os.path.join(output_folder, file))

    shutil.rmtree(folder_name)
    os.remove(filename)
    dl.ok()

if __name__ == "__main__":
    main()