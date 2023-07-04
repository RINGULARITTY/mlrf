import click
import os
import requests
import tarfile
from display_lib import *
from colorama import init
import shutil

@click.command()
def main():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    init()
    group_task("Get data")
    
    t = Task("Downloading")
    response = requests.get(url)
    filename = url.split("/")[-1]
    with open(filename, 'wb') as f:
        f.write(response.content)
    t.stop()
    ok()

    sub_task("Extracting")
    with tarfile.open(filename) as tar:
        tar.extractall()
    ok()

    folder_name = "cifar-10-batches-py"
    file_prefixes = ["data_batch_", "test_batch"]
    output_folder = "./data/raw"

    sub_task("Copying files")
    for file in os.listdir(folder_name):
        for prefix in file_prefixes:
            if file.startswith(prefix):
                os.rename(os.path.join(folder_name, file), os.path.join(output_folder, file))

    shutil.rmtree(folder_name)
    os.remove(filename)
    ok()

if __name__ == "__main__":
    main()