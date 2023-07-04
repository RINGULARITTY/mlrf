import click
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib')))
import display_lib as dl


@click.command()
def main():
    """Clean all processed data
    """
    dl.init()
    processed_data_folder = "./data/interim"

    files_to_delete = [fn for fn in os.listdir(processed_data_folder) if not fn.startswith('.')]
    if len(files_to_delete) == 0:
        dl.ok()
        return

    for filename in tqdm(files_to_delete):
        file_path = os.path.join(processed_data_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            dl.ko(str(e))
            return
    dl.ok()

if __name__ == '__main__':
    main()