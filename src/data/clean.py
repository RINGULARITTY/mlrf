import click
from tqdm import tqdm
import os
from colorama import init
from termcolor import colored
from display_lib import *

@click.command()
def main():
    """Clean all processed data
    """
    init()
    res = confirm("Are you sure you want to delete all processed data ?")
    if res == "y":
        processed_data_folder = "./data/processed"

        files_to_delete = [fn for fn in os.listdir(processed_data_folder) if not fn.startswith('.')]
        if len(files_to_delete) == 0:
            ok()
            return

        for filename in tqdm(files_to_delete):
            file_path = os.path.join(processed_data_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                ko(str(e))
                return
        ok()

if __name__ == '__main__':
    main()