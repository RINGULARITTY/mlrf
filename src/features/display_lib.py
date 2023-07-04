import sys
from termcolor import colored
import os
from itertools import cycle
from threading import Thread
from time import sleep


def group_task(title):
    print(colored(f"{title} :", "magenta"))

def sub_task(task):
    print(colored(f"\t- {task}...", "cyan"), end=" ")
    sys.stdout.flush()

def ok():
    print(colored("Ok", "green"))

def ko(reason):
    print(f'{colored("Ko", "Red")} : {reason}')

def confirm(question):
    return input(colored(f"{question} (y/n)\n", "yellow"))

def check_file(file_path, step_to_do):
    if not os.path.exists(file_path):
        print(colored(f'You must do the "{step_to_do}" step before executing this script', 'red'))
        return False
    return True

def error(text):
    print(colored(text, "Red"))


class Task:
    def __init__(self, desc, is_sub=True, timeout=0.1):
        self.desc = desc
        self.is_sub = is_sub
        self.timeout = timeout

        self._thread = Thread(target=self.__animate, daemon=True)
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = False
        
        self.start()

    def start(self):
        self._thread.start()
        return self

    def __animate(self):
        for c in cycle(self.steps):
            if self.done:
                break
            if self.is_sub:
                print(colored(f"\r\t- {self.desc}... {c} ", "cyan"), flush=True, end="")
            else:
                print(colored(f"\r{self.desc}... {c} ", "magenta"), flush=True, end="")
            sleep(self.timeout)

    def stop(self):
        self.done = True