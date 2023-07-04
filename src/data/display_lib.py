import sys
from termcolor import colored

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