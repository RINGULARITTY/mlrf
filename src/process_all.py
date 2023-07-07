import subprocess

def run_script(script_name):
    subprocess.run(["python", script_name])

def pipeline():
    for script in [
        "src/data/get.py", "src/data/make.py",       # Collect and pre-process data
        "src/features/build.py",                     # Create features
        "src/models/train.py", "src/models/test.py", # Train and test models
        "src/visualization/performances.py"          # Visualize test results
    ]:
        run_script(script)

if __name__ == "__main__":
    pipeline()