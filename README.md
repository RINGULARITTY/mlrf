MLRF
==============================

Machine Learning and Shape Recognition

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── figures        <- Generated graphics and figures to be used in reporting
    |   └── results        <- Generated models results
    |
    └── src                <- Source code for use in this project.
        │
        ├── data           <- Scripts to download or generate data
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │
        ├── models         <- Scripts to train models and then use trained models to make predictions
        |
        └── visualization  <- Scripts to create exploratory and results oriented visualizations

--------

Models pre-trained (KMeans excluded) are available here : https://drive.google.com/file/d/1q9bhnsDhqpZl_PN2W8x0wvwHtDiLXYAO/view?usp=sharing

_Make sure to put them in ./models folder._

**All commands must be done from the root path mlrf/>**

### Python setup 

First, you will have to install python and setup a virtual environment.

```bash
python -m venv venv
```
Then you will be able to activate it and install packages :
```bash
venv/Scripts/activate
pip install -r requirements.txt
```
### Compute all

```bash
python ./src/process_all.py
```

### In depth commands

#### Data part

First let's download the data :
```bash
python ./src/data/get.py
```

You can then retreat it to make a temporary dataset :

- `train_batches=5`: Train batches amount to take

```bash
python ./src/data/make.py --train_batches=5
```

Let's have a look of what data looks like images in the train set.

- `label=0` : Images with this label will be displayed
- `amount=100` : Amount of images to display

```bash
python ./src/visualization/mosaic.py --label=5 --amount=500
```

_All figures are saved in ./reports/figures_

You can also visualize labels repartition :

- `dataset=train` : Display train or test dataset.

```bash
python ./src/visualization/repartition.py
```

#### Features part

Now you can process features. You can compute Flatten, HOG and LBP :

- `features=all` : Choose features to build ex : flatten,hog

```bash
python ./src/features/build.py
```

To check features correlation :
```bash
python ./src/visualization/corr.py
```

#### Models training

Build and train your models :

- `hyper_params={"svm": {"tol": 1e-4, "C": 1.0, "max_iter": 50}, "k-means": {"n_neighbors": 10, "leaf_size": 100}, "xg-boost": {"max_depth": 15, "epochs": 25, "learning_rate": 0.1}}` : Dictionary of hyper-parameters
- `override=False` : Override model, else doesn't compute it

```bash
python ./src/models/train.py
```

_Notice it will display loss history for XG-Boost model_

#### Models testing

You can now tests your models to get performance metrics. Each test will be save with date as key in ./reports/results

- `delete_hist=False`: Remove results history

```bash
python ./src/models/test.py
```

_If you have downloaded pre-trained models and set override to False, it will not compute them._

After models testing, you have a look on figures :

```bash
python ./src/visualization/performances.py
```

#### Additional commands

- Clean interim data

```bash
python ./src/data/clean.py
```