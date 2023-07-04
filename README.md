MLRF
==============================

Projet Machine Learning et Reconnaissance de Formes

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `_` delimited description, e.g.
    │                         `1.0_initial_data_exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        |   ├── get_data.py
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   └── train_models.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            ├── data_mosaic.py
            ├── data_repartition.py
            └── features_corr.py

--------

Models pre-trained (KMeans excluded) are available here : https://drive.google.com/file/d/1q9bhnsDhqpZl_PN2W8x0wvwHtDiLXYAO/view?usp=sharing

_Make sure to put them in ./models folder._
## Exemple to run project :

#### Python setup 

First, you will have to install python and setup a virtual environment.

```bash
python -m venv venv
```
Then you will be able to activate it and install packages :
```bash
venv/Scripts/activate
pip install -r requirements.txt
```

#### Running scripts

##### Data part

_All commands must be done from the root path mlrf/>_

First let's download the data :
```bash
python ./src/data/get_data.py
```

We can then retreat it to make a temporary dataset :

```bash
python ./src/data/make_dataset.py
```

Let's have a look of what data looks like images in the train set.

- `label` [prompt] : Images with this label will be displayed
- `amount=100` : Amount of images to display

```bash
python ./src/visualization/data_mosaic.py --label=5 --amount=500
```

_All figures are saved in ./reports/figures_

We can also visualize labels repartition :
- `dataset=train` : Display train or test dataset.

```bash
python ./src/visualization/data_repartition.py
```

##### Features part

Now we can process features. You can compute Flatten, HOG and LBP :

- `features=all` : Choose features to build ex : flatten,hog

```bash
python ./src/features/build_features.py
```

To check features correlation :
```bash
python ./src/visualization/features_corr.py
```

##### Models training

Build and train us models :

- `hyper_params={"svm": {"tol": 1e-4, "C": 1.0, "max_iter": 50}, "k-means": {"n_neighbors": 10, "leaf_size": 100}, "xg-boost": {"max_depth": 20, "epochs": 50, "learning_rate": 0.1}}` : Dictionary of hyper-parameters
- `override=False` : Override model, else doesn't compute it

```bash
python ./src/models/train_models.py
```

_If you have downloaded pre-trained models and set override to False, it will not compute them._