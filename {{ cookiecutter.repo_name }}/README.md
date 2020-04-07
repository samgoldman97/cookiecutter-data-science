{{cookiecutter.project_name}}
==============================

{{cookiecutter.description}}

Project Organization
------------
```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   
│   ├── dataset           <- Define dataloader
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── learning       <- Training various models
│   │
│   ├── models         <- Store model definitions
│   │   └── model_list.py <- List of models for export and calling in other scripts
│   │   
│   ├── scripts         <- Store training scripts and launchers
│   │   ├── launch_experiments.py <- Helper script to perform grid searches
│   │   └── generic_slurm.sh <- Example SLURM script to pass in other commands
│   │
│   ├── utils                <- Store various helper scripts 
│   │   └──  launcher_utils.py <- Helper script to parse json config files
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── tox.ini            <- tox file with settings for running tox; see tox.testrun.org
```

<p><small>Project based on a modified version of the <a target="_blank" href="https://github.com/samgoldman97/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
