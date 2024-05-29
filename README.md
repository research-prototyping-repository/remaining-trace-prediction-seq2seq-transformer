Remaining_Trace_Prediction_Master_Thesis_Stephan_Faatz
==============================

This project contains all the necessary files and data corresponding to the master thesis of remaining trace prediction with transformer networks.


Important notice
------------
To install the required packages please run:
pip:   pip install -r requirements.txt
- It is possible, that an error will occure and the remaining packages need to be installed manually.

To install this project as package (mandatory):
1. Navigate into the project directory in the console
2. Run: pip install -e .


Control file structure
------------
This project uses control files to store parameters and filenames to allow for easier access. 
Every dataset permutation has its corresponding control file which contains,e.g., name of the raw file, name of the processed file, transformer parameters, prediction duration.
It is created/udated (if already existing) during preprocessing.


Project Organization
------------


    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── benchmark      <- The datasets preprocessed for the competing artifacts
    │   ├── processed      <- The control files, which allow for easier data access and parameter transfer.
    │   ├── interim        <- The interim files, which save and interim state during preprocessing to allow for easier modification.
    │   ├── predictions    <- The final, canonical data sets for modeling.
    │   ├── processed      <- The final data sets for the transformer model.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Documentations and references regarding datasets.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   ├── logistic_regression        <- The logistic regression models
    │   ├── lstm                       <- The lstm models
    │   ├── random_forest              <- The random forest models (too big to share > 400 GB)
    │   └── transformer                <- The transformer models
    │       ├── hyperparameter_tuning_transformer_complexity          <- The resulting interim models of hyperparameter tuning
    │       ├── hyperparameter_tuning_transformer_embedding_dimension <- The resulting interim models of hyperparameter tuning
    │       └── hyperparameter_tuning_transformer_layers_heads        <- The resulting interim models of hyperparameter tuning
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number with regard to the process procedure 
    │
    ├── reports            <- Generated summary reports
    │   ├── evaluation     <- Generated evaluation reports
    │   │   └── figure                                        <- Generated figures 
    │   │   
    │   └── prediction     <- Generated prediction reports
    │   │   └── competing_artifacts                           <- Prediction reports of competing artifacts
    │   │       ├── logsitic_regression                       <- Prediction reports of logistic regression
    │   │       ├── lstm                                      <- Prediction reports of lstm
    │   │       └── random_forest                             <- Prediction reports of random forest
    │   │                 
    │   └── preprocessing  <- Generated preprocessing reports
    │   │   └── benachmark                                    <- Preprocessing reports for benchmark data
    │   │
    │   └── training       <- Generated training reports
    │       ├── competing_artifacts                           <- Training reports of competing artifacts
    │       │   ├── logsitic_regression                       <- Training reports of logistic regression
    │       │   ├── lstm                                      <- Training reports of lstm
    │       │   └── random_forest                             <- Training reports of random forest    
    │       └── hyperparameter_tuning_transformer_batch_size  <- Generated reports for hyperparameter tuning of batch size 
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to explore and preprocess data
    │   │   ├── __init__.py
    │   │   ├── functions_exploration_data.py      <- Functions for data exploration
    │   │   ├── functions_preprocessing_data.py    <- Functions for data preprocessing
    │   │   └── functions_training_data.py         <- Functions for training data  
    │   │
    │   ├── evaluation     <- Scripts to evaluate the prediction
    │   │   ├── __init__.py
    │   │   ├── functions_evaluation.py            <- Functions for evaluation
    │   │   └── functions_prediction.py            <- Functions for prediction
    │   │
    │   ├── general        <- Scripts containing general functions and data structures
    │   │   ├── __init__.py
    │   │   ├── functions_report.py                <- Functions for creating reports
    │   │   ├── functions_time.py                  <- Functions for measuring time
    │   │   └── variables_control.py               <- Data structure for control file
    │   │
    │   ├── models         <- Scripts containg the transformer model code base
    │   │   ├── __init__.py
    │   │   ├── transformer_model.py               <- Code base for transformer model
    │   │   └── transformer_predictor.py           <- Code base for predictor
    │   │
    │   └── visualization  <- Scripts to create visualizations
    │       ├── __init__.py
    │       ├── functions_evaluation_visualize.py  <- Functions to visualize evaluation
    │       ├── functions_exploration_visualize.py <- Functions to visualize exploration
    │       └── functions_training_visualize.py    <- Functions to visualize training
    │
    └── tools              <- Scripts and data to read files, create plots etc.
        └── figure         <- Generated figures
--------

Project structure based on the cookiecutter data science project template
Source: https://drivendata.github.io/cookiecutter-data-science/