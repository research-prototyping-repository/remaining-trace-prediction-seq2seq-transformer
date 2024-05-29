# Work environment
# Import
import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
from src.general.functions_report import report_training_random_forest
from src.general.functions_time import tic, toc, get_timestamp
from src.data.functions_training_data import load_processed_data
# Set tensorflow to GPU-only (data is stored as tensors even when tf is not used)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set path variables
path_raw = 'data/raw/'
path_interim = 'data/interim/'
path_benchmark = 'data/benchmark/'
path_data = 'data/processed/'
path_control = 'data/control/'
path_predictions = 'data/predictions/'
path_models = 'models/'
path_reports = 'reports/'

# List of (all) control files
files = ['variables_helpdesk.pkl','variables_helpdesk_true.pkl','variables_helpdesk_false.pkl','variables_bpi2012.pkl','variables_bpi2012_true.pkl','variables_bpi2012_false.pkl','variables_bpi2017.pkl','variables_bpi2017_true.pkl','variables_bpi2017_false.pkl','variables_bpi2018.pkl','variables_bpi2018_true.pkl','variables_bpi2018_false.pkl','variables_bpi2019.pkl','variables_bpi2019_true.pkl','variables_bpi2019_false.pkl']

# Iterate over all control files
for filename_variables in files: 

    # Print filenme
    print("Filename: ", filename_variables)

    # Initalize variables
    with open(path_control + filename_variables, 'rb') as file:
        variables = pickle.load(file)

    # Timestamp 
    variables['rf_timestamp_training_start'] = get_timestamp()

    # Set model name
    variables['random_forest_model'] = "rf_"+ filename_variables[10:][:-4] +".joblib"

    # Preprocessing
    # Load benchmark data
    x_train, y_train, x_val, y_val, x_test, y_test = load_processed_data(path_benchmark + variables['filename_benchmark_dataset'], tensor = False)

    # Model parameter
    variables['rf_n_estimators'] = 100
    variables['rf_n_jobs'] = -1

    # Training
    # Initalize and train model
    rf = RandomForestClassifier(n_estimators = variables['rf_n_estimators'], max_depth = 12, random_state = 29061998, verbose = 1, n_jobs= variables['rf_n_jobs'])
    tic()
    rf.fit(x_train, y_train.flatten())
    variables['rf_elapsed_time'] = toc()

    # Save model
    dump(rf, path_models + 'random_forest/' + variables['random_forest_model'])

    # ## Evaluation
    y_val_pred = rf.predict(x_val)
    variables['rf_acc'] = accuracy_score(y_val.flatten(), y_val_pred)
    print('acc: ',variables['rf_acc'])

    # Generate report
    report_training_random_forest(filename_variables, variables, variables['rf_timestamp_training_start'], path_reports)

    # Store variables in pickle file
    with open(path_control + filename_variables, 'wb') as file:
        pickle.dump(variables, file)

    del variables, y_val_pred, rf, x_train, y_train, x_val, y_val, x_test, y_test
