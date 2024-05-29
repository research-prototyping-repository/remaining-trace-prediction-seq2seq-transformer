
# Work environment
# Import
import os
import pickle
import numpy as np

from joblib import load
from src.general.functions_time import tic, toc, get_timestamp
from src.data.functions_training_data import load_processed_data
from src.evaluation.functions_prediction import get_multi_dim_prediction
from src.general.functions_report import report_prediction_logistical_regression

# Set tensorflow to GPU-only (data is stored as tensors even when tf is not used)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Parameters
# Set path variables
path_raw = 'data/raw/'
path_interim = 'data/interim/'
path_benchmark = 'data/benchmark/'
path_data = 'data/processed/'
path_control = 'data/control/'
path_predictions = 'data/predictions/'
path_models = 'models/'
path_reports = 'reports/'

files = ['variables_helpdesk.pkl','variables_helpdesk_true.pkl','variables_helpdesk_false.pkl','variables_bpi2012.pkl','variables_bpi2012_true.pkl','variables_bpi2012_false.pkl','variables_bpi2017.pkl','variables_bpi2017_true.pkl','variables_bpi2017_false.pkl','variables_bpi2018.pkl','variables_bpi2018_true.pkl','variables_bpi2018_false.pkl','variables_bpi2019.pkl','variables_bpi2019_true.pkl','variables_bpi2019_false.pkl']

for filename_variables in files:

    with open(path_control + filename_variables, 'rb') as file:
        variables = pickle.load(file)

    # timestamp
    variables['logreg_timestamp_prediction_start'] = get_timestamp()


    # Preprocessing
    # Load interim data
    mapped_array = load_interim_data(path_interim + variables['filename_interim_dataset'])

    # Create input format
    x_input, y_input = create_input_format(mapped_array, variables['mapping'], variables['num_traces'], variables['max_length_trace'], variables['num_ex_activities'], num_features = variables['num_features'], benchmarking = True)

    # Train test split
    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x_input, y_input, train_size = 0.7, val_size = 0.15, test_size = 0.15)


    # Predicting
    logreg = load(path_models +'logistic_regression/'+ variables['regression_model'])
    tic()
    y_pred = get_multi_dim_prediction(logreg, x_test, variables['mapping'])
    variables['elapsed_time_predictions_logreg'] = toc()


    # Save predictions
    data_predictions = np.load(path_predictions + variables['filename_predictions'])
    data_predictions_copy = dict(data_predictions)
    data_predictions_copy['y_pred_rg'] = y_pred
    data_predictions_copy['y_test_benchmark'] = y_test
    np.savez(path_predictions + variables['filename_predictions'], **data_predictions_copy)
    print("Predictions saved to ", path_predictions + variables['filename_predictions'])

    # Generate report
    report_prediction_logistical_regression(filename_variables, variables, variables['logreg_timestamp_prediction_start'], path_reports)

    # Store variables in pickle file
    with open(path_control + filename_variables, 'wb') as file:
        pickle.dump(variables, file)
    print("Variables saved to ", path_control + filename_variables)

    del variables, y_pred, logreg, x_train, y_train, x_val, y_val, x_test, y_test


