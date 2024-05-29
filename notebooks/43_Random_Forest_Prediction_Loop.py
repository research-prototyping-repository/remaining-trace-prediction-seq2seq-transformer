
# Work environment
# Import
import os
import pickle
import numpy as np

from joblib import load
from src.general.functions_time import tic, toc, get_timestamp
from src.data.functions_training_data import load_processed_data
from src.evaluation.functions_prediction import get_multi_dim_prediction
from src.general.functions_report import report_prediction_random_forest

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
    variables['rf_timestamp_prediction_start'] = get_timestamp()

    # Preprocessing
    # Load benchmark data
    x_train, y_train, x_val, y_val, x_test, y_test = load_processed_data(path_benchmark + variables['filename_benchmark_dataset'], tensor = False)

    # Predicting
    # Load model
    rf = load(path_models +'random_forest/'+ variables['random_forest_model'])
    # Get multidimensional predictions
    tic()
    y_pred = get_multi_dim_prediction(rf, x_test, variables['mapping'])
    variables['rf_elapsed_time_predictions'] = toc()


    # Save predictions
    data_predictions = np.load(path_predictions + variables['filename_predictions'])
    data_predictions_copy = dict(data_predictions)
    print(data_predictions_copy['y_test'].shape)
    data_predictions_copy['y_pred_rf'] = y_pred
    data_predictions_copy['y_test_benchmark'] = y_test
    np.savez(path_predictions + variables['filename_predictions'], **data_predictions_copy)
    print("Predictions saved to ", path_predictions + variables['filename_predictions'])


    # Generate report
    report_prediction_random_forest(filename_variables, variables, variables['rf_timestamp_prediction_start'], path_reports)


    # Store variables in pickle file
    with open(path_control + filename_variables, 'wb') as file:
        pickle.dump(variables, file)
        
    del variables, y_pred, rf, x_train, y_train, x_val, y_val, x_test, y_test


