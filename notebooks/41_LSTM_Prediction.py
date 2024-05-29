# Import libraries
import pickle
import numpy as np
import tensorflow as tf

from src.data.functions_training_data import load_processed_data
from src.general.functions_time import tic, toc, get_timestamp
from src.evaluation.functions_prediction import get_multi_dim_prediction
from src.general.functions_report import report_prediction_lstm


# Check available GPUs
print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Specify the GPU device
gpu_id = 0  # Use GPU 0
print("GPU used: ", gpu_id)

# Set GPU device options
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_visible_devices(physical_devices[gpu_id], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)

# Set path variables
path_raw = 'data/raw/'
path_interim = 'data/interim/'
path_benchmark = 'data/benchmark/'
path_data = 'data/processed/'
path_control = 'data/control/'
path_predictions = 'data/predictions/'
path_models = 'models/'
path_reports = 'reports/'

# List of all control files
files = ['variables_helpdesk.pkl','variables_helpdesk_true.pkl','variables_helpdesk_false.pkl','variables_bpi2012.pkl','variables_bpi2012_true.pkl','variables_bpi2012_false.pkl','variables_bpi2017.pkl','variables_bpi2017_true.pkl','variables_bpi2017_false.pkl','variables_bpi2018.pkl','variables_bpi2018_true.pkl','variables_bpi2018_false.pkl','variables_bpi2019.pkl','variables_bpi2019_true.pkl','variables_bpi2019_false.pkl']

# Iterate over control files
for filename_variables in files:
    
    # Print filename_variables
    print("Control file: ", filename_variables)
    # Load variables
    with open(path_control + filename_variables, 'rb') as file:
        variables = pickle.load(file)

    # timestamp
    variables['lstm_timestamp_prediction_start'] = get_timestamp()

    # Load benchmark data
    x_train, y_train, x_val, y_val, x_test, y_test = load_processed_data(path_benchmark + variables['filename_benchmark_dataset'], tensor = False)

    # Load lstm model
    lstm = tf.keras.models.load_model(path_models +'lstm/'+ variables['lstm_model'])

    # Generate predictions
    tic()
    y_pred = get_multi_dim_prediction(lstm, x_test, variables['mapping'], output_probabilites = True)
    variables['lstm_elapsed_time_predictions'] = toc()

    # Save predictions
    data_predictions = np.load(path_predictions + variables['filename_predictions'])
    data_predictions_copy = dict(data_predictions)
    data_predictions_copy['y_pred_lstm'] = y_pred
    data_predictions_copy['y_test_benchmark'] = y_test
    np.savez(path_predictions + variables['filename_predictions'], **data_predictions_copy)
    print("Predictions saved to ", path_predictions + variables['filename_predictions'])

    # Generate report
    report_prediction_lstm(filename_variables, variables, variables['lstm_timestamp_prediction_start'], path_reports)

    # Store variables in pickle file
    with open(path_control + filename_variables, 'wb') as file:
        pickle.dump(variables, file)

    del variables, lstm, x_train, y_train, x_val, y_val, x_test, y_test
