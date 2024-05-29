
#  Working environment

# Import
import os
import numpy as np
import pickle

# Custom library
from src.general.variables_control import variables
from src.data.functions_preprocessing_data import load_interim_data, create_input_format, train_val_test_split
from src.general.functions_report import report_preprocessing_benchmark
from src.general.functions_time import get_timestamp


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

# List of all control files
files = ['variables_helpdesk.pkl','variables_helpdesk_true.pkl','variables_helpdesk_false.pkl','variables_bpi2012.pkl','variables_bpi2012_true.pkl','variables_bpi2012_false.pkl','variables_bpi2017.pkl','variables_bpi2017_true.pkl','variables_bpi2017_false.pkl','variables_bpi2018.pkl','variables_bpi2018_true.pkl','variables_bpi2018_false.pkl','variables_bpi2019.pkl','variables_bpi2019_true.pkl','variables_bpi2019_false.pkl']

# Preprocessing pipeline
# Iterate over all control files
for filename_variables in files:
    # Print filename
    print("Filename variables: ", filename_variables)

    # Initalize control file
    with open(path_control + filename_variables, 'rb') as file:
        variables = pickle.load(file)

    # Set filenames
    variables['filename_interim_dataset'] = 'interim_data_' + filename_variables[10:][:-4] + '.npz'
    variables['filename_benchmark_dataset'] = 'benchmark_data_' + filename_variables[10:][:-4] + '.npz'
        
    # Get timestamp
    variables['timestamp_preprocessing_benchmark'] = get_timestamp()

    # Load interim data
    mapped_array = load_interim_data(path_interim + variables['filename_interim_dataset'])

    # Create input format
    x_input, y_input = create_input_format(mapped_array, variables['mapping'], variables['num_traces'], variables['max_length_trace'], variables['num_ex_activities'], num_features = variables['num_features'], benchmarking = True)

    # Train test split
    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x_input, y_input, train_size = 0.7, val_size = 0.15, test_size = 0.15)

    # Extract only the first value, because of training structure
    y_train = y_train[:, :1]
    y_val = y_val[:, :1]

    # Save shapes
    variables['x_train_shape_benchmark'] = x_train.shape
    variables['x_val_shape_benchmark'] = x_val.shape
    variables['x_test_shape_benchmark'] = x_test.shape

    # Save the preprocessed data to file
    np.savez(path_benchmark + variables['filename_benchmark_dataset'], x_train=x_train, y_train=y_train, x_val = x_val, y_val = y_val, x_test = x_test, y_test = y_test)
    print("Data saved to: ", path_benchmark + variables['filename_benchmark_dataset'])

    # Check if predictions file exists
    if os.path.isfile(path_predictions + variables['filename_predictions']):
        # If the file exists, load it
        data_predictions = np.load(path_predictions + variables['filename_predictions'])
    else:
        # If not, create it
        np.savez(path_predictions + variables['filename_predictions'], y_test_benchmark = y_test)
        data_predictions = np.load(path_predictions + variables['filename_predictions'])

    data_predictions_dict = dict(data_predictions)
    data_predictions_dict['y_test_benchmark'] = y_test

    # Save the modified data back to the npz file
    np.savez(path_predictions + variables['filename_predictions'], **data_predictions_dict)
    print("Predictions saved to: ", path_predictions + variables['filename_predictions'])


    # Store variables in pickle file
    with open(path_control + filename_variables, 'wb') as file:
        pickle.dump(variables, file)
    print("Variables saved to: ", path_control + filename_variables)

    # Summary
    report_preprocessing_benchmark(filename_variables, variables, variables['timestamp_preprocessing_benchmark'], path_reports)


