# Working environment
# Import
import os
import numpy as np
import pickle

# Custom library
from src.data.functions_preprocessing_data import load_data, data_cleaning, extract_meta_data, create_traces, tokenizer, create_input_format, train_val_test_split
from src.data.functions_exploration_data import descriptive_statistics
from src.general.functions_report import report_preprocessing
from src.general.functions_time import get_timestamp

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

# List of all control files
files = ['variables_helpdesk.pkl','variables_helpdesk_true.pkl','variables_helpdesk_false.pkl','variables_bpi2012.pkl','variables_bpi2012_true.pkl','variables_bpi2012_false.pkl','variables_bpi2017.pkl','variables_bpi2017_true.pkl','variables_bpi2017_false.pkl','variables_bpi2018.pkl','variables_bpi2018_true.pkl','variables_bpi2018_false.pkl','variables_bpi2019.pkl','variables_bpi2019_true.pkl','variables_bpi2019_false.pkl']

# loop over every control file
for filename_variables in files:

    # Initiate control file
    print("filename_variables: ", filename_variables)
    path_file_control = os.path.join(path_control, filename_variables)

    if os.path.exists(path_file_control): # Check if the file already exists and then laod it, to not potentially cause data loss from training etc.
        print("Control file exists: ", path_file_control)
        from src.general.variables_control import variables
        with open(path_file_control, 'rb') as file:
            variables_old = pickle.load(file)
            variables.update(variables_old)  # Update mechanism in case the structure of variables was extended
    else:
        print(f"Control file not found: {path_file_control} \nNew control file creation initiated.")


    # Allocate datafile
    filesets = {'helpdesk':'helpdesk.csv', 'bpi2012':'BPI_Challenge_2012.xes.gz', 'bpi2017':'BPI_Challenge_2017.xes.gz', 'bpi2018':'BPI_Challenge_2018.xes.gz', 'bpi2019':'BPI_Challenge_2019.xes'}

    for key in filesets.keys():
        if key in filename_variables:
            variables['filename_dataset'] = filesets[key]
            break

    # Set filenames
    variables['filename_processed_dataset'] = 'preprocessed_data_'+ filename_variables[10:][:-4] + '.npz'
    variables['filename_predictions'] = 'predictions_' + filename_variables[10:][:-4] + '.npz'
    variables['filename_interim_dataset'] = 'interim_data_' + filename_variables[10:][:-4] + '.npz'
    variables['filename_benchmark_dataset'] = 'benchmark_data_' + filename_variables[10:][:-4] + '.npz'

    # Get timestamp
    variables['timestamp_preprocessing'] = get_timestamp()

    # Set params
    variables['trace_length_min'] = 1

    if 'false' in filename_variables:
        variables['interleave'] = False
        variables['features'] = ['concept:name', 'org:resource'] 
    elif 'true' in filename_variables:
        variables['interleave'] = True
        variables['features'] = ['concept:name', 'org:resource'] 
    else:
        variables['interleave'] = True # Dummy variable in this case
        variables['features'] = ['concept:name']

    # Input features
    if variables['filename_dataset'] == 'helpdesk.csv':
        variables['input_features'] = ['Complete Timestamp','Case ID','Activity', 'Resource'] # Helpdesk
    else:
        variables['input_features'] = ['time:timestamp','case:concept:name','concept:name', 'org:resource'] # Standard


    print(variables['filename_dataset'])
    print(variables['interleave'])
    print(variables['features'])


    # Load the dataset
    data = load_data(path_raw + variables['filename_dataset'], variables['input_features'])
    descriptive_statistics(data, variables['features'])


    # Data cleaning
    data = data_cleaning(data, variables['trace_length_min'])
    descriptive_statistics(data, variables['features'])

    # Extract the meta data
    variables['vocab'], variables['vocab_size'], variables['max_length_trace'], variables['num_traces'], variables['num_ex_activities'], variables['num_features'] = extract_meta_data(data,'case:concept:name',variables['features'])

    # Create the traces
    traces = create_traces(data, variables['features'], interleave = variables['interleave'])

    # Tokenize traces
    mapped_array, variables['mapping'] = tokenizer(traces, variables['vocab'])
    np.savez(path_interim + variables['filename_interim_dataset'], mapped_array=mapped_array)

    # Create the input format
    x_input, y_input = create_input_format(mapped_array, variables['mapping'], variables['num_traces'], variables['max_length_trace'], variables['num_ex_activities'], num_features = variables['num_features'], interleave = variables['interleave'])



    # Train test split
    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x_input, y_input, train_size = 0.7, val_size = 0.15, test_size = 0.15, shuffle = False)
    variables['x_train_shape'] = x_train.shape
    variables['x_val_shape'] = x_val.shape
    variables['x_test_shape'] = x_test.shape

    # Save the preprocessed data to file
    np.savez(path_data + variables['filename_processed_dataset'], x_train=x_train, y_train=y_train, x_val = x_val, y_val = y_val, x_test = x_test, y_test = y_test)



    # Check if predictions file exists
    if os.path.isfile(path_predictions + variables['filename_predictions']):
        # If the file exists, load it
        data_predictions = np.load(path_predictions + variables['filename_predictions'])
    else:
        np.savez(path_predictions + variables['filename_predictions'], y_test = y_test)
        data_predictions = np.load(path_predictions + variables['filename_predictions'])

    data_predictions_dict = dict(data_predictions)
    data_predictions_dict['y_test'] = y_test

    # Save the modified data back to the npz file
    np.savez(path_predictions + variables['filename_predictions'], **data_predictions_dict)



    # Store variables in pickle file
    with open(path_control + filename_variables, 'wb') as file:
        pickle.dump(variables, file)


    # Summary
    report_preprocessing(filename_variables, variables, variables['timestamp_preprocessing'], path_reports)


    del variables, data, traces, mapped_array, x_input, y_input, x_train, y_train, x_val, y_val, x_test, y_test,data_predictions


