# Import libraries
import numpy as np
import pandas as pd

import tensorflow as tf

import ast

# ------------------------------------------------------------------------------

# Functions
def load_variables(filename):
    """
    ------- V 1.0 -------
    This funtion load the following variables from a .csv file:
    - vocab
    - vocab_size
    - max_length_trace
    - num_traces
    - num_ex_activities
    - mapping
    - num_features
    - features
    - interleave
    - input_features
    - filename_dataset
    - filename_processed_dataset


    Input:
    - filename: name of the .csv file in which the variables are saved

    Output:
    - vocab: list which contains the vocabulary
    - vocab_size: integer which specifies the vocab size
    - max_length_trace: the maximum length of longest trace
    - num_ex_activities: the number of executed activites in total
    - mapping: dictionary which contains the mapping from the vocab to integers
    - num_features: integer which specifies the number of features
    - features: list which  contains the feature names
    """
    # Read file
    variables = pd.read_csv(filename, index_col = 0)

    # Load variables
    vocab = np.array(ast.literal_eval(variables.loc['vocab'][0]))
    vocab_size = int(variables.loc['vocab_size'][0])
    max_length_trace = int(variables.loc['max_length_trace'][0])
    num_traces = int(variables.loc['num_traces'][0])
    num_ex_activities = int(variables.loc['num_ex_activities'][0])
    mapping = ast.literal_eval(variables.loc['mapping'][0])
    num_features = int(variables.loc['num_features'][0])
    features = ast.literal_eval(variables.loc['features'][0])
    interleave = variables.loc['interleave'][0]
    input_features = ast.literal_eval(variables.loc['input_features'][0])
    filename_dataset = variables.loc['filename_dataset'][0]
    filename_processed_dataset = variables.loc['filename_processed_dataset'][0]

    # Print variables
    print("Summary:")
    print("\n")
    print("Dataset:             ", filename_dataset)
    print("\n")
    print("vocab (first 5):      ", vocab[:5])
    print("vocab_size:           ", vocab_size)
    print("max_length_trace:     ", max_length_trace)
    print("num_traces:           ", num_traces)
    print("num_ex_activities:    ", num_ex_activities)
    print("num_features:         ", num_features)
    print("features:             ", features)
    print("interleave:           ", interleave)
    print("\n")
    print("Filename Input data:  ", filename_processed_dataset)
    print("Filename variables:   ", filename)

    return vocab, vocab_size, max_length_trace, num_traces, num_ex_activities, mapping, num_features, features, interleave, input_features, filename_dataset, filename_processed_dataset

# ------------------------------------------------------------------------------

def load_processed_data(filename, tensor = True):
    """
    This function loads data from an .npz file if it is stored with the access keys:
    - x_train
    - y_train
    - x_val
    - y_val
    - x_test
    - y_test
    Then the data is converted into tensorflow tensors with the same name.

    Input:
    - filename: name of the .npz file
    - tensor [optional]: boolean variable, which determines if data output is tenor or array

    Output:
    
    >> Notice <<
    Output only as tensor if tensor = True, otherwise output as numpy array.

    - x_train_tensor: tensorflow vector containing the input taining data
    - y_train_tensor: tensorflow vector containing the output training data
    - x_val_tensor: tensorflow vector containing the input validation data
    - y_val_tensor: tensorflow vector containing the output validation data
    - x_test_tensor: tensorflow vector containing the input test data
    - y_test_tensor: tensorflow vector containing the output test data
    """

    # Load the saved arrays from the file
    data = np.load(filename)

    # Access arrays by their names
    x_train = data['x_train']
    y_train = data['y_train']
    x_val = data['x_val']
    y_val = data['y_val']
    x_test = data['x_test']
    y_test = data['y_test']

    # Converting them into tensorflow tensors if required
    if tensor == True:
        x_train_tensor = tf.convert_to_tensor(x_train)
        y_train_tensor = tf.convert_to_tensor(y_train)
        x_val_tensor = tf.convert_to_tensor(x_val)
        y_val_tensor = tf.convert_to_tensor(y_val)
        x_test_tensor = tf.convert_to_tensor(x_test)
        y_test_tensor = tf.convert_to_tensor(y_test)

        del x_train, y_train, x_val, y_val, x_test, y_test
    else:
        x_train_tensor = x_train
        y_train_tensor = y_train
        x_val_tensor = x_val
        y_val_tensor = y_val
        x_test_tensor = x_test
        y_test_tensor = y_test

    # Output the shape
    print("\nx_train_tensor shape: ", x_train_tensor.shape)
    print("y_train_tensor shape: ", y_train_tensor.shape)
    print("x_val_tensor shape:   ", x_val_tensor.shape)
    print("y_val_tensor shape:   ", y_val_tensor.shape)
    print("x_test_tensor shape:  ", x_test_tensor.shape)
    print("y_test_tensor shape: ", y_test_tensor.shape)
    print("\n")

    return x_train_tensor, y_train_tensor, x_val_tensor, y_val_tensor, x_test_tensor, y_test_tensor

# ------------------------------------------------------------------------------

def create_dataset(x_tensor, y_tensor, batch_size):
    """
    ------- V 1.5 -------
    This function takes a tf.Tensor and turns it into a tf.data.Dataset.
    Additionally the dataset is transformed to contain all the batches with the right size.
    For the dataset teacher forcing is applied to shift the sequences for model input.

    Input:
    - x_tensor: tf.Tensor containing the x input data
    - y_tensor: tf.Tensor containing the y input data
    - batch_size: int determine the batch size

    Output:
    - dataset: tf.data.Dataset containing data divided into batches with the determined batch_size
    """
    # Create dataset from tensors and create batches
    dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))
    dataset = dataset.batch(batch_size)

    # Teacher forcing
    dataset = dataset.map(lambda x, y: ((x, y[:, :-1]), y[:, 1:]))

    return dataset