# Libraries
# -----------------------------------------------------------------------------
# Data libraries
import numpy as np
import pandas as pd

# math libraries
import math

# PM library
import pm4py

# Preprocessing libraries
from sklearn.model_selection import train_test_split

# Useability libraries
from tqdm import tqdm

# Functions
# ------------------------------------------------------------------------------
def load_data(filename, columns):

  """
  This function inputs the data and then prepares it for further use in a transformer encoder model.
  It takes csv and xes files and the filetype is automatically choosen based on the file extension.
  The columns indicate the selected columns from the original data in the output dataframe. Please pay attention to the order of the columns due to a renaming process to make it more standard.
  The data is automatically sorted in ascending order by the timestamp.

  Input:
  - filename = 'filename'
  - columns = ['time:timestamp', 'case:concept:name', 'concept:name', 'org:resource']: columns takes a list of all the needed columns,if they don't fit the standard scheme, they are renamed. Please pay attention to the order of columns in columns.

  Output:
  - df: dataframe containing the loaded and transformed data

  """
  # Import function specific libraries
  import datetime

  # Choose file type
  if filename[-4:] == ".csv":
    df = pd.read_csv(filename)
  elif ".xes" in filename:
    df = pm4py.read_xes(filename)
  else:
    raise ValueError('Allowed file extensions are only .csv and .xes.gz')

  # Rename columns
  standard_columns = ['time:timestamp', 'case:concept:name', 'concept:name', 'org:resource']
  new_column_names = {columns[i]: standard_columns[i] for i in range(len(columns))}

  df = df.rename(columns=new_column_names)
  # Include only the specified columns
  df = df[new_column_names.values()]

  # Column name timestamps
  column_name = 'time:timestamp'

  # Check if timestamps are in the right format and convert if not
  column_type = df[column_name].dtypes

  if column_type != 'datetime64[ns, UTC]':
      df[column_name] = pd.to_datetime(df[column_name])

  # Sort by case:concept:name and time:timestamp
  if column_name in new_column_names.values():
    df = df.sort_values(by=['case:concept:name', column_name])
  else:
    print('Data is not sorted by case and timestamp, because the column names don\'t correspont to the standard naming scheme.')

  return df

# ------------------------------------------------------------------------------

def data_cleaning(df, trace_length_min = 1):
  """
  This function cleans the data for further preprocessing.
  1. It removes all traces containing nan-values in one of the features
  2. It removes a certain percentile of traces with regards to the upper bound of the IQR method
  3. It removes traces bellow a directly set minimum length (if set to 0 no traces are removed)
  4. It removes all spaces in concept:name and replaces them with '-'


  Input:
  - df: dataframe containing the event log
  - trace_length_min [optional]: determins the minimum length of a trace and removes all traces below minimum length

  Output:
  - df_new: dataframe containing the cleaned data
  """
  # General variables
  length_traces = df['case:concept:name'].value_counts()

  # Remove all traces containing nan-values
  traces_with_nan = df[df.isna().any(axis=1)]['case:concept:name'].unique()
  df = df[~df['case:concept:name'].isin(traces_with_nan)]
  print("Remove traces containing nan:")
  print(f"{len(traces_with_nan)} traces removed." )
  print(f"{round(len(traces_with_nan)/len(df['case:concept:name'].unique()), 4)} % of traces removed.\n")

  # Remove upper bound outliers
  Q1 = length_traces.quantile(0.25)                                # Calculate Q1
  Q3 = length_traces.quantile(0.75)                                # Calculate Q3
  IQR = Q3 - Q1                                                          # Calculate the IQR
  upper_bound = Q3 + 1.5 * IQR                                           # Define the lower and upper bounds
  filtered_indices = length_traces[length_traces > upper_bound].index    # filter too long traces
  df = df[~df['case:concept:name'].isin(filtered_indices)]               # Remove traces that are too long
  print("Remove too long traces:")
  print(f"Upper bound of {upper_bound} applied.")
  print(f"Traces longer than {math.floor(upper_bound)} events removed.")
  print(f"{len(filtered_indices)} values removed.\n")

  # Remove lower bound outliers (traces with only one element)
  inital_length = df.shape[0]
  df = df[df['case:concept:name'].isin(length_traces[length_traces > trace_length_min].index)]
  print("Remove too short traces:")
  print(f"Traces shorter than {trace_length_min} events removed.")
  print(f"{inital_length-df.shape[0]} values removed.\n")

  # Remove all spaces in concept:name column
  df_new = df.copy()
  initial_spaces = df_new['concept:name'].str.count(' ').sum()
  df_new.loc[:,'concept:name'] = df['concept:name'].str.replace(' ', '-')
  print("Spaces in the concept:name column replaced by '-'.")
  print(f"{initial_spaces} values replaced.\n")
  del df

  return df_new

# ------------------------------------------------------------------------------

def load_interim_data(filename):
  """
  This function loads data from an .npz file if it is stored with the access key:
  - mapped_array

  Input:
  - filename: name of the .npz file

  Outout:
  - mapped_array: numpy array containing the encoded traces
  """

  # Load the saved data from the file
  data = np.load(filename, allow_pickle=True)

  # Access the array by its name
  mapped_array = data['mapped_array']

  # Output the shape
  print("\nmapped_array shape: ", mapped_array.shape)
  print("\n")

  return mapped_array

# ------------------------------------------------------------------------------

def extract_meta_data( df, case_id, features, output = True):
  """
  This function is for extracting relevant meta data from the dataset.

  This includes:
    - vocabulary
    - vocabulary size
    - max length of a trace
    - number traces
    - number of executed activites

  Input:
    - DataFrame
    - case_id: name of the column containing the ID of the case
    - features: list of column names used as input data for example
      - case_name: name of the column containing the activities
      - resource_name: name of the column containing the resource
    - output [optional]: boolean value indicating if meta data is printed


  Output:
    - vocab: numpy array containg the vocabulary
    - vocab_size: integer containing the size of the vocabulary
    - max_length_trace: integer containing the maximum possible trace length
    - num_traces: integer containing the number of traces
    - num_ex_activities: integer containing the number of executed activites
    - num_features: integer containing the number of used features


  Important notice:
  The vocabulary is expaneded to include the three tokens <start>, <end> and <pad>.
  This also increases the vocabulary size by 3 and the max_lenght_trace by 2.
  <start>: start of trace
  <end>: end of trace
  <pad>: used for padding to create the right input size
  """
  # Create the vocabulary
  vocabulary = np.array(["<pad>", "<unk>", "<start>", "<end>"])

  for feature in features:
    vocabulary = np.append(vocabulary, df[feature].unique())

  # Create a boolean mask and then apply the mask to drop string value
  mask = vocabulary != 'EMPTY'
  vocabulary = vocabulary[mask]

  # Drop nan's
  vocabulary = pd.DataFrame(vocabulary).dropna().values.flatten()

  # Determine the size of the vocabulary
  vocabulary_size = len(vocabulary)

  # Determine the lenght of a trace
  max_length_trace = df[case_id].value_counts().max()

  # Determine the number of traces
  num_traces = df[case_id].nunique()

  # Determine the number of executed activities
  num_ex_activities = df.shape[0]

  # Determine the number of features
  num_features = len(features)

  # Output of variables
  if output == True:
    print("Summary: ")
    print("vocab:                ", vocabulary[:6])
    print("vocab_size:           ", vocabulary_size)
    print("max_length_trace:     ", max_length_trace)
    print("num_traces:           ", num_traces)
    print("num_ex_activities:    ", num_ex_activities)
    print("\n")
    print("Features: ")
    print("num_features:         ", num_features)
    for feature in features:
      print("Feature:              ", feature)

  return vocabulary, vocabulary_size, max_length_trace, num_traces, num_ex_activities, num_features

# ------------------------------------------------------------------------------

def create_traces(df, features, interleave = True):
  """
  This function helps to create an array containing arrays of traces.

  Input:
  - df: DataFrame containing the input data
  - features: list containing the names of the features
  - interleave [optional]: boolean parameter that steers wether the features are interleaved or just rowwise appended

  Output:
  - group_array: Array containing arrays with the traces
  """
  # Grouping of the activites by case id
  groups_ids = df.groupby('case:concept:name')
  # Creating traces for each feature and saving the access to them in a list
  extraceted_features = [None] * len(features)

  for index, feature in enumerate(features):
    extraceted_features[index] = groups_ids[feature].apply(np.array).values

  # Conatecating the features so that the sub arrays are interleaved
  if len(features) == 1:
    traces = extraceted_features[0]
  else:
    if interleave == True:
      # Combine nested arrays
      combined_array = np.array([np.column_stack(sub_arrays) for sub_arrays in zip(*extraceted_features)], dtype = object)
      # Flatten the subarrays
      traces = np.array([np.reshape(sub_array, -1) for sub_array in combined_array], dtype=object)
    else:
      # Combine arrays row-wise
      traces = np.array([np.concatenate(temp_features) for temp_features in zip(*extraceted_features)], dtype = object)

  return traces

# ------------------------------------------------------------------------------

def tokenizer(data, vocab):
  """
  The function tokenizer creates a mapping dictionary which contains the words and the corresponding numerical expressions.

  Input:
  - data: pandas Series or nested numpy array containing the traces
  - vocab: numpy array containing the vocabulary

  Output:
  - tokenized_arry: array containing the sub arrays with the traces transformed into a numerical representation
  - mapping: dictionary containing the words as keys and the numbers as values
  """
  # Create the mapping dictionary
  mapping = {string: index for index, string in enumerate(vocab)}

  # Apply the mapping to the data and exclude negative values, which if occuring are (relative) timestamps
  tokenized_array = np.array([[mapping.get(element, mapping['<unk>']) for element in sub_array] for sub_array in tqdm(data, desc="Mapping")], dtype=object)
  tokenized_array = np.array([np.array(sub_array, dtype=object) for sub_array in tqdm(tokenized_array, desc="Processing Arrays")],dtype=object)

  return tokenized_array, mapping

# ------------------------------------------------------------------------------

def create_input_format(mapped_array, mapping, num_traces, max_length_trace, num_ex_activities, num_features = 1, interleave = False, benchmarking = False):
  """
  This function splits the traces in the right input shapes for the prediction of each step.
  Every splitted trace, respectivly every input and output trace, gets a <start> and an <end> token.
  The input_data and output_data is padded with the numerized <pad> token after the sequence.
  In benchmarking mode the sequences are sliced with y only containing a sequence with length 1.

  Input:
  - mapped array: array containing subarray with the tokenized traces
  - mapping: dictinary containing the mapping between tokens and numerical values
  - num_traces: int containing the number of traces
  - max_length_trace: int containing the most events in one trace
  - num_ex_activites: int containing the number of executed activites
  - num_features [optional]: int containing the number of features
  - interleave [optional]: boolean value determine the if interleaved or non interleaved traces are processed
  - benchmarking [optional]: boolean value determine

  Output:
  - x_input: input array containing a row for activity in each trace, where it is predicted
  - y_input: output array which contains the to predict string
  """
  # Calculate input  length and set array size
  input_length = max_length_trace*num_features + 2 # Because every splitted trace gets the <start> and <end> token
  x_input = np.empty((num_ex_activities*num_features, input_length))
  y_input = np.empty((num_ex_activities*num_features, input_length-1))
  row_number = 0

  if (interleave == True or num_features == 1) and benchmarking == False:

    # Iteratie over arrays
    for array in tqdm(mapped_array, desc="Processing Arrays"):

      for index, _ in enumerate(array):

        # Input array
        temp_x_array = np.concatenate(([mapping['<start>']], array[:index+1], [mapping['<end>']])) # Concatenate sliced array with special tokens
        temp_x_input = np.pad(temp_x_array, (0,input_length - np.size(temp_x_array)) , mode='constant', constant_values=mapping['<pad>']) # Padding of the sequence
        x_input[row_number] = temp_x_input
    
        # Output array
        temp_y_array = np.concatenate(([mapping['<start>']], array[index+1:], [mapping['<end>']]))# Concatenate sliced array with special tokens
        temp_y_input = np.pad(temp_y_array, (0,input_length - 1 - np.size(temp_y_array)) , mode='constant', constant_values=mapping['<pad>'])# Padding of the sequence
        y_input[row_number] = temp_y_input
    
        row_number += 1


  elif (interleave == False and num_features == 2) and benchmarking == False:
    for array in tqdm(mapped_array, desc="Processing Arrays"):

      # Slicing of the array into feature seperate arrays to more easily combine them
      temp_len = len(array)
      slicing_param = int(temp_len/2)
      events = array[:slicing_param]
      resources = array[slicing_param:]

      for i in range(0,int(len(events))):

        # Adding of an event
        # Input array
        temp_x_combination = np.concatenate((events[:i+1], resources[:i])) # Combine sequences
        temp_x_array = np.concatenate(([mapping['<start>']], temp_x_combination, [mapping['<end>']])) # Concatenate sequence with token
        temp_x_input = np.pad(temp_x_array, (0,input_length - np.size(temp_x_array)) , mode='constant', constant_values=mapping['<pad>']) # Pad sequence
        x_input[row_number] = temp_x_input
        # Output array
        temp_y_combination = np.concatenate((events[i+1:], resources[i:])) # Combine sequences
        temp_y_array = np.concatenate(([mapping['<start>']], temp_y_combination, [mapping['<end>']])) # Concatenate sequence with token
        temp_y_input = np.pad(temp_y_array, (0,input_length - 1 - np.size(temp_y_array)) , mode='constant', constant_values=mapping['<pad>']) # Pad sequence
        y_input[row_number] = temp_y_input
        row_number += 1
        
        
        # Adding of a resource
        # Input array
        temp_x_combination = np.concatenate((events[:i+1], resources[:i+1])) # Combine sequences
        temp_x_array = np.concatenate(([mapping['<start>']], temp_x_combination, [mapping['<end>']])) # Concatenate sequence with token
        temp_x_input = np.pad(temp_x_array, (0,input_length - np.size(temp_x_array)) , mode='constant', constant_values=mapping['<pad>']) # Pad sequence
        x_input[row_number] = temp_x_input
        # Output array
        temp_y_combination = np.concatenate((events[i+1:], resources[i+1:])) # Combine sequences
        temp_y_array = np.concatenate(([mapping['<start>']], temp_y_combination, [mapping['<end>']])) # Concatenate sequence with token
        temp_y_input = np.pad(temp_y_array, (0,input_length - 1 - np.size(temp_y_array)) , mode='constant', constant_values=mapping['<pad>']) # Pad sequence
        y_input[row_number] = temp_y_input
        row_number += 1


  elif benchmarking == True:

    # Insert <end> token at the end of each subarray
    mapped_array = np.array([np.insert(subarray, len(subarray), mapping['<end>']) for subarray in mapped_array], dtype = object)

    # Defining different properties based on the needs
    input_length = max_length_trace * num_features
    x_input = np.empty((num_ex_activities * num_features, input_length))
    y_input = np.empty((num_ex_activities * num_features, input_length))
    row_number = 0

    for array in tqdm(mapped_array, desc="Processing Arrays"):

      for i in range(1,len(array)):
        # Input array
        temp_x_array = array[:i] # Slice array
        temp_x_input = np.pad(temp_x_array, (input_length - np.size(temp_x_array), 0) , mode='constant', constant_values=mapping['<pad>']) # Pad sequence
        x_input[row_number] = temp_x_input

        # Output array
        temp_y_array = array[i:] # Slice array
        temp_y_input = np.pad(temp_y_array, (0, input_length - np.size(temp_y_array)) , mode='constant', constant_values=mapping['<pad>']) # Pad sequence
        y_input[row_number] = temp_y_input
        

        row_number += 1
  
  else:
    print("At the moment the function doesn't support the required functionality for the requested purpose.")

  return x_input, y_input

# ------------------------------------------------------------------------------

def train_val_test_split(x_input, y_input, train_size = 0.7, val_size = 0.15, test_size = 0.15, shuffle = False):
  """
  This function makes an train/val/test split of the input data. It takes only nested numpy arrays and no tf.Tensor data.

  Input:
  - x_input: nested arrays which contain the x_input data
  - y_input: nested arrays which contain the y_input data
  - train_size [optional]: float containing the train size
  - val_size [optional]: float containing the val size
  - test_size [optional]: float containing the test size
  - shuffle [optional]: boolean indicating if data is shuffled for set creation


  Output:
  """

  # Check if split adds up
  total = train_size + val_size + test_size
  if total != 1:
      raise ValueError("The split values do not add up to 1. Please check the determined sizes of the split.")

  # Determine the split sizes
  split1_size = 1 - train_size
  split2_size = val_size/(test_size + val_size)

  # Train test split
  x_train, x_temp, y_train, y_temp = train_test_split(x_input, y_input, test_size=split1_size, shuffle = shuffle)
  x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=split2_size, shuffle = shuffle)

  # Print summary
  print("Number of training samples:   ", x_train.shape[0])
  print("Number of validation samples: ", x_val.shape[0])
  print("Number of test samples:       ", x_test.shape[0])

  return x_train, y_train, x_val, y_val, x_test, y_test
