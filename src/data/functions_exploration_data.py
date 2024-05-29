# Import libraries
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import LabelEncoder

from src.data.functions_preprocessing_data import extract_meta_data, create_traces, tokenizer

# Functions
# ------------------------------------------------------------------------------
def descriptive_statistics(df, features):
  """
  ------- V 1.0 -------
  This function calculates and prints descriptive statistics for the given dataframe.

  Inputs:
  - df: dataframe containing the data
  - features: list of features/columns to consider for analysis

  Output:
  - None
  """
  # Extract meta data
  vocabulary, vocabulary_size, max_length_trace, num_traces, num_ex_activities, num_features = extract_meta_data(df,'case:concept:name',features, output = False)

  # Calulation of the unique numbers of ressources
  num_unique_resources = df['org:resource'].nunique()

  # Count number of activites and resources
  num_activites = df['case:concept:name'].count()
  num_resources = df['org:resource'].count()


  # Calculation of the miniumum and average case length
  trace_length = df['case:concept:name'].value_counts()
  min_length_trace = trace_length.min()
  average_case_length = trace_length.mean().round(2)

  # Calculation of the trace length of the 99.99%, 99%, 95%, 75%, 50% and 25% percentile
  percentile_9999 = np.percentile(trace_length, 99.99)
  percentile_99 = np.percentile(trace_length, 99.00)
  percentile_95 = np.percentile(trace_length, 95.00)
  percentile_75 = np.percentile(trace_length, 75.00)
  percentile_50 = np.percentile(trace_length, 50.00)
  percentile_25 = np.percentile(trace_length, 25.00)

  # Number of unique traces
  # Concatenate 'concept:name' values for each trace
  unique_combinations = df.groupby('case:concept:name')['concept:name'].apply(tuple).unique()
  # Count the number of unique combinations
  num_unique_combinations = len(unique_combinations)

  # Variance of occurence
  # Concatenate 'concept:name' values for each trace
  unique_combinations = df.groupby('case:concept:name')['concept:name'].apply(tuple).tolist()
  # Count the occurrences of each unique combination
  combination_counts = pd.Series(unique_combinations).value_counts()
  # Calculate the variance of the occurrence counts
  variance_in_occurence_counts = combination_counts.var()

  # Variance of process
  # Create an instance of LabelEncoder and transform the data
  traces = df.groupby('case:concept:name')['concept:name'].apply(lambda x: np.array(x)).tolist()
  traces = np.concatenate(traces)
  label_encoder = LabelEncoder()
  encoded_data = label_encoder.fit_transform(traces)
  variance_in_process = encoded_data.var()

  # Feature nan values
  count_nan = []
  for index, feature in enumerate(features):
    count_nan.append(df[feature].isna().sum())

  # Output the statistics
  print("Number of activites:          ", num_activites)
  print("Number of resources:          ", num_resources)
  print("Unique activites:             ", vocabulary_size - 4)
  print("Unique resources:             ", num_unique_resources)
  print("Number of cases:              ", num_traces)
  print("Unique processes:             ", num_unique_combinations)
  print("Maximum case length:          ", max_length_trace)
  print("Minimum case length:          ", min_length_trace)
  print("Average case length:          ", average_case_length)
  print("99.99% percentile:            ", percentile_9999)
  print("99.00% percentile:            ", percentile_99)
  print("95.00% percentile:            ", percentile_95)
  print("75.00% percentile:            ", percentile_75)
  print("50.00% percentile:            ", percentile_50)
  print("25.00% percentile:            ", percentile_25)
  print('\n'.join([f"{feature} nan values:        {count_nan[index]}" for index, feature in enumerate(features)]))
  print("Variance of occurence counts: ", variance_in_occurence_counts)
  print("Variance in process:          ", variance_in_process)

# ------------------------------------------------------------------------------

def check_nan_traces(df, features):
  """
  This function checks the traces for nan values and gives explict information on the average nan value per trace where it occures.
  Further a histogram is plotted, which shows how many nan values there are per trace.

  Input:
  - df: dataframe containg the event log
  - features: list containing the used features

  Output:
  - None
  """
  # Create traces
  traces = create_traces(df, features)

  # Extract the meta data
  vocab, vocab_size, max_length_trace, num_traces, num_ex_activities, num_features = extract_meta_data(df,'case:concept:name',features, output = False)

  # Tokenize the traces
  mapped_array, mapping = tokenizer(traces, vocab)

  # Set the variables
  num_traces = len(mapped_array)
  num_nan = np.zeros(num_traces)

  # Iterate over subarrays
  for index, subarray in enumerate(mapped_array):
    for element in subarray:
      if element == 1: # 1 is the number of the unknow token
        num_nan[index] += 1

  num_traces_with_nan = np.count_nonzero(num_nan)

  if num_traces_with_nan != 0:

    # Plot the histogram
    num_nan_without_zeros = num_nan[num_nan != 0]

    print("\n\nnum_traces:          ", num_traces)
    print("num_traces_with_nan: ", num_traces_with_nan)
    print("num_nan:             ", num_nan)
    print("Average of nan per trace: ", num_nan_without_zeros.mean())

    # Plot the histogram
    plt.hist(num_nan_without_zeros, bins=21, edgecolor='black')

    plt.xlabel('Trace Length')
    plt.ylabel('Frequency')
    plt.title('Histogram of Trace Length containing only traces with nan')

    # Display the plot
    plt.show()

  else:
    print("\n\nNo NaN values have been detected.")

# ------------------------------------------------------------------------------

def count_unique_subarrays(nested_array):
    """
    Count the number of unique subarrays in a nested array.

    Input: 
    nested_array: numpy.ndarray containing the nested array

    Output:
    num_unique_subarrays: integer containing the number of unique subarrays
    """
    
    # Empty set to store unique subarrays
    unique_subarrays = set()  

    # Count unique subarray
    for i, subarray in tqdm(enumerate(nested_array), desc = "Counting"):
        # Add subarray if not already present in the set
        subarray_tuple = tuple(subarray)
        if subarray_tuple not in unique_subarrays:
            unique_subarrays.add(subarray_tuple)

    # Count array length
    num_unique_subarrays = len(unique_subarrays)

    return num_unique_subarrays

# ------------------------------------------------------------------------------

def count_unique_elements(nested_array):
    """
    Count the number of unique elements in a nested array.

    Input: 
    nested_array: numpy.ndarray containing the nested array

    Output:
    num_unique_subarrays: integer containing the number of unique elements
    """

    # Convert nested array to a single list of elements
    flattened_list = []
    for subarray in tqdm(nested_array, desc = "Counting"):
        flattened_list.extend(subarray)

    # Convert flattened list to a set to remove duplicates
    unique_elements = set(flattened_list)

    # Count unique elements using Counter
    unique_element_counts = Counter(unique_elements)

    # Calculate total count
    total_count = sum(unique_element_counts.values())

    return total_count