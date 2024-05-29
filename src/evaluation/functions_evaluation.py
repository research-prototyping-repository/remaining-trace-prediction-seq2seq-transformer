# Import libraries
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from fastDamerauLevenshtein import damerauLevenshtein

# Function
# ------------------------------------------------------------------------------
def prediction_statistics(predicted_data, path_reports, model_name, timestamp, bins = 'auto', save = True, ref = False):
  """
  This function determines relevant statistic metrics for the predictions.
  Including the the predicted trace length and the output of it as a histogram.

  Input:
  - predicte_data: nested array containg the predicted multi-dimensional output data
  - path_reports: string containing the path for reports
  - model_name: string containing the filename of the used model
  - timestamp. string containing the timestamp
  - bins [optional]: int or str determining the distribution/number of bins
  - save [optional]: boolean save/not save plot
  - ref [optional]: boolean use it for the test/reference set

  Output:
  - predicted_trace_length: numpy array containing the trace length for every predicted trace
  - avg_pred_trace_length: float number containing the average of the predicted trace_length

  Warning:
  Please be aware that the <end> token is included in all these metrics.
  """

  # Calculate sizes of non-zero subarrays
  predicted_trace_length = np.argmax(predicted_data == 0, axis=1)

  # Calculate average size
  avg_pred_trace_length = np.mean(predicted_trace_length)

  # Print results
  if ref == False:
    print(f"Average prediction length for ({model_name}): {avg_pred_trace_length.round(4)}")
    print("Please be aware that the average prediction length contains the end token.\n")
  else:
    print("Average Trace length:", avg_pred_trace_length.round(4))
    print("Please be aware that the average trace length contains the end token.\n")

  # Plot histogram of predicted trace length
  plt.figure()
  plt.hist(predicted_trace_length, bins=bins, edgecolor='black')
  plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # Set the x-axis to display only integer ticks
  if ref == False:
    plt.xlabel('Size of predicted traces (inc. <end> token)')
    plt.title('Histogram of the predicted trace length: ' + str(model_name.split('.')[0]))
  else:
    plt.xlabel('Size of referece traces (inc. <end> token)')
    plt.title('Histogram of the reference trace length: '+ str(model_name.split('.')[0])) 
  plt.ylabel('Frequency')
  if save == True:
    plt.savefig(path_reports+ "evaluation/figures/" + timestamp + "_length_predicted_traces_" + model_name +'.png')
  plt.show()
  plt.close()

  return predicted_trace_length, avg_pred_trace_length


# ------------------------------------------------------------------------------

def damerau_levenshtein_similarity(y_test, y_pred, trace_length, trace_length_pred, length_indicator = 250):
    """Calculates the levenshtein similarity.

    Input:
    - y_test:numpy array containing the reference suffixes
    - y_pred:numpy array containing the predicted suffixes
    - trace_length:numpy array containing the trace length of the reference data
    - trace_length_pred:numpy array containing the trace length of the predicted data 
    - length_indicator: int number determining the upper limit of considered trace length (not included)

    Output:
    - similarity: array containing the damerau-levenshtein similarity  
    """  
    # Filter traces
    y_pred = y_pred[trace_length < length_indicator,:length_indicator]
    y_test = y_test[trace_length < length_indicator,:length_indicator]

    # Calculate similarity
    similarity = np.zeros(y_pred.shape[0])

    for i in tqdm(range(y_pred.shape[0]), desc = 'Comparing Sequences'):
      temp_similiarity = damerauLevenshtein(list(y_test[i,:trace_length[i]]), list(y_pred[i,:trace_length_pred[i]]), similarity = True)
      similarity[i] = temp_similiarity
      
    return similarity

# ------------------------------------------------------------------------------

def evaluate_seq_accuracy(y_pred, y_test):
    """
    This function evaluates the sequence accuracy based on precomputed predictions and ground truth.

    Input:
    - y_true: NumPy array containing true sequence values
    - y_pred: NumPy array containing predicted sequence values

    Output:
    - sequence_accuracy: float containing the sequence accuracy
    """

    # Transform the datatype of y_true to match y_pred
    y_test = y_test.astype(y_pred.dtype)

    # Match sequences
    sequence_match = np.all(y_test == y_pred, axis=1)
    sequence_match = sequence_match.astype(np.float32)

    correct_sequences = np.sum(sequence_match)
    total_sequences = y_pred.shape[0]

    # sequence accuracy
    sequence_accuracy = correct_sequences / total_sequences

    # Output
    print("Correct sequences: ", int(correct_sequences))
    print("Total sequences:   ", total_sequences)
    print("Sequence accuracy: ", round(sequence_accuracy, 4))

    return sequence_accuracy