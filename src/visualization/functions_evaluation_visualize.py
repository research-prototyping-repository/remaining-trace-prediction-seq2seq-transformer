# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------

# Functions
def accuracy_sequence_histogram(true_output_data, predicted_data):
  """
  This function plots a hsitogram of the correct predicted sequences and their length, compared to all predictions.

  Input:
  - true_outptut_data: nested array containg the multi-dimensional expected output data
  - predicte_data: nested array containg the predicted multi-dimensional output data

  Output:
  - None
  """
  # Set variables
  correct_sequences = []

  # Transform true_output_data
  true_output_data = true_output_data.astype(int)

  # Check if subarrays are equal
  for subarray1, subarray2 in tqdm(zip(true_output_data, predicted_data), desc = "Evaluating"):
      if np.array_equal(subarray1, subarray2):
          correct_sequences.append(np.argmax(subarray1 == 0))

  # Get the length of the predicted traces
  predicted_trace_length = np.argmax(predicted_data == 0, axis=1)
  ture_trace_length = np.argmax(true_output_data == 0, axis=1)

  # Create bins
  bins = np.linspace(ture_trace_length.min(), ture_trace_length.max(), 25)

  # Plot histograms
  plt.hist(predicted_trace_length, bins=bins, alpha=0.5, label='All Predictions')
  plt.hist(ture_trace_length, bins=bins, alpha=0.5, label='True trace length')
  plt.hist(correct_sequences, bins=bins, alpha=0.5, label='Correct Sequences')


  # Add labels and title
  plt.xlabel('Sequence Length')
  plt.ylabel('Frequency')
  plt.title('Distribution of Correct Sequence Lengths over all Predicted Traces')

  # Add legend
  plt.legend()

  # Show the plot
  plt.show()

  # ------------------------------------------------------------------------------

def plot_levenshtein_distance(distance, bins = 100):
  # Create a histogram
  plt.hist(distance, bins=bins, color='blue', alpha=0.7)

  # Add labels and a title
  plt.xlabel('Distance')
  plt.ylabel('Frequency')
  plt.title('Levenshtein Distance')

  # Show the histogram
  plt.show()

  # ------------------------------------------------------------------------------