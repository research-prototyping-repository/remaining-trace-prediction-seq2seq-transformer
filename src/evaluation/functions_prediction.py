# Libraries
# ------------------------------------------------------------------------------
import numpy as np
from tqdm import tqdm


# Functions
# ------------------------------------------------------------------------------
def get_multi_dim_prediction(model, x_input, mapping, output_probabilites = False, num_prediction_runs = None):
  """
  This function predicts a sequence based on the iterating single predictions and feeding them attached to the orignal sequence tail.
  The window is determined by the amount of  columns in x_input.
  The number of prediction runs can be set to a specific value, but usually the standard (number of columns of x_input) is fine.
  After the first apperance of the <end> token, all following predictions for this trace are set to zero.

  Input:
  - model: model that is used
  - x_input: nested array containg the data for which the predictions should be made
  - output_probabilites: boolean which indicates wether the model outputs the classes directly or the probabilites
  - num_prediction_runs [optional]: int which determines how many prediction runs are made

  Output:
  - y_pred: nested array containg the multi-dimensional predictions
  """

  # Determine the number of prediction runs
  if num_prediction_runs is None:
      num_prediction_runs = x_input.shape[1]

  # Create empty array to store predicitons in
  y_pred = np.empty((x_input.shape))

  # Copy the input data for modifying it temporally
  temp_x_input = np.copy(x_input)

  # Iterate over the amount of columns
  for i in tqdm(range(0, num_prediction_runs), desc = 'Predicting'):
    # Make one prediction
    temp_y_pred = model.predict(temp_x_input)

    # Get the predicted classes if model generates probabilites (indices of the class with highest probability)
    if output_probabilites == True:
      temp_y_pred = np.argmax(temp_y_pred, axis=1)

    # Safe the prediction in the prediction vector
    y_pred[:,i] = temp_y_pred

    # Append prediction to the temporary input data
    temp_y_pred = temp_y_pred[:, np.newaxis]
    temp_x_input = np.hstack((temp_x_input, temp_y_pred))

    # To guarantee the right input size, remove one column from input data
    temp_x_input = temp_x_input[:, 1:]

  # Replace all predictions after the <end> token witz zero
  # Find the indices of the first occurrence of 3 in each row
  indices = np.where(y_pred == mapping['<end>'])
  row_indices, col_indices = indices

  # Keep only the first occurrence of the end token for each row
  unique_rows, col_index_unique = np.unique(row_indices, return_index=True)
  unique_col = col_indices[col_index_unique]

  # Replace the rest of the array with 0 after the occurrence
  for i in tqdm(range(len(unique_rows)), desc="Cleaning predictions"):
      y_pred[unique_rows[i], unique_col[i]+1:] = 0

  y_pred = y_pred.astype(int)

  return y_pred