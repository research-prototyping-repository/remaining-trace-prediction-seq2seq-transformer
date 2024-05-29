# Import libraries
import os
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from src.general.functions_time import tic, toc,get_timestamp
from src.general.functions_report import report_prediction
from src.data.functions_training_data import load_processed_data, create_dataset
from src.models.transformer_model import Transformer, CustomSchedule, masked_loss, masked_accuracy
from src.models.transformer_predictor import Predictor

# Functions
# ------------------------------------------------------------------------------
# Create model function
def create_transformer_model(num_layers, d_model, num_heads, dff, vocab_size, dropout_rate, optimizer):
    model = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        dropout_rate=dropout_rate)

    model.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy])

    return model
# ------------------------------------------------------------------------------
# Prediction Function
def get_prediction_transformer(predictor,input_data, mapping, MAX_TOKENS):
  """
  This function is for getting multiple predictions from a transformer model at once.

  Input:
  - input_data: tf.Tensor nested conatining the  input traces
  - MAX_TOKENS: int containing the amount of maximum tokens to predict
  - mapping: dict containing the str/num mapping of the tokens

  Output:
  - predictions: nested array containing the output sequences
  """

  predictions = np.empty((input_data.shape[0], MAX_TOKENS))

  for index, array in tqdm(enumerate(input_data), desc = 'Predicting', total = input_data.shape[0]):
    temp_array_tensor = tf.convert_to_tensor(np.array([array]))
    temp_output = predictor(temp_array_tensor, mapping, MAX_TOKENS)
    temp_output = np.pad(temp_output[0], (0, MAX_TOKENS - np.size(temp_output)) , mode='constant', constant_values=mapping['<pad>']) 
    predictions[index] = temp_output

  return predictions
# ------------------------------------------------------------------------------

# Check available GPUs
print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Specify the GPU device
gpu_id = 1  # Use GPU 0
print("GPU used: ", gpu_id)

# Set GPU device options
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_visible_devices(physical_devices[gpu_id], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)


"""### Quick Parameters"""

# Set the random seed
tf.random.set_seed(2961998)

# Set memory variable
os.environ['TF_ALLOW_GROWTH'] = 'true'

# Set path variables
path_data = 'data/processed/'
path_control = 'data/control/'
path_predictions = 'data/predictions/'
path_models = 'models/'
path_reports = 'reports/'

# List of all control files
# files = ['variables_helpdesk.pkl','variables_helpdesk_true.pkl','variables_helpdesk_false.pkl','variables_bpi2012.pkl','variables_bpi2012_true.pkl','variables_bpi2012_false.pkl','variables_bpi2017.pkl','variables_bpi2017_true.pkl','variables_bpi2017_false.pkl','variables_bpi2018.pkl','variables_bpi2018_true.pkl','variables_bpi2018_false.pkl','variables_bpi2019.pkl','variables_bpi2019_true.pkl','variables_bpi2019_false.pkl']

files = ['variables_bpi2019.pkl','variables_bpi2019_true.pkl','variables_bpi2019_false.pkl']


# Iterate over control files
for filename_variables in files:

    # Set control filename
    print("filename_variables: ", filename_variables)

    # Load variables
    with open(path_control + filename_variables, 'rb') as file:
        variables = pickle.load(file)

    # Set file names
    # filename_transformer_weights = 'transformer_weights_bpi2012_true.h5'
    variables['filename_predictions'] = 'predictions_'+  filename_variables[10:][:-4] +'.npz'

    # Get timestamp
    variables['transformer_timestamp_prediction_start'] = get_timestamp()

    # Load data and create dummy data for inializing model
    x_train_tensor, y_train_tensor, x_val_tensor, y_val_tensor, x_test_tensor, y_test_tensor = load_processed_data(path_data + variables['filename_processed_dataset'])
    x_dummy_tensor = np.zeros((1, x_train_tensor.shape[1])) # Create dummmy data to initilalize the transformer
    y_dummy_tensor = np.zeros((1, y_train_tensor.shape[1]))
    train_data = create_dataset(x_dummy_tensor, y_dummy_tensor, 1) # Create dummy dataset to initalize the transformer


    # Transform data
    x_test = x_test_tensor.numpy()
    y_test = y_test_tensor.numpy()
    
    print(y_test.shape)

    # Variables for Transformer
    optimizer = tf.keras.optimizers.legacy.Adam(variables['learning_rate'], beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    variables['MAX_TOKENS'] = y_test.shape[1] # Including the start token
    print("MAX_TOKENS: ", variables['MAX_TOKENS'])

    # Build transformer and the predictior
    transformer = create_transformer_model(variables['num_layers'], variables['d_model'], variables['num_heads'], variables['dff'], variables['vocab_size'], variables['dropout_rate'], optimizer)
    history = transformer.fit(train_data, epochs = 1, verbose = 1 # Initialzie model
    print("Transformer initialized with dummy variables.")
    transformer.load_weights(path_models + 'transformer/' + variables['transformer_model']) # Load weights
    print("Transformer loaded: ", variables['transformer_model'])
    predictor = Predictor(transformer)
    print("Predictor initalized.")

    # Prediction
    tic()
    y_pred = get_prediction_transformer(predictor, x_test, variables['mapping'], variables['MAX_TOKENS']) 
    variables['elapsed_time_prediction'] = toc()

    # Save predictions
    data_predictions = np.load(path_predictions + variables['filename_predictions'])
    data_predictions_copy = dict(data_predictions)
    print(data_predictions_copy['y_test'].shape)
    data_predictions_copy['y_pred_transformer'] = y_pred
    np.savez(path_predictions + variables['filename_predictions'], **data_predictions_copy)
    print("Predictions saved to ", path_data + 'predictions/' + variables['filename_predictions'])

    # Create report
    report_prediction(filename_variables, variables, variables['transformer_timestamp_prediction_start'], path_reports)

    # Save variables
    with open(path_control + filename_variables, 'wb') as file:
        pickle.dump(variables, file)
    print("Variables saved to ", path_control + filename_variables)
    
    del variables,y_pred,data_predictions,data_predictions_copy,x_train_tensor, y_train_tensor, x_val_tensor, y_val_tensor, x_test_tensor, y_test_tensor,x_dummy_tensor,y_dummy_tensor, train_data