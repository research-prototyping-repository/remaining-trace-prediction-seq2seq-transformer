# Import libraries
import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import custom functions
from src.general.functions_time import tic, toc
from src.data.functions_training_data import load_variables, load_processed_data, create_dataset
from src.models.transformer_model import Transformer, CustomSchedule, masked_loss, masked_accuracy


# Import hyperparameter-tuning
from keras_tuner.tuners import GridSearch
from keras_tuner import Objective

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

# Set variables
filename_variables = 'variables_bpi2012_true.pkl'

# Set path variables
path_data = 'data/processed/'
path_control = 'data/control/'
path_predictions = 'data/predictions/'
path_models = 'models/'
path_reports = 'reports/'

# Load variables
with open(path_control + filename_variables, 'rb') as file:
    variables = pickle.load(file)
    
# Load data
x_train_tensor, y_train_tensor, x_val_tensor, y_val_tensor, x_test_tensor, y_test_tensor = load_processed_data(path_data + variables['filename_processed_dataset'])


# Variables for Transformer
num_layers = 2 # 4 
d_model = 32 # 12
num_heads = 8 # 3
dff = 48   # 48 hidden dim
dropout_rate = 0.1 # 0.1
num_epochs = 50 #10
batch_size = 512 # 256 
vocab_size = variables['vocab_size']

# Prepare input data as a tuple
train_data = create_dataset(x_train_tensor, y_train_tensor, batch_size)
val_data = create_dataset(x_val_tensor, y_val_tensor, batch_size)
test_data = create_dataset(x_test_tensor, y_test_tensor, batch_size)

# Create model function for hyperparameter tuning
def create_model_hyperparameter(hp): 
    
  model = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=hp.Choice('dff', values=[32,64,128,256,512,1024]),
    input_vocab_size=vocab_size,
    target_vocab_size=vocab_size,
    dropout_rate= hp.Choice('dropout_rate', values=[0.1,0.3,0.5])
  )

  model.compile(
    loss=masked_loss,
    optimizer=tf.keras.optimizers.legacy.Adam(CustomSchedule(d_model, warmup_steps= 1000), beta_1=0.9, beta_2=0.98, epsilon=1e-9),
    metrics=[masked_accuracy]
  )

  return model


# Define the objective metric for tuner
objective = Objective(
    name='val_masked_accuracy',
    direction='max')

# Initalize tuner
tuner = GridSearch(
    create_model_hyperparameter,
    objective = objective,
    max_trials = 50,
    executions_per_trial = 1,
    directory = path_models + 'transformer',
    project_name = 'hyperparameter_tuning_transformer_complexity-backup2'
)

# Early stopping
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Tuner search
tuner.search_space_summary()
tuner.search(train_data, epochs = num_epochs, validation_data = val_data, callbacks=[stop_early])
tuner.results_summary(num_trials=1000)
print("Hyperparameter-Tuning abgeschlossen.")

