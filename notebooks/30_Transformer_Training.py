# -*- coding: utf-8 -*-
"""20230609_Seq2Seq_Transfromer_Processing_console.ipynb

Source: https://www.tensorflow.org/text/tutorials/transformer
"""

# Import libraries
import os
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from src.general.functions_time import tic, toc, get_timestamp
from src.data.functions_training_data import load_processed_data, create_dataset
from src.models.transformer_model import Transformer, CustomSchedule, masked_loss, masked_accuracy
from src.visualization.functions_training_visualize import display_training_curves
from src.general.functions_report import report_training

# # Functions

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

files = ['variables_bpi2018.pkl','variables_bpi2018_true.pkl','variables_bpi2018_false.pkl']

# Iterate over all control files
for filename_variables in files:
    # Set control filename
    print("filename_variables: ", filename_variables)

    # Load variables
    with open(path_control + filename_variables, 'rb') as file:
        variables = pickle.load(file)

    # timestamp
    variables['transformer_timestamp_training_start'] = get_timestamp()

    # Set filename
    variables['transformer_model'] = 'transformer_weights_' +  filename_variables[10:][:-4] + '.h5'

    # Load data
    x_train_tensor, y_train_tensor, x_val_tensor, y_val_tensor, x_test_tensor, y_test_tensor = load_processed_data(path_data + variables['filename_processed_dataset'])


    """### Set params"""
    # Variables for Transformer
    variables['num_layers'] = 2
    variables['d_model'] = 32
    variables['num_heads'] = 8
    variables['dff'] = 128
    variables['dropout_rate'] = 0.1
    variables['num_epochs'] = 50
    variables['batch_size'] = 512

    variables['learning_rate'] = CustomSchedule(variables['d_model'], warmup_steps= 1000)

    optimizer = tf.keras.optimizers.legacy.Adam(variables['learning_rate'], beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    variables['monitor_early_stopping'] = 'val_loss'
    variables['min_delta'] = 0.001
    variables['patience_early_stopping'] = 5
    variables['monitor_checkpoint'] = 'val_loss'


    # Prepare input data as a tuple
    train_data = create_dataset(x_train_tensor, y_train_tensor, variables['batch_size'])
    val_data = create_dataset(x_val_tensor, y_val_tensor, variables['batch_size'])
    test_data = create_dataset(x_test_tensor, y_test_tensor, variables['batch_size'])

    """### Build model"""

    # Build transformer
    transformer = create_transformer_model(variables['num_layers'], variables['d_model'], variables['num_heads'], variables['dff'], variables['vocab_size'], variables['dropout_rate'], optimizer)

    # Early stopping and checkpoint
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=variables['monitor_early_stopping'], min_delta = variables ['min_delta'], patience=variables['patience_early_stopping'], verbose = 1, restore_best_weights=True)


    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath= path_models + 'transformer/' + variables['transformer_model'], 
                                 monitor=variables['monitor_checkpoint'], 
                                 save_best_only=True,
                                 save_weights_only=True)

    """### Train model"""

    # Train transformer
    tic()
    history = transformer.fit(train_data, epochs = variables['num_epochs'], validation_data=val_data, callbacks = [early_stopping,checkpoint], verbose = 1)
    transformer.save_weights(path_models + 'transformer/' +  variables['transformer_model'])
    print("Model has been saved at: ", path_models + 'transformer/' +  variables['transformer_model']) 
    variables['elapsed_time'] = toc()
    variables['num_epochs_run'] = len(history.history['loss'])

    transformer.summary()

    # Plot training curves
    filename_plot = path_reports+ "training/" + variables['transformer_timestamp_training_start'] +  "_training_curves_" + os.path.splitext(variables["transformer_model"])[0] +'.svg'
    plt.subplots(figsize=(7,6))
    plt.subplots_adjust(hspace=0.375)
    ax1 = display_training_curves(history.history['masked_accuracy'], history.history['val_masked_accuracy'], 'accuracy', 211)
    ax2 = display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 212)
    ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.suptitle("Training curves: " + variables["transformer_model"][:-3])
    plt.savefig(filename_plot)
    print("Training curves saved: ", filename_plot)
    plt.show()

    variables['masked_accuracy'] = history.history['masked_accuracy']
    variables['val_masked_accuracy'] = history.history['val_masked_accuracy']
    variables['masked_loss'] = history.history['loss']
    variables['val_masked_loss'] = history.history['val_loss']

    """## Evaluation"""

    # Evaluation
    variables['transformer_loss'], variables['transformer_acc'] = transformer.evaluate(test_data)
    print('loss: ',variables['transformer_loss'])
    print('acc: ',variables['transformer_acc'])

    # Generate report
    report_training(filename_variables, variables, variables['transformer_timestamp_training_start'], path_reports)

    # Save variables
    with open(path_control + filename_variables, 'wb') as file:
        pickle.dump(variables, file)
    print("Variables saved to ", path_control + filename_variables)


    del variables,train_data, val_data, test_data 
