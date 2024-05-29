"""
Source for architecture
https://link.springer.com/chapter/10.1007/978-3-030-58666-9_14
"""
# Import libraries
import os
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping


from src.data.functions_training_data import load_processed_data
from src.visualization.functions_training_visualize import display_training_curves
from src.general.functions_time import tic, toc, get_timestamp
from src.general.functions_report import report_training_lstm

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
path_raw = 'data/raw/'
path_interim = 'data/interim/'
path_benchmark = 'data/benchmark/'
path_data = 'data/processed/'
path_control = 'data/control/'
path_predictions = 'data/predictions/'
path_models = 'models/'
path_reports = 'reports/'

# ------------------------------------------------------------------------------
# List of control files
files = ['variables_helpdesk.pkl','variables_helpdesk_true.pkl','variables_helpdesk_false.pkl','variables_bpi2012.pkl','variables_bpi2012_true.pkl','variables_bpi2012_false.pkl','variables_bpi2017.pkl','variables_bpi2017_true.pkl','variables_bpi2017_false.pkl','variables_bpi2018.pkl','variables_bpi2018_true.pkl','variables_bpi2018_false.pkl','variables_bpi2019.pkl','variables_bpi2019_true.pkl','variables_bpi2019_false.pkl']

# files = ['variables_helpdesk.pkl','variables_helpdesk_true.pkl','variables_helpdesk_false.pkl','variables_bpi2012.pkl','variables_bpi2012_true.pkl','variables_bpi2012_false.pkl','variables_bpi2017.pkl','variables_bpi2017_true.pkl','variables_bpi2017_false.pkl']


# Iterate over all control files
for filename_variables in files: 
    
    # Load variables
    print("Filename: ", filename_variables)
    with open(path_control + filename_variables, 'rb') as file:
        variables = pickle.load(file)

    # Write model name to control variables
    variables['lstm_model'] = 'lstm_'+ filename_variables[10:][:-4] +'.h5'

    # Get timestamp
    variables['lstm_timestamp_training_start'] = get_timestamp()

    # Load benchmark data
    x_train, y_train, x_val, y_val, x_test, y_test = load_processed_data(path_benchmark + variables['filename_benchmark_dataset'], tensor = False)

    # Model parameter
    variables['lstm_batch_size'] = 32
    variables['lstm_num_epochs'] = 50
    variables['lstm_learning_rate'] = 0.0001
    variables['lstm_monitor_early_stopping'] = 'val_loss'
    variables['lstm_min_delta'] = 0.001
    variables['lstm_patience_early_stopping'] = 5

    num_samples = len(x_train)  # Total number of training samples
    num_steps_per_epoch = num_samples // variables['lstm_batch_size']

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=variables['lstm_learning_rate'])

    early_stop = EarlyStopping(monitor=variables['lstm_monitor_early_stopping'], min_delta=variables['lstm_min_delta'],
                            patience=variables['lstm_patience_early_stopping'], verbose=1, mode='auto',
                            baseline=0, restore_best_weights=True)

    lstm = tf.keras.Sequential()
    lstm.add(LSTM(x_train.shape[1]*2, input_shape=(x_train.shape[1],1), return_sequences=True))#True = many to many
    lstm.add(LSTM(x_train.shape[1]*2, return_sequences=False))
    lstm.add(Dense(variables['vocab_size'],kernel_initializer='normal',activation='softmax'))
    # Compilation of the model
    lstm.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer =optimizer, metrics=['accuracy'])
    print(lstm.summary())

    tic()
    history = lstm.fit(x_train,
                    y_train.flatten(),
                    epochs=variables['lstm_num_epochs'],
                    batch_size=variables['lstm_batch_size'],
                    validation_data=(x_val, y_val.flatten()),
                    callbacks = early_stop,
                    verbose=1)
    variables['lstm_elapsed_time'] = toc()
    variables['lstm_epochs_run'] = len(history.history['loss'])

    variables['lstm_loss'], variables['lstm_acc'] = lstm.evaluate(x_test,
                        y_test[:,0].flatten(),
                        verbose=1,
                        batch_size=variables['lstm_batch_size'])

    print('loss: ',variables['lstm_loss'])
    print('acc: ',variables['lstm_acc'])

    # Plot training curves
    filename_plot = path_reports+ "training/competing_artifacts/lstm/" + variables['lstm_timestamp_training_start'] +  "_training_curves_" + variables['lstm_model'] +'.png'
    plt.subplots(figsize=(5,5))
    plt.tight_layout()
    display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 211)
    display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 212)
    plt.savefig(filename_plot)
    print("Training curves saved: ", filename_plot)
    plt.show()

    variables['lstm_masked_accuracy'] = history.history['accuracy']
    variables['lstm_val_masked_accuracy'] = history.history['val_accuracy']
    variables['lstm_masked_loss'] = history.history['loss']
    variables['lstm_val_masked_loss'] = history.history['val_loss']

    # Save the model
    lstm.save(path_models + 'lstm/' + variables['lstm_model'])
    print("Model saved: ", path_models + variables['lstm_model'])

    # Generate report
    report_training_lstm(filename_variables, variables, variables['lstm_timestamp_training_start'], path_reports)


    # Save variables
    with open(path_control + filename_variables, 'wb') as file:
        pickle.dump(variables, file)
    print("Variables saved to ", path_control + filename_variables)

    del variables, lstm, x_train, y_train, x_val, y_val, x_test, y_test
