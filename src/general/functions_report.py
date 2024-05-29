# Functions
# ------------------------------------------------------------------------------
# Report preprocessing

def report_preprocessing(filename_variables, variables, timestamp, path_reports):
  """
  This function prints a summary of the preprocessing and saves the report to a file.

  Input:
  - filename_variables: string containing the filename
  - variables: dict containing the control variables
  - path_reports: string containing the path for reports

  Output:
  - file: txt-file containing the report
  """

  # Print summary
  print("Summary:")
  print("\n")
  print("Dataset:                ", variables['filename_dataset'])
  print("Filename processed data:", variables['filename_processed_dataset'])
  print("Filename variables:     ", filename_variables)
  print("\n")
  print("vocab (first 6):        ", variables['vocab'][:6])
  print("vocab_size:             ", variables['vocab_size'])
  print("max_length_trace:       ", variables['max_length_trace'])
  print("num_traces:             ", variables['num_traces'])
  print("num_ex_activities:      ", variables['num_ex_activities'])
  print("num_features:           ", variables['num_features'])
  print("features:               ", variables['features'])
  print("interleave:             ", variables['interleave'])
  print("\n")
  print("Samples in training:    ", variables['x_train_shape'])
  print("Samples in validation:  ", variables['x_val_shape'])
  print("Samples in test:        ", variables['x_test_shape'])

  # Write report to file
  # Create filename
  instance_name = variables['filename_processed_dataset'][18:-4]
  output_file_name = timestamp + '_report'+'_preprocessing_'+ instance_name +'.txt'
  output_file_path = path_reports + 'preprocessing/' + output_file_name

  # Open the file for writing (creates the file if it doesn't exist)
  with open(output_file_path, 'w') as file:
    # Write the summary information to the file
    file.write("Summary:\n\n")
    file.write("Dataset:                {}\n".format(variables['filename_dataset']))
    file.write("Filename processed data: {}\n".format(variables['filename_processed_dataset']))
    file.write("Filename variables:     {}\n".format(filename_variables))
    file.write("\n")
    file.write("vocab:                  {}\n".format(variables['vocab']))
    file.write("vocab_size:             {}\n".format(variables['vocab_size']))
    file.write("max_length_trace:       {}\n".format(variables['max_length_trace']))
    file.write("num_traces:             {}\n".format(variables['num_traces']))
    file.write("num_ex_activities:      {}\n".format(variables['num_ex_activities']))
    file.write("num_features:           {}\n".format(variables['num_features']))
    file.write("features:               {}\n".format(variables['features']))
    file.write("interleave:             {}\n".format(variables['interleave']))
    file.write("\n")
    file.write("Samples in training:    {}\n".format(variables['x_train_shape']))
    file.write("Samples in validation:  {}\n".format(variables['x_val_shape']))
    file.write("Samples in test:        {}\n".format(variables['x_test_shape']))

  # Confirm that the data has been written to the file
  print("\nReport has been written to '{}'".format(output_file_path))

# ------------------------------------------------------------------------------
# Report preprocessing benchmark

def report_preprocessing_benchmark(filename_variables, variables, timestamp, path_reports):
  """
  This function prints a summary of the preprocessing for the benchmark files and saves the report to a file.

  Input:
  - filename_variables: string containing the filename
  - variables: dict containing the control variables
  - path_reports: string containing the path for reports

  Output:
  - file: txt-file containing the report
  """

  # Print summary
  print("Summary:")
  print("\n")
  print("Dataset:                ", variables['filename_dataset'])
  print("Filename interim data:  ", variables['filename_interim_dataset'])
  print("Filename benchmark data:", variables['filename_benchmark_dataset'])
  print("Filename variables:     ", filename_variables)
  print("\n")
  print("Samples in training:    ", variables['x_train_shape_benchmark'])
  print("Samples in validation:  ", variables['x_val_shape_benchmark'])
  print("Samples in test:        ", variables['x_test_shape_benchmark'])

  # Write report to file
  # Create filename
  instance_name = variables['filename_benchmark_dataset'][18:-4]
  output_file_name = timestamp + '_report'+'_preprocessing_'+ instance_name +'.txt'
  output_file_path = path_reports + 'preprocessing/benchmark/' + output_file_name

  # Open the file for writing (creates the file if it doesn't exist)
  with open(output_file_path, 'w') as file:

    # Write the summary information to the file
    file.write("Summary:\n\n")
    file.write("Dataset:                {}\n".format(variables['filename_dataset']))
    file.write("Filename interim data:  {}\n".format(variables['filename_interim_dataset']))
    file.write("Filename benchmark data: {}\n".format(variables['filename_benchmark_dataset']))
    file.write("Filename variables:     {}\n\n".format(filename_variables))

    file.write("vocab:                  {}\n".format(variables['vocab']))
    file.write("vocab_size:             {}\n".format(variables['vocab_size']))
    file.write("max_length_trace:       {}\n".format(variables['max_length_trace']))
    file.write("num_traces:             {}\n".format(variables['num_traces']))
    file.write("num_ex_activities:      {}\n".format(variables['num_ex_activities']))
    file.write("num_features:           {}\n".format(variables['num_features']))
    file.write("features:               {}\n".format(variables['features']))
    file.write("interleave:             {}\n\n".format(variables['interleave']))


    file.write("Samples in training:    {}\n".format(variables['x_train_shape_benchmark']))
    file.write("Samples in validation:  {}\n".format(variables['x_val_shape_benchmark']))
    file.write("Samples in test:        {}\n".format(variables['x_test_shape_benchmark']))

  # Confirm that the data has been written to the file
  print("\nReport has been written to '{}'".format(output_file_path))

# ------------------------------------------------------------------------------
# Report Training Transformer
def report_training(filename_variables, variables, timestamp, path_reports):
  """
  This function prints a summary of the training process and saves the report to a file.

  Input:
  - filename_variables: string containing the filename
  - variables: dict containing the control variables
  - path_reports: string containing the path for reports

  Output:
  - file: txt-file containing the report
  """

  # Print summary
  print("\nSummary:")
  print("\n")
  print("Dataset:                ", variables['filename_dataset'])
  print("Filename interim data:  ", variables['filename_interim_dataset'])
  print("Filename processed data:", variables['filename_processed_dataset'])
  print("Filename variables:     ", filename_variables)
  print("\n")
  print("vocab (first 6):        ", variables['vocab'][:6])
  print("vocab_size:             ", variables['vocab_size'])
  print("max_length_trace:       ", variables['max_length_trace'])
  print("num_traces:             ", variables['num_traces'])
  print("num_ex_activities:      ", variables['num_ex_activities'])
  print("num_features:           ", variables['num_features'])
  print("features:               ", variables['features'])
  print("interleave:             ", variables['interleave'])
  print("\n")
  print("Samples in training:    ", variables['x_train_shape'])
  print("Samples in validation:  ", variables['x_val_shape'])
  print("Samples in test:        ", variables['x_test_shape'])
  print("\n")
  print("Training Transformer:")
  print("Elapsed time:           ", variables['elapsed_time'])
  print("Number of epochs run:   ", variables['num_epochs_run'])
  print("Transformer model:      ", variables['transformer_model'])
  print("\n")
  print("Parameters Transformer:")
  print("num_layers:             ", variables['num_layers'])
  print("d_model:                ", variables['d_model'])
  print("num_heads:              ", variables['num_heads'])
  print("dff:                    ", variables['dff'])
  print("dropout_rate:           ", variables['dropout_rate'])
  print("num_epochs:             ", variables['num_epochs'])
  print("batch_size:             ", variables['batch_size'])
  print("learning_rate:          ", variables['learning_rate'])
  print("\n")
  print("Training-Evaluation:")
  print("loss_transformer:       ", variables['transformer_loss'])
  print("acc_transformer:        ", variables['transformer_acc'])
  print("masked_accuracy:        ", variables['masked_accuracy'][-1])
  print("val_masked_accuracy:    ", variables['val_masked_accuracy'][-1])
  print("masked_loss:            ", variables['masked_loss'][-1])
  print("val_masked_loss:        ", variables['val_masked_loss'][-1])


  # Write report to file
  # Create filename
  instance_name = variables['filename_processed_dataset'][18:-4]
  output_file_name = timestamp + '_report'+'_training_'+ instance_name +'.txt'
  output_file_path = path_reports + 'training/' + output_file_name

  # Open the file for writing (creates the file if it doesn't exist)
  with open(output_file_path, 'w') as file:
    file.write("Summary:\n\n")
    file.write("Dataset:                {}\n".format(variables['filename_dataset']))
    file.write("Filename interim data:  {}\n".format(variables['filename_interim_dataset']))
    file.write("Filename processed data:{}\n".format(variables['filename_processed_dataset']))
    file.write("Filename variables:     {}\n".format(filename_variables))
    file.write("\n")
    file.write("vocab (first 6):        {}\n".format(variables['vocab'][:6]))
    file.write("vocab_size:             {}\n".format(variables['vocab_size']))
    file.write("max_length_trace:       {}\n".format(variables['max_length_trace']))
    file.write("num_traces:             {}\n".format(variables['num_traces']))
    file.write("num_ex_activities:      {}\n".format(variables['num_ex_activities']))
    file.write("num_features:           {}\n".format(variables['num_features']))
    file.write("features:               {}\n".format(variables['features']))
    file.write("interleave:             {}\n".format(variables['interleave']))
    file.write("\n")
    file.write("Samples in training:    {}\n".format(variables['x_train_shape']))
    file.write("Samples in validation:  {}\n".format(variables['x_val_shape']))
    file.write("Samples in test:        {}\n".format(variables['x_test_shape']))
    file.write("\n")
    file.write("Training Transformer:\n")
    file.write("Elapsed time:           {}\n".format(variables['elapsed_time']))
    file.write("Number of epochs run:   {}\n".format(variables['num_epochs_run']))
    file.write("Transformer model:      {}\n".format(variables['transformer_model']))
    file.write("\n")
    file.write("Parameters Transformer:\n")
    file.write("num_layers:             {}\n".format(variables['num_layers']))
    file.write("d_model:                {}\n".format(variables['d_model']))
    file.write("num_heads:              {}\n".format(variables['num_heads']))
    file.write("dff:                    {}\n".format(variables['dff']))
    file.write("dropout_rate:           {}\n".format(variables['dropout_rate']))
    file.write("num_epochs:             {}\n".format(variables['num_epochs']))
    file.write("batch_size:             {}\n".format(variables['batch_size']))
    file.write("learning_rate:          {}\n".format(variables['learning_rate']))
    file.write("\n")
    file.write("Training-Evaluation:\n")
    file.write("loss_transformer:       {}\n".format(variables['transformer_loss']))
    file.write("acc_transformer:        {}\n".format(variables['transformer_acc']))
    file.write("masked_accuracy:        {}\n".format(variables['masked_accuracy'][-1]))
    file.write("val_masked_accuracy:    {}\n".format(variables['val_masked_accuracy'][-1]))
    file.write("masked_loss:            {}\n".format(variables['masked_loss'][-1]))
    file.write("val_masked_loss:        {}\n".format(variables['val_masked_loss'][-1]))

  # Confirm that the data has been written to the file
  print("\nReport has been written to '{}'".format(output_file_path))

# ------------------------------------------------------------------------------
# Report Training LSTM
def report_training_lstm(filename_variables, variables, timestamp, path_reports):
  """
  This function prints a summary of the training process and saves the report to a file.

  Input:
  - filename_variables: string containing the filename
  - variables: dict containing the control variables
  - path_reports: string containing the path for reports

  Output:
  - file: txt-file containing the report
  """

  # Print summary
  print("\nSummary:")
  print("\n")
  print("Dataset:                ", variables['filename_dataset'])
  print("Filename interim data:  ", variables['filename_interim_dataset'])
  print("Filename processed data:", variables['filename_processed_dataset'])
  print("Filename variables:     ", filename_variables)
  print("\n")
  print("vocab (first 6):        ", variables['vocab'][:6])
  print("vocab_size:             ", variables['vocab_size'])
  print("max_length_trace:       ", variables['max_length_trace'])
  print("num_traces:             ", variables['num_traces'])
  print("num_ex_activities:      ", variables['num_ex_activities'])
  print("num_features:           ", variables['num_features'])
  print("features:               ", variables['features'])
  print("interleave:             ", variables['interleave'])
  print("\n")
  print("Samples in training:    ", variables['x_train_shape'])
  print("Samples in validation:  ", variables['x_val_shape'])
  print("Samples in test:        ", variables['x_test_shape'])
  print("\n")
  print("Training LSTM:")
  print("Elapsed time:           ", variables['lstm_elapsed_time'])
  print("Number of epochs run:   ", variables['lstm_epochs_run'])
  print("LSTM model:             ", variables['lstm_model'])
  print("\n")
  print("Parameters LSTM:")
  print("num_epochs:             ", variables['lstm_num_epochs'])
  print("batch_size:             ", variables['lstm_batch_size'])
  print("learning_rate:          ", variables['lstm_learning_rate'])
  print("\n")
  print("Training-Evaluation:")
  print("lstm_loss:              ", variables['lstm_loss'])
  print("lstm_acc:               ", variables['lstm_acc'])
  print("lstm_masked_accuracy:   ", variables['lstm_masked_accuracy'][-1])
  print("lstm_val_masked_accuracy:", variables['lstm_val_masked_accuracy'][-1])
  print("lstm_masked_loss:       ", variables['lstm_masked_loss'][-1])
  print("lstm_val_masked_loss:   ", variables['lstm_val_masked_loss'][-1])


  # Write report to file
  # Create filename
  instance_name = variables['filename_processed_dataset'][18:-4]
  output_file_name = timestamp + '_report'+'_training_lstm_'+ instance_name +'.txt'
  output_file_path = path_reports + 'training/competing_artifacts/lstm/' + output_file_name

  # Open the file for writing (creates the file if it doesn't exist)
  with open(output_file_path, 'w') as file:
    file.write("\nSummary:\n\n")
    file.write("Dataset:                {}\n".format(variables['filename_dataset']))
    file.write("Filename interim data:  {}\n".format(variables['filename_interim_dataset']))
    file.write("Filename processed data: {}\n".format(variables['filename_processed_dataset']))
    file.write("Filename variables:     {}\n".format(filename_variables))
    file.write("\n")
    file.write("vocab (first 6):        {}\n".format(variables['vocab'][:6]))
    file.write("vocab_size:             {}\n".format(variables['vocab_size']))
    file.write("max_length_trace:       {}\n".format(variables['max_length_trace']))
    file.write("num_traces:             {}\n".format(variables['num_traces']))
    file.write("num_ex_activities:      {}\n".format(variables['num_ex_activities']))
    file.write("num_features:           {}\n".format(variables['num_features']))
    file.write("features:               {}\n".format(variables['features']))
    file.write("interleave:             {}\n".format(variables['interleave']))
    file.write("\n")
    file.write("Samples in training:    {}\n".format(variables['x_train_shape']))
    file.write("Samples in validation:  {}\n".format(variables['x_val_shape']))
    file.write("Samples in test:        {}\n".format(variables['x_test_shape']))
    file.write("\n")
    file.write("Training LSTM:\n")
    file.write("Elapsed time:           {}\n".format(variables['lstm_elapsed_time']))
    file.write("Number of epochs run:   {}\n".format(variables['lstm_epochs_run']))
    file.write("LSTM model:             {}\n".format(variables['lstm_model']))
    file.write("\n")
    file.write("Parameters LSTM:\n")
    file.write("num_epochs:             {}\n".format(variables['lstm_num_epochs']))
    file.write("batch_size:             {}\n".format(variables['lstm_batch_size']))
    file.write("learning_rate:          {}\n".format(variables['lstm_learning_rate']))
    file.write("\n")
    file.write("Training-Evaluation:\n")
    file.write("lstm_loss:              {}\n".format(variables['lstm_loss']))
    file.write("lstm_acc:               {}\n".format(variables['lstm_acc']))
    file.write("lstm_masked_accuracy:   {}\n".format(variables['lstm_masked_accuracy'][-1]))
    file.write("lstm_val_masked_accuracy: {}\n".format(variables['lstm_val_masked_accuracy'][-1]))
    file.write("lstm_masked_loss:       {}\n".format(variables['lstm_masked_loss'][-1]))
    file.write("lstm_val_masked_loss:   {}\n".format(variables['lstm_val_masked_loss'][-1]))

  # Confirm that the data has been written to the file
  print("\nReport has been written to '{}'".format(output_file_path))


# ------------------------------------------------------------------------------
# Report Training Random Forest
def report_training_random_forest(filename_variables, variables, timestamp, path_reports):
  """
  This function prints a summary of the training process and saves the report to a file.

  Input:
  - filename_variables: string containing the filename
  - variables: dict containing the control variables
  - path_reports: string containing the path for reports

  Output:
  - file: txt-file containing the report
  """

  # Print summary
  print("\nSummary:")
  print("\n")
  print("Dataset:                ", variables['filename_dataset'])
  print("Filename interim data:  ", variables['filename_interim_dataset'])
  print("Filename variables:     ", filename_variables)
  print("\n")
  print("vocab (first 6):        ", variables['vocab'][:6])
  print("vocab_size:             ", variables['vocab_size'])
  print("max_length_trace:       ", variables['max_length_trace'])
  print("num_traces:             ", variables['num_traces'])
  print("num_ex_activities:      ", variables['num_ex_activities'])
  print("num_features:           ", variables['num_features'])
  print("features:               ", variables['features'])
  print("interleave:             ", variables['interleave'])
  print("\n")
  print("Samples in training:    ", variables['x_train_shape'])
  print("Samples in validation:  ", variables['x_val_shape'])
  print("Samples in test:        ", variables['x_test_shape'])
  print("\n")
  print("Training Random Forest:")
  print("Elapsed time:           ", variables['rf_elapsed_time'])
  print("Random forest model:    ", variables['random_forest_model'])
  print("\n")
  print("Parameters Random Forest:")
  print("rf_n_estimators:        ", variables['rf_n_estimators'])
  print("rf_n_jobs:              ", variables['rf_n_jobs'])
  print("\n")
  print("Training-Evaluation:")
  print("rf_acc          :       ", variables['rf_acc'])


  # Write report to file
  # Create filename
  instance_name = variables['filename_processed_dataset'][18:-4]
  output_file_name = timestamp + '_report'+'_training_rf_'+ instance_name +'.txt'
  output_file_path = path_reports + 'training/competing_artifacts/random_forest/' + output_file_name

  # Open the file for writing (creates the file if it doesn't exist)
  with open(output_file_path, 'w') as file:
    file.write("\nSummary:\n\n")
    file.write("Dataset:                {}\n".format(variables['filename_dataset']))
    file.write("Filename interim data:  {}\n".format(variables['filename_interim_dataset']))
    file.write("Filename variables:     {}\n".format(filename_variables))
    file.write("\n")
    file.write("vocab (first 6):        {}\n".format(variables['vocab'][:6]))
    file.write("vocab_size:             {}\n".format(variables['vocab_size']))
    file.write("max_length_trace:       {}\n".format(variables['max_length_trace']))
    file.write("num_traces:             {}\n".format(variables['num_traces']))
    file.write("num_ex_activities:      {}\n".format(variables['num_ex_activities']))
    file.write("num_features:           {}\n".format(variables['num_features']))
    file.write("features:               {}\n".format(variables['features']))
    file.write("interleave:             {}\n".format(variables['interleave']))
    file.write("\n")
    file.write("Samples in training:    {}\n".format(variables['x_train_shape']))
    file.write("Samples in validation:  {}\n".format(variables['x_val_shape']))
    file.write("Samples in test:        {}\n".format(variables['x_test_shape']))
    file.write("\n")
    file.write("Training Random Forest:\n")
    file.write("Elapsed time:           {}\n".format(variables['rf_elapsed_time']))
    file.write("Random forest model:    {}\n".format(variables['random_forest_model']))
    file.write("\n")
    file.write("Parameters Random Forest:\n")
    file.write("rf_n_estimators:        {}\n".format(variables['rf_n_estimators']))
    file.write("rf_n_jobs:              {}\n".format(variables['rf_n_jobs']))
    file.write("\n")
    file.write("Training-Evaluation:\n")
    file.write("rf_acc          :       {}\n".format(variables['rf_acc']))

  # Confirm that the data has been written to the file
  print("\nReport has been written to '{}'".format(output_file_path))

# ------------------------------------------------------------------------------
# Report Training Logistical Regression
def report_training_logistical_regression(filename_variables, variables, timestamp, path_reports):
  """
  This function prints a summary of the training process and saves the report to a file.

  Input:
  - filename_variables: string containing the filename
  - variables: dict containing the control variables
  - path_reports: string containing the path for reports

  Output:
  - file: txt-file containing the report
  """

  # Print summary
  print("\nSummary:")
  print("\n")
  print("Dataset:                ", variables['filename_dataset'])
  print("Filename interim data:  ", variables['filename_interim_dataset'])
  print("Filename variables:     ", filename_variables)
  print("\n")
  print("vocab (first 6):        ", variables['vocab'][:6])
  print("vocab_size:             ", variables['vocab_size'])
  print("max_length_trace:       ", variables['max_length_trace'])
  print("num_traces:             ", variables['num_traces'])
  print("num_ex_activities:      ", variables['num_ex_activities'])
  print("num_features:           ", variables['num_features'])
  print("features:               ", variables['features'])
  print("interleave:             ", variables['interleave'])
  print("\n")
  print("Samples in training:    ", variables['x_train_shape'])
  print("Samples in validation:  ", variables['x_val_shape'])
  print("Samples in test:        ", variables['x_test_shape'])
  print("\n")
  print("Training Logistical Regression:")
  print("Elapsed time:           ", variables['logreg_elapsed_time'])
  print("Regression model:       ", variables['regression_model'])
  print("\n")
  print("Parameters Logistical Regression:")
  print("logreg_max_iter:        ", variables['logreg_max_iter'])
  print("logreg_n_jobs:          ", variables['logreg_n_jobs'])
  print("\n")
  print("Training-Evaluation:")
  print("logreg_acc:             ", variables['logreg_acc'])


  # Write report to file
  # Create filename
  instance_name = variables['filename_processed_dataset'][18:-4]
  output_file_name = timestamp + '_report'+'_training_logreg_'+ instance_name +'.txt'
  output_file_path = path_reports + 'training/competing_artifacts/logistical_regression/' + output_file_name

  # Open the file for writing (creates the file if it doesn't exist)
  with open(output_file_path, 'w') as file:
    file.write("\nSummary:\n\n")
    file.write("Dataset:                {}\n".format(variables['filename_dataset']))
    file.write("Filename interim data:  {}\n".format(variables['filename_interim_dataset']))
    file.write("Filename variables:     {}\n".format(filename_variables))
    file.write("\n")
    file.write("vocab (first 6):        {}\n".format(variables['vocab'][:6]))
    file.write("vocab_size:             {}\n".format(variables['vocab_size']))
    file.write("max_length_trace:       {}\n".format(variables['max_length_trace']))
    file.write("num_traces:             {}\n".format(variables['num_traces']))
    file.write("num_ex_activities:      {}\n".format(variables['num_ex_activities']))
    file.write("num_features:           {}\n".format(variables['num_features']))
    file.write("features:               {}\n".format(variables['features']))
    file.write("interleave:             {}\n".format(variables['interleave']))
    file.write("\n")
    file.write("Samples in training:    {}\n".format(variables['x_train_shape']))
    file.write("Samples in validation:  {}\n".format(variables['x_val_shape']))
    file.write("Samples in test:        {}\n".format(variables['x_test_shape']))
    file.write("\n")
    file.write("Training Logistical Regression:\n")
    file.write("Elapsed time:           {}\n".format(variables['logreg_elapsed_time']))
    file.write("Regression model:       {}\n".format(variables['regression_model']))
    file.write("\n")
    file.write("Parameters Logistical Regression:\n")
    file.write("logreg_max_iter:        {}\n".format(variables['logreg_max_iter']))
    file.write("logreg_n_jobs:          {}\n".format(variables['logreg_n_jobs']))
    file.write("\n")
    file.write("Training-Evaluation:\n")
    file.write("logreg_acc:             {}\n".format(variables['logreg_acc']))

  # Confirm that the data has been written to the file
  print("\nReport has been written to '{}'".format(output_file_path))


# ------------------------------------------------------------------------------
# Report Prediction Transformer
def report_prediction(filename_variables, variables, timestamp, path_reports):
  """
  This function prints a summary of the prediction process and saves the report to a file.

  Input:
  - filename_variables: string containing the filename
  - variables: dict containing the control variables
  - path_reports: string containing the path for reports

  Output:
  - file: txt-file containing the report
  """

  # Print summary
  print("\nSummary:")
  print("\n")
  print("Dataset:                ", variables['filename_dataset'])
  print("Filename interim data:  ", variables['filename_interim_dataset'])
  print("Filename processed data:", variables['filename_processed_dataset'])
  print("Filename variables:     ", filename_variables)
  print("\n")
  print("vocab (first 6):        ", variables['vocab'][:6])
  print("vocab_size:             ", variables['vocab_size'])
  print("max_length_trace:       ", variables['max_length_trace'])
  print("num_traces:             ", variables['num_traces'])
  print("num_ex_activities:      ", variables['num_ex_activities'])
  print("num_features:           ", variables['num_features'])
  print("features:               ", variables['features'])
  print("interleave:             ", variables['interleave'])
  print("\n")
  print("Samples in training:    ", variables['x_train_shape'])
  print("Samples in validation:  ", variables['x_val_shape'])
  print("Samples in test:        ", variables['x_test_shape'])
  print("\n")
  print("Prediction Transformer:")
  print("Elapsed time:           ", variables['elapsed_time_prediction'])
  print("Transformer model:      ", variables['transformer_model'])
  print("\n")
  print("Parameters Transformer:")
  print("num_layers:             ", variables['num_layers'])
  print("d_model:                ", variables['d_model'])
  print("num_heads:              ", variables['num_heads'])
  print("dff:                    ", variables['dff'])
  print("dropout_rate:           ", variables['dropout_rate'])
  print("num_epochs:             ", variables['num_epochs'])
  print("batch_size:             ", variables['batch_size'])
  print("learning_rate:          ", variables['learning_rate'])


  # Write report to file
  # Create filename
  instance_name = variables['filename_processed_dataset'][18:-4]
  output_file_name = timestamp + '_report'+'_prediction_'+ instance_name + '.txt'
  output_file_path = path_reports + 'prediction/' + output_file_name

  # Open the file for writing (creates the file if it doesn't exist)
  with open(output_file_path, 'w') as file:
    file.write("Summary:\n\n")
    file.write("Dataset:                {}\n".format(variables['filename_dataset']))
    file.write("Filename interim data:  {}\n".format(variables['filename_interim_dataset']))
    file.write("Filename processed data:{}\n".format(variables['filename_processed_dataset']))
    file.write("Filename variables:     {}\n".format(filename_variables))
    file.write("\n")
    file.write("vocab (first 6):        {}\n".format(variables['vocab'][:6]))
    file.write("vocab_size:             {}\n".format(variables['vocab_size']))
    file.write("max_length_trace:       {}\n".format(variables['max_length_trace']))
    file.write("num_traces:             {}\n".format(variables['num_traces']))
    file.write("num_ex_activities:      {}\n".format(variables['num_ex_activities']))
    file.write("num_features:           {}\n".format(variables['num_features']))
    file.write("features:               {}\n".format(variables['features']))
    file.write("interleave:             {}\n".format(variables['interleave']))
    file.write("\n")
    file.write("Samples in training:    {}\n".format(variables['x_train_shape']))
    file.write("Samples in validation:  {}\n".format(variables['x_val_shape']))
    file.write("Samples in test:        {}\n".format(variables['x_test_shape']))
    file.write("\n")
    file.write("Prediction Transformer:\n")
    file.write("Elapsed time:           {}\n".format(variables['elapsed_time_prediction']))
    file.write("Transformer model:      {}\n".format(variables['transformer_model']))
    file.write("\n")
    file.write("Parameters Transformer:\n")
    file.write("num_layers:             {}\n".format(variables['num_layers']))
    file.write("d_model:                {}\n".format(variables['d_model']))
    file.write("num_heads:              {}\n".format(variables['num_heads']))
    file.write("dff:                    {}\n".format(variables['dff']))
    file.write("dropout_rate:           {}\n".format(variables['dropout_rate']))
    file.write("num_epochs:             {}\n".format(variables['num_epochs']))
    file.write("batch_size:             {}\n".format(variables['batch_size']))
    file.write("learning_rate:          {}\n".format(variables['learning_rate']))

  # Confirm that the data has been written to the file
  print("\nReport has been written to '{}'".format(output_file_path))

# ------------------------------------------------------------------------------
# Report Training LSTM
def report_prediction_lstm(filename_variables, variables, timestamp, path_reports):
  """
  This function prints a summary of the prediction process and saves the report to a file.

  Input:
  - filename_variables: string containing the filename
  - variables: dict containing the control variables
  - path_reports: string containing the path for reports

  Output:
  - file: txt-file containing the report
  """

  # Print summary
  print("\nSummary:")
  print("\n")
  print("Dataset:                ", variables['filename_dataset'])
  print("Filename interim data:  ", variables['filename_interim_dataset'])
  print("Filename processed data:", variables['filename_processed_dataset'])
  print("Filename variables:     ", filename_variables)
  print("\n")
  print("vocab (first 6):        ", variables['vocab'][:6])
  print("vocab_size:             ", variables['vocab_size'])
  print("max_length_trace:       ", variables['max_length_trace'])
  print("num_traces:             ", variables['num_traces'])
  print("num_ex_activities:      ", variables['num_ex_activities'])
  print("num_features:           ", variables['num_features'])
  print("features:               ", variables['features'])
  print("interleave:             ", variables['interleave'])
  print("\n")
  print("Samples in training:    ", variables['x_train_shape'])
  print("Samples in validation:  ", variables['x_val_shape'])
  print("Samples in test:        ", variables['x_test_shape'])
  print("\n")
  print("Prediction LSTM:")
  print("Elapsed time:           ", variables['lstm_elapsed_time_predictions'])
  print("LSTM model:             ", variables['lstm_model'])
  print("\n")
  print("Parameters LSTM:")
  print("num_epochs:             ", variables['lstm_num_epochs'])
  print("batch_size:             ", variables['lstm_batch_size'])
  print("learning_rate:          ", variables['lstm_learning_rate'])



  # Write report to file
  # Create filename
  instance_name = variables['filename_processed_dataset'][18:-4]
  output_file_name = timestamp + '_report'+'_prediction_lstm_'+ instance_name +'.txt'
  output_file_path = path_reports + 'prediction/competing_artifacts/lstm/' + output_file_name

  # Open the file for writing (creates the file if it doesn't exist)
  with open(output_file_path, 'w') as file:
    file.write("\nSummary:\n\n")
    file.write("Dataset:                {}\n".format(variables['filename_dataset']))
    file.write("Filename interim data:  {}\n".format(variables['filename_interim_dataset']))
    file.write("Filename processed data: {}\n".format(variables['filename_processed_dataset']))
    file.write("Filename variables:     {}\n".format(filename_variables))
    file.write("\n")
    file.write("vocab (first 6):        {}\n".format(variables['vocab'][:6]))
    file.write("vocab_size:             {}\n".format(variables['vocab_size']))
    file.write("max_length_trace:       {}\n".format(variables['max_length_trace']))
    file.write("num_traces:             {}\n".format(variables['num_traces']))
    file.write("num_ex_activities:      {}\n".format(variables['num_ex_activities']))
    file.write("num_features:           {}\n".format(variables['num_features']))
    file.write("features:               {}\n".format(variables['features']))
    file.write("interleave:             {}\n".format(variables['interleave']))
    file.write("\n")
    file.write("Samples in training:    {}\n".format(variables['x_train_shape']))
    file.write("Samples in validation:  {}\n".format(variables['x_val_shape']))
    file.write("Samples in test:        {}\n".format(variables['x_test_shape']))
    file.write("\n")
    file.write("Prediction LSTM:\n")
    file.write("Elapsed time:           {}\n".format(variables['lstm_elapsed_time_predictions']))
    file.write("LSTM model:             {}\n".format(variables['lstm_model']))
    file.write("\n")
    file.write("Parameters LSTM:\n")
    file.write("num_epochs:             {}\n".format(variables['lstm_num_epochs']))
    file.write("batch_size:             {}\n".format(variables['lstm_batch_size']))
    file.write("learning_rate:          {}\n".format(variables['lstm_learning_rate']))

  # Confirm that the data has been written to the file
  print("\nReport has been written to '{}'".format(output_file_path))

# ------------------------------------------------------------------------------
# Report Prediction Random Forest
def report_prediction_random_forest(filename_variables, variables, timestamp, path_reports):
  """
  This function prints a summary of the prediction process and saves the report to a file.

  Input:
  - filename_variables: string containing the filename
  - variables: dict containing the control variables
  - path_reports: string containing the path for reports

  Output:
  - file: txt-file containing the report
  """

  # Print summary
  print("\nSummary:")
  print("\n")
  print("Dataset:                ", variables['filename_dataset'])
  print("Filename interim data:  ", variables['filename_interim_dataset'])
  print("Filename processed data:", variables['filename_processed_dataset'])
  print("Filename variables:     ", filename_variables)
  print("\n")
  print("vocab (first 6):        ", variables['vocab'][:6])
  print("vocab_size:             ", variables['vocab_size'])
  print("max_length_trace:       ", variables['max_length_trace'])
  print("num_traces:             ", variables['num_traces'])
  print("num_ex_activities:      ", variables['num_ex_activities'])
  print("num_features:           ", variables['num_features'])
  print("features:               ", variables['features'])
  print("interleave:             ", variables['interleave'])
  print("\n")
  print("Samples in training:    ", variables['x_train_shape'])
  print("Samples in validation:  ", variables['x_val_shape'])
  print("Samples in test:        ", variables['x_test_shape'])
  print("\n")
  print("Prediction Random Forest:")
  print("Elapsed time:           ", variables['rf_elapsed_time_predictions'])
  print("Random Forest model:    ", variables['random_forest_model'])
  print("\n")
  print("Parameters Random Forest:")
  print("rf_n_estimators:        ", variables['rf_n_estimators'])
  print("rf_n_jobs:              ", variables['rf_n_jobs'])


  # Write report to file
  # Create filename
  instance_name = variables['filename_processed_dataset'][18:-4]
  output_file_name = timestamp + '_report'+'_prediction_rf_'+ instance_name + '.txt'
  output_file_path = path_reports + 'prediction/competing_artifacts/random_forest/' + output_file_name

  # Open the file for writing (creates the file if it doesn't exist)
  with open(output_file_path, 'w') as file:
    file.write("\nSummary:\n\n")
    file.write("Dataset:                {}\n".format(variables['filename_dataset']))
    file.write("Filename interim data:  {}\n".format(variables['filename_interim_dataset']))
    file.write("Filename processed data: {}\n".format(variables['filename_processed_dataset']))
    file.write("Filename variables:     {}\n".format(filename_variables))
    file.write("\n")
    file.write("vocab (first 6):        {}\n".format(variables['vocab'][:6]))
    file.write("vocab_size:             {}\n".format(variables['vocab_size']))
    file.write("max_length_trace:       {}\n".format(variables['max_length_trace']))
    file.write("num_traces:             {}\n".format(variables['num_traces']))
    file.write("num_ex_activities:      {}\n".format(variables['num_ex_activities']))
    file.write("num_features:           {}\n".format(variables['num_features']))
    file.write("features:               {}\n".format(variables['features']))
    file.write("interleave:             {}\n".format(variables['interleave']))
    file.write("\n")
    file.write("Samples in training:    {}\n".format(variables['x_train_shape']))
    file.write("Samples in validation:  {}\n".format(variables['x_val_shape']))
    file.write("Samples in test:        {}\n".format(variables['x_test_shape']))
    file.write("\n")
    file.write("Prediction Random Forest:\n")
    file.write("Elapsed time:           {}\n".format(variables['rf_elapsed_time_predictions']))
    file.write("Random Forest model:    {}\n".format(variables['random_forest_model']))
    file.write("\n")
    file.write("Parameters Random Forest:\n")
    file.write("rf_n_estimators:        {}\n".format(variables['rf_n_estimators']))
    file.write("rf_n_jobs:              {}\n".format(variables['rf_n_jobs']))

  # Confirm that the data has been written to the file
  print("\nReport has been written to '{}'".format(output_file_path))

  # ------------------------------------------------------------------------------
# Report Prediction Logistical Regression
def report_prediction_logistical_regression(filename_variables, variables, timestamp, path_reports):
  """
  This function prints a summary of the prediction process and saves the report to a file.

  Input:
  - filename_variables: string containing the filename
  - variables: dict containing the control variables
  - path_reports: string containing the path for reports

  Output:
  - file: txt-file containing the report
  """

  # Print summary
  print("\nSummary:")
  print("\n")
  print("Dataset:                ", variables['filename_dataset'])
  print("Filename interim data:  ", variables['filename_interim_dataset'])
  print("Filename processed data:", variables['filename_processed_dataset'])
  print("Filename variables:     ", filename_variables)
  print("\n")
  print("vocab (first 6):        ", variables['vocab'][:6])
  print("vocab_size:             ", variables['vocab_size'])
  print("max_length_trace:       ", variables['max_length_trace'])
  print("num_traces:             ", variables['num_traces'])
  print("num_ex_activities:      ", variables['num_ex_activities'])
  print("num_features:           ", variables['num_features'])
  print("features:               ", variables['features'])
  print("interleave:             ", variables['interleave'])
  print("\n")
  print("Samples in training:    ", variables['x_train_shape'])
  print("Samples in validation:  ", variables['x_val_shape'])
  print("Samples in test:        ", variables['x_test_shape'])
  print("\n")
  print("Prediction Logistical Regression:\n")
  print("Elapsed time:           ", variables['elapsed_time_predictions_logreg'])
  print("Regression model:       ", variables['regression_model'])
  print("\n")
  print("Parameters Logistical Regression:\n")
  print("logreg_max_iter:        ", variables['logreg_max_iter'])
  print("logreg_n_jobs:          ", variables['logreg_n_jobs'])


  # Write report to file
  # Create filename
  instance_name = variables['filename_processed_dataset'][18:-4]
  output_file_name = timestamp + '_report'+'_prediction_logreg_'+ instance_name + '.txt'
  output_file_path = path_reports + 'prediction/competing_artifacts/logistical_regression/' + output_file_name

  # Open the file for writing (creates the file if it doesn't exist)
  with open(output_file_path, 'w') as file:
    file.write("\nSummary:\n\n")
    file.write("Dataset:                {}\n".format(variables['filename_dataset']))
    file.write("Filename interim data:  {}\n".format(variables['filename_interim_dataset']))
    file.write("Filename processed data: {}\n".format(variables['filename_processed_dataset']))
    file.write("Filename variables:     {}\n".format(filename_variables))
    file.write("\n")
    file.write("vocab (first 6):        {}\n".format(variables['vocab'][:6]))
    file.write("vocab_size:             {}\n".format(variables['vocab_size']))
    file.write("max_length_trace:       {}\n".format(variables['max_length_trace']))
    file.write("num_traces:             {}\n".format(variables['num_traces']))
    file.write("num_ex_activities:      {}\n".format(variables['num_ex_activities']))
    file.write("num_features:           {}\n".format(variables['num_features']))
    file.write("features:               {}\n".format(variables['features']))
    file.write("interleave:             {}\n".format(variables['interleave']))
    file.write("\n")
    file.write("Samples in training:    {}\n".format(variables['x_train_shape']))
    file.write("Samples in validation:  {}\n".format(variables['x_val_shape']))
    file.write("Samples in test:        {}\n".format(variables['x_test_shape']))
    file.write("\n")
    file.write("Prediction Logistical Regression:\n")
    file.write("Elapsed time:           {}\n".format(variables['elapsed_time_predictions_logreg']))
    file.write("Regression model:       {}\n".format(variables['regression_model']))
    file.write("\n")
    file.write("Parameters Logistical Regression:\n")
    file.write("logreg_max_iter:        {}\n".format(variables['logreg_max_iter']))
    file.write("logreg_n_jobs:          {}\n".format(variables['logreg_n_jobs']))

  # Confirm that the data has been written to the file
  print("\nReport has been written to '{}'".format(output_file_path))

  # ------------------------------------------------------------------------------
# Report Evaluation all models
def report_evaluation(filename_variables, variables, timestamp, path_reports):
  """
  This function prints a summary of the evaluation process and saves the report to a file.

  Input:
  - filename_variables: string containing the filename
  - variables: dict containing the control variables
  - path_reports: string containing the path for reports

  Output:
  - file: txt-file containing the report
  """

  # Print summary
  print("\nSummary:")
  print("\n")
  print("Dataset:                ", variables['filename_dataset'])
  print("Filename interim data:  ", variables['filename_interim_dataset'])
  print("Filename processed data:", variables['filename_processed_dataset'])
  print("Filename variables:     ", filename_variables)
  print("\n")
  print("vocab (first 6):        ", variables['vocab'][:6])
  print("vocab_size:             ", variables['vocab_size'])
  print("max_length_trace:       ", variables['max_length_trace'])
  print("num_traces:             ", variables['num_traces'])
  print("num_ex_activities:      ", variables['num_ex_activities'])
  print("num_features:           ", variables['num_features'])
  print("features:               ", variables['features'])
  print("interleave:             ", variables['interleave'])
  print("\n")
  print("Dataset")
  print("Samples in training:    ", variables['x_train_shape'])
  print("Samples in validation:  ", variables['x_val_shape'])
  print("Samples in test:        ", variables['x_test_shape'])
  print("\n")
  print("Benchmark Dataset:")
  print("Samples in training:    ", variables['x_train_shape_benchmark'])
  print("Samples in validation:  ", variables['x_val_shape_benchmark'])
  print("Samples in test:        ", variables['x_test_shape_benchmark'])
  print("\n")
  print("Dataset:")
  print("\n")
  print("Prediction length:")
  print("\n")
  print("Prediction length 25%: ", variables['prediction_length_25'])
  print("Prediction length 50%: ", variables['prediction_length_50'])
  print("Prediction length 75%: ", variables['prediction_length_75'])
  print("\n")
  print("Evaluation Reference data:")
  print("Average pred. trace len:", variables['trace_length_reference'].mean()) 
  print("\n")
  print("Evaluation Reference benchmark data:")
  print("Average pred. trace len:", variables['trace_length_reference_benchmark'].mean()) 
  print("\n")
  print("Evaluation Transformer:")
  print("Average pred. trace len:", variables['predicted_trace_length_transformer'].mean())
  print("DLS mean 100%:          ", variables['similarity_transformer_100'].mean())
  print("DLS min 100%:           ", variables['similarity_transformer_100'].min())
  print("DLS max 100%:           ", variables['similarity_transformer_100'].max())
  print("Sequence accuracy 100%: ", variables['seq_acc_transformer_100'])
  print("\n")
  print("DLS mean 25%:           ", variables['similarity_transformer_25'].mean())
  print("DLS min 25%:            ", variables['similarity_transformer_25'].min())
  print("DLS max 25%:            ", variables['similarity_transformer_25'].max())
  print("Sequence accuracy 25%:  ", variables['seq_acc_transformer_25'])
  print("\n")
  print("DLS mean 50%:           ", variables['similarity_transformer_50'].mean())
  print("DLS min 50%:            ", variables['similarity_transformer_50'].min())
  print("DLS max 50%:            ", variables['similarity_transformer_50'].max())
  print("Sequence accuracy 50%:  ", variables['seq_acc_transformer_50'])
  print("\n")
  print("DLS mean 75%:           ", variables['similarity_transformer_75'].mean())
  print("DLS min 75%:            ", variables['similarity_transformer_75'].min())
  print("DLS max 75%:            ", variables['similarity_transformer_75'].max())
  print("Sequence accuracy 75%:  ", variables['seq_acc_transformer_75'])
  print("\n")
  print("Evaluation LSTM:")
  print("Average pred. trace len:", variables['predicted_trace_length_lstm'].mean())
  print("DLS mean 100%:          ", variables['similarity_lstm_100'].mean())
  print("DLS min 100%:           ", variables['similarity_lstm_100'].min())
  print("DLS max 100%:           ", variables['similarity_lstm_100'].max())
  print("Sequence accuracy 100%: ", variables['seq_acc_lstm_100'])
  print("\n")
  print("DLS mean 25%:           ", variables['similarity_lstm_25'].mean())
  print("DLS min 25%:            ", variables['similarity_lstm_25'].min())
  print("DLS max 25%:            ", variables['similarity_lstm_25'].max())
  print("Sequence accuracy 25%:  ", variables['seq_acc_lstm_25'])
  print("\n")
  print("DLS mean 50%:           ", variables['similarity_lstm_50'].mean())
  print("DLS min 50%:            ", variables['similarity_lstm_50'].min())
  print("DLS max 50%:            ", variables['similarity_lstm_50'].max())
  print("Sequence accuracy 50%:  ", variables['seq_acc_lstm_50'])
  print("\n")
  print("DLS mean 75%:           ", variables['similarity_lstm_75'].mean())
  print("DLS min 75%:            ", variables['similarity_lstm_75'].min())
  print("DLS max 75%:            ", variables['similarity_lstm_75'].max())
  print("Sequence accuracy 75%:  ", variables['seq_acc_lstm_75'])
  print("\n")
  print("Evaluation Random Forest:")
  print("Average pred. trace len:", variables['predicted_trace_length_rf'].mean())
  print("DLS mean 100%:          ", variables['similarity_rf_100'].mean())
  print("DLS min 100%:           ", variables['similarity_rf_100'].min())
  print("DLS max 100%:           ", variables['similarity_rf_100'].max())
  print("Sequence accuracy 100%: ", variables['seq_acc_rf_100'])
  print("\n")
  print("DLS mean 25%:           ", variables['similarity_rf_25'].mean())
  print("DLS min 25%:            ", variables['similarity_rf_25'].min())
  print("DLS max 25%:            ", variables['similarity_rf_25'].max())
  print("Sequence accuracy 25%:  ", variables['seq_acc_rf_25'])
  print("\n")
  print("DLS mean 50%:           ", variables['similarity_rf_50'].mean())
  print("DLS min 50%:            ", variables['similarity_rf_50'].min())
  print("DLS max 50%:            ", variables['similarity_rf_50'].max())
  print("Sequence accuracy 50%:  ", variables['seq_acc_rf_50'])
  print("\n")
  print("DLS mean 75%:           ", variables['similarity_rf_75'].mean())
  print("DLS min 75%:            ", variables['similarity_rf_75'].min())
  print("DLS max 75%:            ", variables['similarity_rf_75'].max())
  print("Sequence accuracy 75%:  ", variables['seq_acc_rf_75'])
  print("\n")
  print("Evaluation Logistical Regression:")
  print("Average pred. trace len:", variables['predicted_trace_length_lg'].mean())
  print("DLS mean 100%:          ", variables['similarity_lg_100'].mean())
  print("DLS min 100%:           ", variables['similarity_lg_100'].min())
  print("DLS max 100%:           ", variables['similarity_lg_100'].max())
  print("Sequence accuracy 100%: ", variables['seq_acc_lg_100'])
  print("\n")
  print("DLS mean 25%:           ", variables['similarity_lg_25'].mean())
  print("DLS min 25%:            ", variables['similarity_lg_25'].min())
  print("DLS max 25%:            ", variables['similarity_lg_25'].max())
  print("Sequence accuracy 25%:  ", variables['seq_acc_lg_25'])
  print("\n")
  print("DLS mean 50%:           ", variables['similarity_lg_50'].mean())
  print("DLS min 50%:            ", variables['similarity_lg_50'].min())
  print("DLS max 50%:            ", variables['similarity_lg_50'].max())
  print("Sequence accuracy 50%:  ", variables['seq_acc_lg_50'])
  print("\n")
  print("DLS mean 75%:           ", variables['similarity_lg_75'].mean())
  print("DLS min 75%:            ", variables['similarity_lg_75'].min())
  print("DLS max 75%:            ", variables['similarity_lg_75'].max())
  print("Sequence accuracy 75%:  ", variables['seq_acc_lg_75'])

  # Write report to file
  # Create filename
  instance_name = variables['filename_processed_dataset'][18:-4]
  output_file_name = timestamp + '_report'+'_evaluation_'+ instance_name + '.txt'
  output_file_path = path_reports + 'evaluation/' + output_file_name

  # Open the file for writing (creates the file if it doesn't exist)
  with open(output_file_path, 'w') as file:
    file.write("Summary:\n\n")
    file.write("Dataset:                {}\n".format(variables['filename_dataset']))
    file.write("Filename interim data:  {}\n".format(variables['filename_interim_dataset']))
    file.write("Filename processed data:{}\n".format(variables['filename_processed_dataset']))
    file.write("Filename variables:     {}\n\n".format(filename_variables))

    file.write("vocab (first 6):        {}\n".format(variables['vocab'][:6]))
    file.write("vocab_size:             {}\n".format(variables['vocab_size']))
    file.write("max_length_trace:       {}\n".format(variables['max_length_trace']))
    file.write("num_traces:             {}\n".format(variables['num_traces']))
    file.write("num_ex_activities:      {}\n".format(variables['num_ex_activities']))
    file.write("num_features:           {}\n".format(variables['num_features']))
    file.write("features:               {}\n".format(variables['features']))
    file.write("interleave:             {}\n\n".format(variables['interleave']))

    file.write("Dataset:\n")
    file.write("Samples in training:    {}\n".format(variables['x_train_shape']))
    file.write("Samples in validation:  {}\n".format(variables['x_val_shape']))
    file.write("Samples in test:        {}\n\n".format(variables['x_test_shape']))

    file.write("Benchmark Dataset:\n")
    file.write("Samples in training:    {}\n".format(variables['x_train_shape_benchmark']))
    file.write("Samples in validation:  {}\n".format(variables['x_val_shape_benchmark']))
    file.write("Samples in test:        {}\n\n".format(variables['x_test_shape_benchmark']))

    file.write("Dataset:\n\n")
    file.write("Prediction length:\n")
    file.write("Prediction length 25%:  {}\n".format(variables['prediction_length_25']))
    file.write("Prediction length 50%:  {}\n".format(variables['prediction_length_50']))
    file.write("Prediction length 75%:  {}\n\n".format(variables['prediction_length_75']))
    file.write("Evaluation Reference data:\n")
    file.write("Average pred. trace len:{}\n\n".format(variables['trace_length_reference'].mean()))
    file.write("Evaluation Reference benchmark data:\n")
    file.write("Average pred. trace len:{}\n\n".format(variables['trace_length_reference_benchmark'].mean()))

    file.write("Evaluation Transformer:\n")
    file.write("Average pred. trace len:{}\n".format(variables['predicted_trace_length_transformer'].mean()))
    file.write("DLS mean 100%:          {}\n".format(variables['similarity_transformer_100'].mean()))
    file.write("DLS min 100%:           {}\n".format(variables['similarity_transformer_100'].min()))
    file.write("DLS max 100%:           {}\n".format(variables['similarity_transformer_100'].max()))
    file.write("Sequence accuracy 100%: {}\n\n".format(variables['seq_acc_transformer_100']))

    file.write("DLS mean 25%:           {}\n".format(variables['similarity_transformer_25'].mean()))
    file.write("DLS min 25%:            {}\n".format(variables['similarity_transformer_25'].min()))
    file.write("DLS max 25%:            {}\n".format(variables['similarity_transformer_25'].max()))
    file.write("Sequence accuracy 25%:  {}\n\n".format(variables['seq_acc_transformer_25']))

    file.write("DLS mean 50%:           {}\n".format(variables['similarity_transformer_50'].mean()))
    file.write("DLS min 50%:            {}\n".format(variables['similarity_transformer_50'].min()))
    file.write("DLS max 50%:            {}\n".format(variables['similarity_transformer_50'].max()))
    file.write("Sequence accuracy 50%:  {}\n\n".format(variables['seq_acc_transformer_50']))

    file.write("DLS mean 75%:           {}\n".format(variables['similarity_transformer_75'].mean()))
    file.write("DLS min 75%:            {}\n".format(variables['similarity_transformer_75'].min()))
    file.write("DLS max 75%:            {}\n".format(variables['similarity_transformer_75'].max()))
    file.write("Sequence accuracy 75%:  {}\n\n".format(variables['seq_acc_transformer_75']))

    file.write("Evaluation LSTM:\n")
    file.write("Average pred. trace len:{}\n".format(variables['predicted_trace_length_lstm'].mean()))
    file.write("DLS mean 100%:          {}\n".format(variables['similarity_lstm_100'].mean()))
    file.write("DLS min 100%:           {}\n".format(variables['similarity_lstm_100'].min()))
    file.write("DLS max 100%:           {}\n".format(variables['similarity_lstm_100'].max()))
    file.write("Sequence accuracy 100%: {}\n\n".format(variables['seq_acc_lstm_100']))
    
    file.write("DLS mean 25%:           {}\n".format(variables['similarity_lstm_25'].mean()))
    file.write("DLS min 25%:            {}\n".format(variables['similarity_lstm_25'].min()))
    file.write("DLS max 25%:            {}\n".format(variables['similarity_lstm_25'].max()))
    file.write("Sequence accuracy 25%:  {}\n\n".format(variables['seq_acc_lstm_25']))

    file.write("DLS mean 50%:           {}\n".format(variables['similarity_lstm_50'].mean()))
    file.write("DLS min 50%:            {}\n".format(variables['similarity_lstm_50'].min()))
    file.write("DLS max 50%:            {}\n".format(variables['similarity_lstm_50'].max()))
    file.write("Sequence accuracy 50%:  {}\n\n".format(variables['seq_acc_lstm_50']))

    file.write("DLS mean 75%:           {}\n".format(variables['similarity_lstm_75'].mean()))
    file.write("DLS min 75%:            {}\n".format(variables['similarity_lstm_75'].min()))
    file.write("DLS max 75%:            {}\n".format(variables['similarity_lstm_75'].max()))
    file.write("Sequence accuracy 75%:  {}\n\n".format(variables['seq_acc_lstm_75']))

    file.write("Evaluation Random Forest:\n")
    file.write("Average pred. trace len:{}\n".format(variables['predicted_trace_length_rf'].mean()))
    file.write("DLS mean 100%:          {}\n".format(variables['similarity_rf_100'].mean()))
    file.write("DLS min 100%:           {}\n".format(variables['similarity_rf_100'].min()))
    file.write("DLS max 100%:           {}\n".format(variables['similarity_rf_100'].max()))
    file.write("Sequence accuracy 100%: {}\n\n".format(variables['seq_acc_rf_100']))

    file.write("DLS mean 25%:           {}\n".format(variables['similarity_rf_25'].mean()))
    file.write("DLS min 25%:            {}\n".format(variables['similarity_rf_25'].min()))
    file.write("DLS max 25%:            {}\n".format(variables['similarity_rf_25'].max()))
    file.write("Sequence accuracy 25%:  {}\n\n".format(variables['seq_acc_rf_25']))

    file.write("DLS mean 50%:           {}\n".format(variables['similarity_rf_50'].mean()))
    file.write("DLS min 50%:            {}\n".format(variables['similarity_rf_50'].min()))
    file.write("DLS max 50%:            {}\n".format(variables['similarity_rf_50'].max()))
    file.write("Sequence accuracy 50%:  {}\n\n".format(variables['seq_acc_rf_50']))

    file.write("DLS mean 75%:           {}\n".format(variables['similarity_rf_75'].mean()))
    file.write("DLS min 75%:            {}\n".format(variables['similarity_rf_75'].min()))
    file.write("DLS max 75%:            {}\n".format(variables['similarity_rf_75'].max()))
    file.write("Sequence accuracy 75%:  {}\n\n".format(variables['seq_acc_rf_75']))

    file.write("Evaluation Logistical Regression:\n")
    file.write("Average pred. trace len:{}\n".format(variables['predicted_trace_length_lg'].mean()))
    file.write("DLS mean 100%:          {}\n".format(variables['similarity_lg_100'].mean()))
    file.write("DLS min 100%:           {}\n".format(variables['similarity_lg_100'].min()))
    file.write("DLS max 100%:           {}\n".format(variables['similarity_lg_100'].max()))
    file.write("Sequence accuracy 100%: {}\n\n".format(variables['seq_acc_lg_100']))
    
    file.write("DLS mean 25%:           {}\n".format(variables['similarity_lg_25'].mean()))
    file.write("DLS min 25%:            {}\n".format(variables['similarity_lg_25'].min()))
    file.write("DLS max 25%:            {}\n".format(variables['similarity_lg_25'].max()))
    file.write("Sequence accuracy 25%:  {}\n\n".format(variables['seq_acc_lg_25']))

    file.write("DLS mean 50%:           {}\n".format(variables['similarity_lg_50'].mean()))
    file.write("DLS min 50%:            {}\n".format(variables['similarity_lg_50'].min()))
    file.write("DLS max 50%:            {}\n".format(variables['similarity_lg_50'].max()))
    file.write("Sequence accuracy 50%:  {}\n\n".format(variables['seq_acc_lg_50']))

    file.write("DLS mean 75%:           {}\n".format(variables['similarity_lg_75'].mean()))
    file.write("DLS min 75%:            {}\n".format(variables['similarity_lg_75'].min()))
    file.write("DLS max 75%:            {}\n".format(variables['similarity_lg_75'].max()))
    file.write("Sequence accuracy 75%:  {}\n\n".format(variables['seq_acc_lg_75']))


  # Confirm that the data has been written to the file
  print("\nReport has been written to '{}'".format(output_file_path))
