# ## Work environment

# Import
import os
import pickle
import numpy as np
import math

from src.evaluation.functions_evaluation import prediction_statistics, damerau_levenshtein_similarity, evaluate_seq_accuracy
from src.visualization.functions_evaluation_visualize import plot_levenshtein_distance
from src.general.functions_time import get_timestamp
from src.general.functions_report import report_evaluation

# Set tensorflow to GPU-only (data is stored as tensors even when tf is not used)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# ## Parameter
# Set path variables
path_raw = 'data/raw/'
path_interim = 'data/interim/'
path_benchmark = 'data/benchmark/'
path_data = 'data/processed/'
path_control = 'data/control/'
path_predictions = 'data/predictions/'
path_models = 'models/'
path_reports = 'reports/'

# List of all control files
files = ['variables_helpdesk.pkl','variables_helpdesk_true.pkl','variables_helpdesk_false.pkl','variables_bpi2012.pkl','variables_bpi2012_true.pkl','variables_bpi2012_false.pkl','variables_bpi2017.pkl','variables_bpi2017_true.pkl','variables_bpi2017_false.pkl','variables_bpi2018.pkl','variables_bpi2018_true.pkl','variables_bpi2018_false.pkl','variables_bpi2019.pkl','variables_bpi2019_true.pkl','variables_bpi2019_false.pkl']

# Iterate over control files
for filename_variables in files:

    # Get timestamp
    timestamp = get_timestamp()


    # Initalize variables
    print("Filename variables: ", filename_variables)

    with open(path_control + filename_variables, 'rb') as file:
        variables = pickle.load(file)


    #--- Evaluation ------------------------------------------------------------------


    # Load data
    predictions = np.load(path_predictions + variables['filename_predictions'])

    # Load predictions
    y_test = predictions['y_test'][:,1:] # Remove start token
    y_test_benchmark = predictions['y_test_benchmark']
    y_pred_transformer = predictions['y_pred_transformer'][:,1:] # Remove start token
    y_pred_lstm = predictions['y_pred_lstm']
    y_pred_rf = predictions['y_pred_rf']
    y_pred_lg = predictions['y_pred_rg']


    # ##  Reference data
    variables['trace_length_reference'], avg_pred_trace_length = prediction_statistics(y_test, path_reports, 'Reference data', timestamp, bins = 100, ref = True)
    variables['trace_length_reference_benchmark'], avg_pred_trace_length = prediction_statistics(y_test_benchmark, path_reports, 'Reference data benchmark', timestamp, bins = 100, ref = True)


     # Calculate percentiles
    variables['prediction_length_25'] = math.ceil(np.percentile(variables['trace_length_reference'], 25))
    variables['prediction_length_50'] = math.ceil(np.percentile(variables['trace_length_reference'], 50))
    variables['prediction_length_75'] = math.ceil(np.percentile(variables['trace_length_reference'], 75))
    
    # ## Transformer
    # Calculate prediction statistics
    variables['predicted_trace_length_transformer'], avg_pred_trace_length = prediction_statistics(y_pred_transformer, path_reports, variables['transformer_model'], timestamp, bins = 100)
    
    # ### DLS
    # Calculate similarity for the full sequence length
    variables['similarity_transformer_100'] = damerau_levenshtein_similarity(y_test, y_pred_transformer, variables['trace_length_reference'], variables['predicted_trace_length_transformer'], variables["MAX_TOKENS"])
    # Calculate similarity for 25% sequence length
    variables['similarity_transformer_25'] = damerau_levenshtein_similarity(y_test, y_pred_transformer, variables['trace_length_reference'], variables['predicted_trace_length_transformer'], variables['prediction_length_25'])
    # Calculate similarity for 50% sequence length
    variables['similarity_transformer_50'] = damerau_levenshtein_similarity(y_test, y_pred_transformer, variables['trace_length_reference'], variables['predicted_trace_length_transformer'], variables['prediction_length_50'])
    # Calculate similarity for 75% sequence length
    variables['similarity_transformer_75'] = damerau_levenshtein_similarity(y_test, y_pred_transformer, variables['trace_length_reference'], variables['predicted_trace_length_transformer'], variables['prediction_length_75'])

    # ### Sequence accuracy
    # Calculate seq_acc for the full sequence length
    variables['seq_acc_transformer_100'] = evaluate_seq_accuracy(y_pred_transformer, y_test)
    # Calculate seq_acc for 25% sequence length
    variables['seq_acc_transformer_25'] = evaluate_seq_accuracy(y_pred_transformer[:,:variables['prediction_length_25']], y_test[:,:variables['prediction_length_25']])
    # Calculate seq_acc for 50% sequence length
    variables['seq_acc_transformer_50'] = evaluate_seq_accuracy(y_pred_transformer[:,:variables['prediction_length_50']], y_test[:,:variables['prediction_length_50']])
    # Calculate seq_acc for 75% sequence length
    variables['seq_acc_transformer_75'] = evaluate_seq_accuracy(y_pred_transformer[:,:variables['prediction_length_75']], y_test[:,:variables['prediction_length_75']])


    # ## LSTM
    variables['predicted_trace_length_lstm'], avg_pred_trace_length = prediction_statistics(y_pred_lstm, path_reports, variables['lstm_model'], timestamp, bins = 100)

    # ### DLS
    # Calculate similarity for the full sequence length
    variables['similarity_lstm_100'] = damerau_levenshtein_similarity(y_test_benchmark, y_pred_lstm, variables['trace_length_reference_benchmark'], variables['predicted_trace_length_lstm'], variables["MAX_TOKENS"])
    # Calculate similarity for 25% sequence length
    variables['similarity_lstm_25'] = damerau_levenshtein_similarity(y_test_benchmark, y_pred_lstm, variables['trace_length_reference_benchmark'], variables['predicted_trace_length_lstm'], variables['prediction_length_25'])
    # Calculate similarity for 50% sequence length
    variables['similarity_lstm_50'] = damerau_levenshtein_similarity(y_test_benchmark, y_pred_lstm, variables['trace_length_reference_benchmark'], variables['predicted_trace_length_lstm'], variables['prediction_length_50'])
    # Calculate similarity for 75% sequence length
    variables['similarity_lstm_75'] = damerau_levenshtein_similarity(y_test_benchmark, y_pred_lstm, variables['trace_length_reference_benchmark'], variables['predicted_trace_length_lstm'], variables['prediction_length_75'])


    # ### Sequence Accuracy
    # Calculate sequence accuracy for the full sequence length
    variables['seq_acc_lstm_100'] = evaluate_seq_accuracy(y_pred_lstm, y_test_benchmark)
    # Calculate sequence accuracy for 25% of the sequence length
    variables['seq_acc_lstm_25'] = evaluate_seq_accuracy(y_pred_lstm[:,:variables['prediction_length_25']], y_test_benchmark[:,:variables['prediction_length_25']])
    # Calculate sequence accuracy for 25% of the sequence length
    variables['seq_acc_lstm_50'] = evaluate_seq_accuracy(y_pred_lstm[:,:variables['prediction_length_50']], y_test_benchmark[:,:variables['prediction_length_50']])
    # Calculate sequence accuracy for 25% of the sequence length
    variables['seq_acc_lstm_75'] = evaluate_seq_accuracy(y_pred_lstm[:,:variables['prediction_length_75']], y_test_benchmark[:,:variables['prediction_length_75']])


    # ## Random Forest
    variables['predicted_trace_length_rf'], avg_pred_trace_length = prediction_statistics(y_pred_rf, path_reports, variables['random_forest_model'], timestamp, bins = 100)

    # ### DLS
    # Calculate similarity over the full sequence length
    variables['similarity_rf_100'] = damerau_levenshtein_similarity(y_test_benchmark, y_pred_rf, variables['trace_length_reference_benchmark'], variables['predicted_trace_length_rf'], variables["MAX_TOKENS"])
    # Calculate similarity over 25% sequence length
    variables['similarity_rf_25'] = damerau_levenshtein_similarity(y_test_benchmark, y_pred_rf, variables['trace_length_reference_benchmark'], variables['predicted_trace_length_rf'], variables["prediction_length_25"])
    # Calculate similarity over 50% sequence length
    variables['similarity_rf_50'] = damerau_levenshtein_similarity(y_test_benchmark, y_pred_rf, variables['trace_length_reference_benchmark'], variables['predicted_trace_length_rf'], variables["prediction_length_50"])
    # Calculate similarity over 75% sequence length
    variables['similarity_rf_75'] = damerau_levenshtein_similarity(y_test_benchmark, y_pred_rf, variables['trace_length_reference_benchmark'], variables['predicted_trace_length_rf'], variables["prediction_length_75"])


    # ### Sequence Accuracy
    # Calculate sequence accuracy for the complete sequence length
    variables['seq_acc_rf_100'] = evaluate_seq_accuracy(y_pred_rf, y_test_benchmark)
    # Calculate sequence accuracy for 25% of the sequence length
    variables['seq_acc_rf_25'] = evaluate_seq_accuracy(y_pred_rf[:,:variables['prediction_length_25']], y_test_benchmark[:,:variables['prediction_length_25']])
    # Calculate sequence accuracy for 50% of the sequence length
    variables['seq_acc_rf_50'] = evaluate_seq_accuracy(y_pred_rf[:,:variables['prediction_length_50']], y_test_benchmark[:,:variables['prediction_length_50']])
    # Calculate sequence accuracy for 75% of the sequence length
    variables['seq_acc_rf_75'] = evaluate_seq_accuracy(y_pred_rf[:,:variables['prediction_length_75']], y_test_benchmark[:,:variables['prediction_length_75']])


    # ## Logistical Regression
    # Prediction statistics
    variables['predicted_trace_length_lg'], avg_pred_trace_length = prediction_statistics(y_pred_lg, path_reports, variables['regression_model'], timestamp, bins = 100)

    # ### DLS
    # Calculate similarity for complete sequence length
    variables['similarity_lg_100'] = damerau_levenshtein_similarity(y_test_benchmark, y_pred_lg, variables['trace_length_reference_benchmark'], variables['trace_length_reference_benchmark'], variables["MAX_TOKENS"])
    # Calculate similarity over 25% sequence length
    variables['similarity_lg_25'] = damerau_levenshtein_similarity(y_test_benchmark, y_pred_lg, variables['trace_length_reference_benchmark'], variables['trace_length_reference_benchmark'], variables["prediction_length_25"])
    # Calculate similarity over 50% sequence length
    variables['similarity_lg_50'] = damerau_levenshtein_similarity(y_test_benchmark, y_pred_lg, variables['trace_length_reference_benchmark'], variables['trace_length_reference_benchmark'], variables["prediction_length_50"])
    # Calculate similarity over 75% sequence length
    variables['similarity_lg_75'] = damerau_levenshtein_similarity(y_test_benchmark, y_pred_lg, variables['trace_length_reference_benchmark'], variables['trace_length_reference_benchmark'], variables["prediction_length_75"])


    # ### Sequence Accuracy
    # Calculate sequence accuracy for complete sequence length
    variables['seq_acc_lg_100'] = evaluate_seq_accuracy(y_pred_lg, y_test_benchmark)
    # Calculate sequence accuracy for 25% of the sequence length
    variables['seq_acc_lg_25'] = evaluate_seq_accuracy(y_pred_lg[:,:variables['prediction_length_25']], y_test_benchmark[:,:variables['prediction_length_25']])
    # Calculate sequence accuracy for 50% of the sequence length
    variables['seq_acc_lg_50'] = evaluate_seq_accuracy(y_pred_lg[:,:variables['prediction_length_50']], y_test_benchmark[:,:variables['prediction_length_50']])
    # Calculate sequence accuracy for 75% of the sequence length
    variables['seq_acc_lg_75'] = evaluate_seq_accuracy(y_pred_lg[:,:variables['prediction_length_75']], y_test_benchmark[:,:variables['prediction_length_75']])


    # ### Generate Report 
    # Generate evaluation report
    report_evaluation(filename_variables, variables, timestamp, path_reports)

    del timestamp, variables, predictions, y_test, y_test_benchmark, y_pred_transformer, y_pred_lstm, y_pred_rf, y_pred_lg


