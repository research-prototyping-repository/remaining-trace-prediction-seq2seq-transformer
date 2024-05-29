variables = { 
            # Preprocessing Datasets
             'timestamp_preprocessing': None,
             'vocab': None,  
             'vocab_size': None, 
             'max_length_trace': None, 
             'num_traces': None, 
             'num_ex_activities': None, 
             'mapping': None, 
             'num_features': None, 
             'features': None, 
             'interleave': None, 
             'input_features': None, 
             'trace_length_min': None,

             # Data shape 
             'x_train_shape': None, 
             'x_val_shape': None, 
             'x_test_shape': None, 

             # Preprocessing Datasets Benchmark
             'timestamp_preprocessing_benchmark': None,

             # Data shape benchmark
             'x_train_shape_benchmark': None, 
             'x_val_shape_benchmark': None, 
             'x_test_shape_benchmark': None, 

             # Evaluation
             'trace_length_reference': None,
             'trace_length_reference_benchmark': None,
             'prediction_length_25': None,
             'prediction_length_50': None,
             'prediction_length_75': None,

             # Filenames
             'filename_dataset': None, 
             'filename_interim_dataset': None,
             'filename_benchmark_dataset': None,
             'filename_processed_dataset':  None,
             'filename_predictions': None,

             # Models 
             'transformer_model': None, 
             'regression_model': None, 
             'random_forest_model': None, 
             'lstm_model': None, 

             # Transformer parameters
             'num_layers': None,
             'd_model': None,
             'num_heads': None,
             'dff': None,
             'dropout_rate': None,
             'num_epochs': None,
             'batch_size': None,
             'learning_rate': None,

             # Transformer callbacks
             'monitor_early_stopping': None,
             'min_delta': None,
             'patience_early_stopping': None,
             'monitor_checkpoint': None,

             # Transformer training
             'transformer_timestamp_training_start': None,
             'elapsed_time': None,
             'num_epochs_run': None,
             'masked_accuracy': None,
             'val_masked_accuracy': None,
             'masked_loss': None,
             'val_masked_loss': None,

             # Transformer training evaluation
             'transformer_loss': None, 
             'transformer_acc': None,

             # Transformer predictions
             'transformer_timestamp_prediction_start': None,
             'elapsed_time_prediction': None,
             'MAX_TOKENS': None,

             # Transformer real evaluation
             'predicted_trace_length_transformer': None,
             'similarity_transformer_25': None,
             'similarity_transformer_50': None,
             'similarity_transformer_75': None,
             'similarity_transformer_100': None,
             'seq_acc_transformer_25': None,
             'seq_acc_transformer_50': None,
             'seq_acc_transformer_75': None,
             'seq_acc_transformer_100': None, 
             
             # LSTM parameters
             'lstm_num_epochs': None,
             'lstm_batch_size': None,
             'lstm_learning_rate': None,
             
             # LSTM callback
             'lstm_monitor_early_stopping': None,
             'lstm_min_delta': None,
             'lstm_patience_early_stopping': None,
             
             # LSTM Training
             'lstm_timestamp_training_start': None,
             'lstm_elapsed_time': None, 
             'lstm_epochs_run': None,
             'lstm_masked_accuracy': None,
             'lstm_val_masked_accuracy': None,
             'lstm_masked_loss': None,
             'lstm_val_masked_loss': None,

             # LSTM training evaluation
             'lstm_loss': None, 
             'lstm_acc': None,

             # LSTM predictions
             'lstm_timestamp_prediction_start': None,
             'lstm_elapsed_time_predictions': None,

             # LSTM real evaluation
             'predicted_trace_length_lstm': None,
             'similarity_lstm_25': None,
             'similarity_lstm_50': None,
             'similarity_lstm_75': None,
             'similarity_lstm_100': None,
             'seq_acc_lstm_25': None, 
             'seq_acc_lstm_50': None, 
             'seq_acc_lstm_75': None, 
             'seq_acc_lstm_100': None, 

             # Random Forest parameters
             'rf_n_estimators': None,
             'rf_n_jobs': None,

             # Random Forest training
             'rf_timestamp_training_start': None,
             'rf_elapsed_time': None,

             # Random Forest training evaluation
             'rf_acc': None, 

             # Random Forest predictions
             'rf_timestamp_prediction_start': None,
             'rf_elapsed_time_predictions': None,

             # Random Forest real evaluation
             'predicted_trace_length_rf': None,
             'similarity_rf_25': None,
             'similarity_rf_50': None,
             'similarity_rf_75': None,
             'similarity_rf_100': None,
             'seq_acc_rf_25': None, 
             'seq_acc_rf_50': None, 
             'seq_acc_rf_75': None, 
             'seq_acc_rf_100': None, 

             # Logistical Regression parameters
             'logreg_max_iter': None,
             'logreg_n_jobs': None,

             # Logistical Regression training
             'logreg_timestamp_training_start': None,
             'logreg_elapsed_time': None,

             # Logistical Regression training evaluation
             'logreg_acc': None,

             # Logistical Regression predictions
             'logreg_timestamp_prediction_start': None,
             'logreg_elapsed_time_predictions': None,

             # Logistical Regression real evaluation
             'predicted_trace_length_lg': None,
             'similarity_lg_25': None,
             'similarity_lg_50': None,
             'similarity_lg_75': None,
             'similarity_lg_100': None,
             'seq_acc_lg_25': None,
             'seq_acc_lg_50': None,
             'seq_acc_lg_75': None,
             'seq_acc_lg_100': None

             }
