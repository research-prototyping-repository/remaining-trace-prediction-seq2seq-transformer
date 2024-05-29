# Import libraries
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pm4py
from tqdm import tqdm
from src.data.functions_preprocessing_data import extract_meta_data, create_traces

# Functions
# ------------------------------------------------------------------------------
# @title Functions Plot
def plot_histogram_trace_length(df):
    """
    This function plots a histogram of trace lengths based on the given dataframe.

    Input:
    - df: dataframe containing the data

    Output:
    - None
    """

    # Group by case and count the number of events in each case
    trace_lengths = df.groupby('case:concept:name').size()

    # Plot the histogram
    plt.hist(trace_lengths, bins=10, edgecolor='black')

    # Set labels and title
    plt.xlabel('Trace Length')
    plt.ylabel('Frequency')
    plt.title('Histogram of Trace Length')

    # Display the plot
    plt.show()

# ------------------------------------------------------------------------------

def plot_histogram_activity_occurrence(df):
    """
    This function plots a histogram of activity occurrences based on the given dataframe.

    Input:
    - df: dataframe containing the data

    Output:
    - None
    """

    # Group by case and count the number of events for each activity
    activity_counts = df.groupby('concept:name').size().sort_values(ascending=False)

    # Plot the histogram
    plt.bar(activity_counts.index, activity_counts.values)

    # Set labels and title
    plt.xlabel('Activities')
    plt.ylabel('Frequency')
    plt.title('Occurrences of Activities')
    plt.xticks(rotation=90)

    # Display the plot
    plt.show()

# ------------------------------------------------------------------------------

def plot_heatmap(df, features):
  """
  This function plots an heatmap, in which the position of an event in the trace is shown with respect to it's occurance at that specific position.

  Input:
  - df: dataframe containing the event log

  Output:
  - None
  """
  # Extract the needed values
  vocabulary, vocabulary_size, max_length_trace, num_traces, num_ex_activities, num_features = extract_meta_data(df,'case:concept:name',features, output = False)

  # Remove the reservesed tokens except for the <unk> token
  events = np.concatenate((np.array([vocabulary[1]]), vocabulary[4:]))

  # Empty heatmap matrix
  heatmap = np.zeros((len(events), max_length_trace*num_features))

  # Replace nan values in df
  df = df.fillna(vocabulary[1])

  # Create traces
  traces = create_traces(df, features, interleave = True)

  # Count occurrences of events at each position
  for sequence in tqdm(traces, desc ='Generating heatmap data'):
      for position, event in enumerate(sequence):
          event_index = np.where(events == event)[0][0]  # Find the index of the event
          heatmap[event_index, position] += 1  # Increment the corresponding position in the heatmap

  # Normalize the heatmap
  normalized_heatmap = heatmap / np.max(heatmap)

  # Set the figure size
  fig, ax = plt.subplots(figsize=(12, 8))

  # Visualize the normalized heatmap
  im = ax.imshow(normalized_heatmap, cmap='hot', interpolation='nearest')
  plt.xticks(range(max_length_trace * num_features))
  plt.yticks(range(len(events)), events)
  plt.xlabel('Position')
  plt.ylabel('Event')
  plt.title('Normalized Event Frequency Heatmap')
  plt.colorbar(im, label='Normalized Frequency')
  plt.show()

# ------------------------------------------------------------------------------

def plot_bpmn_map(df):
  """
  This function takes an eventlog in the form of a dataframe and converts it into a bpmn_map.

  Input:
  - df: dataframe containing the event log

  Output:
  - process_model: variable containing the process model
  """
  # Create and display process model
  process_model = pm4py.discover_bpmn_inductive(df)
  pm4py.view_bpmn(process_model)

  return process_model

def plot_petri_net(bpmn_map):
    """
    This function converts the given BPMN map to a Petri net and visualizes it.

    Input:
    - bpmn_map: BPMN map object

    Output:
    - net: Petri net object
    - im: initial marking
    - fm: final marking
    """
    
    # Create and display petri net
    net, im, fm = pm4py.convert_to_petri_net(bpmn_map)
    pm4py.view_petri_net(net, im, fm, format='png')

    return net, im, fm

# ------------------------------------------------------------------------------

def plot_dfg(df, num_variants=5):
    """
    This function filters the given dataframe and discovers a Directly-Follows Graph (DFG) based on the filtered data.
    It then visualizes the DFG.

    Inputs:
    - df: dataframe containing the data
    - num_variants: number of top variants to consider (default: 5)

    Outputs:
    - dfg: Directly-Follows Graph object
    - start_activities: start activities in the DFG
    - end_activities: end activities in the DFG
    """

    # Filter dataframe for top variant
    filtered_dataframe = pm4py.filter_variants_top_k(df, num_variants, activity_key='concept:name', timestamp_key='time:timestamp', case_id_key='case:concept:name')
    # Create and plot directly follows graph
    dfg, start_activities, end_activities = pm4py.discover_dfg(filtered_dataframe, case_id_key='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')
    pm4py.view_dfg(dfg, start_activities, end_activities)

    return dfg, start_activities, end_activities

# ------------------------------------------------------------------------------