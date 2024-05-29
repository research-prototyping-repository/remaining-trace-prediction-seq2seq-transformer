# Import libriaires
import time
import datetime

# ------------------------------------------------------------------------------

# Time Function
def tic():
    """Start the timer."""
    global _start_time
    _start_time = time.time()

def toc():
    """Stop the timer and print the elapsed time."""
    if '_start_time' in globals():
        elapsed_time = time.time() - _start_time
        print(f"\nElapsed time: {elapsed_time:.6f} seconds")
    else:
        print("Error: Start time not set. Call tic() before calling toc().")

    return elapsed_time

# Get timestamp
def get_timestamp():
    # Get the timestamp
    current_datetime = datetime.datetime.now()
    timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    return timestamp