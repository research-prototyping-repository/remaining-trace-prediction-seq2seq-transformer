# Import libraries
import matplotlib.pyplot as plt

# Functions
def display_training_curves(training, validation, title, subplot):
  """
  This function displays the curves of the training the model.

  Input:
  - training: list which contains the training accuracy or loss values from training
  - validation: list which contains the validation accuracy or loss values from training
  - title: str which is used as title
  - subplot: int specifies in which subplot the data is plotted (2 plots in one column would be 211 & 212)

  Output:
  - ax: the matplotlib subplot
  """

    # Create plot
    ax = plt.subplot(subplot)
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    ax.set_xlabel('epoch')
    ax.legend(['training', 'validation'])

  return ax