import matplotlib.pyplot as plt

def plot_history(train_accuracies, test_accuracies, train_losses, test_losses, filename: str):
#   acc = history.history['accuracy']

  epochs = range(len(train_losses))
  
  plt.figure(figsize=(20,4))
  plt.subplot(1,2,1)
  plt.plot(epochs, train_accuracies, label='Training accuracy')
  plt.plot(epochs, test_accuracies, 'r', label='Validation accuracy')
  plt.title('Training and validation acc')
  plt.legend()

  plt.subplot(1,2,2)
  plt.plot(epochs, train_losses, label='Training loss')
  plt.plot(epochs, test_losses, 'r', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()

  plt.savefig(filename)