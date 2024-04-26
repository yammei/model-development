import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('model-fit-data.csv')

# Plot training and validation loss
plt.plot(df['Epoch'], df['Loss'], 'bo', label='Training Loss')
plt.plot(df['Epoch'], df['Val_Loss'], 'ro', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.plot(df['Epoch'], df['Accuracy'], 'bo', label='Training Accuracy')
plt.plot(df['Epoch'], df['Val_Accuracy'], 'ro', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
