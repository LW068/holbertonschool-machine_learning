#!/usr/bin/env python3

# Import necessary modules
import numpy as np
import zipfile
from io import BytesIO

# Import required functionalities from other modules
DeepNN = __import__('26-deep_neural_network').DeepNeuralNetwork
encode_labels = __import__('24-one_hot_encode').one_hot_encode
decode_labels = __import__('25-one_hot_decode').one_hot_decode

# Set path to the zipped dataset
dataset_zip_path = 'data/Binary_Train.zip'

# Open the zip file in read mode
zip_file = zipfile.ZipFile(dataset_zip_path, 'r')

# Access the npz file within the zip file
npz_file_in_zip = zip_file.open('Binary_Train.npz')

# Load the file into a BytesIO object
npz_bytes = BytesIO(npz_file_in_zip.read())

# Load the dataset from the npz file
dataset = np.load(npz_bytes)
X_train_3D, Y_train = dataset['X'], dataset['Y']

# Reshape the training set
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T

# Set seed for reproducibility
np.random.seed(0)

# Create an instance of the Deep Neural Network and train it
deepNN = DeepNN(X_train.shape[0], [3, 1])
A, cost = deepNN.train(X_train, Y_train, iterations=500, graph=False)

# Save the trained model
deepNN.save('26-output')

# Delete the current instance to test loading in the next step
del deepNN

# Load the saved model
loaded_model = DeepNN.load('26-output.pkl')

# Evaluate the loaded model
A_loaded, cost_loaded = loaded_model.evaluate(X_train, Y_train)

# Compare the performance of the original and loaded model
print(np.array_equal(A, A_loaded) and cost == cost_loaded)
