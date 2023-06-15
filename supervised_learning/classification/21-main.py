#!/usr/bin/env python3

import numpy as np
import zipfile
import tempfile

Deep = __import__('21-deep_neural_network').DeepNeuralNetwork

zip_path = 'data/Binary_Train.zip'
zip_file = zipfile.ZipFile(zip_path, 'r')
npz_file = zip_file.open('Binary_Train.npz')

with tempfile.NamedTemporaryFile() as tmp_file:
    tmp_file.write(npz_file.read())
    tmp_file.flush()
    
    lib_train = np.load(tmp_file.name)
    X_3D, Y = lib_train['X'], lib_train['Y']
    X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X.shape[0], [5, 3, 1])
A, cache = deep.forward_prop(X)
deep.gradient_descent(Y, cache, 0.5)
print(deep.weights)
