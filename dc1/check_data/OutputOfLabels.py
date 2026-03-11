import numpy as np

# Change this path if your files are in a different folder
y_train = np.load("../data/Y_train.npy")

# 1. Check the shape (how many items, and what dimensions)
print("Shape of Y_train:", y_train.shape)

# 2. Print the first 10 labels to see the format
print("First 10 labels:\n", y_train[:10])