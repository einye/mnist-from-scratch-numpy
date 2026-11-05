import numpy as np
import tensorflow as tf
import os
import pandas as pd

# Directory to store the data
data_dir = "data"

# Function to download MNIST dataset using TensorFlow
def download_and_convert_mnist():
    # Download the MNIST dataset using TensorFlow
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
       
    # Flatten the images to 1D arrays (28x28 -> 784)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Ensure the directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Save to CSV
    train_data = np.hstack((y_train.reshape(-1, 1), X_train))  # Add labels as the first column
    test_data = np.hstack((y_test.reshape(-1, 1), X_test))  # Add labels as the first column
    
    # Convert to DataFrame for easy CSV handling
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    # Save the data to CSV files
    train_df.to_csv(os.path.join(data_dir, 'mnist_train.csv'), index=False, header=False)
    test_df.to_csv(os.path.join(data_dir, 'mnist_test.csv'), index=False, header=False)
    
    print("MNIST data downloaded and saved in CSV.")

# Run the function
if __name__ == "__main__":
    download_and_convert_mnist()
