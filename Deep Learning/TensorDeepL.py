import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# tensorflow training
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def custom_sort(filename):
    if filename == 'asteroid_last.csv':
        return float('inf') 
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0


def data_pre(directory : str, earthquake_path : str, num_rows=30369, skip_row=0):
    """
    Prepares and reads data from earthquake and asteroid datasets.

    Args:
        directory (str): Directory of dataset that stores positions of .csv files.
        earthquake_path (str): Path of earthquake dataset.
        num_rows (int): Number of rows to read in each part. Default is 30369.
        skip_row (int): Row to start reading from. Default is 0.
    
    Returns:
        pandas.DataFrame: Dataframe of earthquake data.
        pandas.DataFrame: Dataframe of star positions.
        int: Updated skip row to next read.
    """
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    csv_files = sorted(csv_files, key=custom_sort)

    earthquake = pd.read_csv(earthquake_path, skiprows=skip_row , nrows=num_rows)['cluster_e']

    # Read data for create columns name.
    c = pd.read_csv(os.path.join(directory, csv_files[0]), skiprows=0 , nrows=1)       
    for i in range(1, len(csv_files)):
        ndf = pd.read_csv(os.path.join(directory, csv_files[i]), skiprows=0, nrows=1).iloc[:, 1:]
        c = pd.concat([c, ndf], axis=1)
    col = c.columns

    # Read asteroid data with the same number of rows
    asteroids = pd.read_csv(os.path.join(directory, csv_files[0]), skiprows=skip_row, nrows=num_rows, header=None)
    for i in range(1, len(csv_files)):
        ndf = pd.read_csv(os.path.join(directory, csv_files[i]), skiprows=skip_row, nrows=num_rows, header=None).iloc[:, 1:]
        asteroids = pd.concat([asteroids, ndf], axis=1)
    asteroids.columns = col
    
    skip_row += num_rows
    
    return asteroids, earthquake, skip_row

def data_pre_tf(directory: str, earthquake_path: str, num_rows=30369, skip_row=0):
    """
    Prepares and reads data from earthquake and asteroid datasets using TensorFlow.

    Args:
        directory (str): Directory of dataset that stores positions of .csv files.
        earthquake_path (str): Path of earthquake dataset.
        num_rows (int): Number of rows to read in each part. Default is 30369.
        skip_row (int): Row to start reading from. Default is 0.

    Returns:
        tf.data.Dataset: Dataset of star positions.
        tf.data.Dataset: Dataset of earthquake data.
        int: Updated skip row to next read.
    """
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    csv_files = sorted(csv_files, key=custom_sort)

    def decode_line(line, num_columns):
        record_defaults = [[0.0] for _ in range(num_columns)]
        return tf.io.decode_csv(line, record_defaults)

    def load_earthquake_data():
        earthquake_dataset = tf.data.experimental.CsvDataset(
            earthquake_path,
            [tf.float32],  # Adjust based on the actual data types in your CSV
            header=True,
            skip_rows=skip_row
        ).batch(num_rows)
        return earthquake_dataset

    earthquake_dataset = load_earthquake_data()

    def get_column_names(file_path):
        with open(file_path, 'r') as f:
            return f.readline().strip().split(',')

    col_names = get_column_names(os.path.join(directory, csv_files[0]))
    num_columns = len(col_names)

    def load_asteroid_data():
        datasets = []
        for csv_file in csv_files:
            file_path = os.path.join(directory, csv_file)
            new_dataset = tf.data.experimental.CsvDataset(
                file_path,
                [tf.float32] * num_columns,
                header=False,
                skip_rows=skip_row
            )
            datasets.append(new_dataset)
        asteroid_dataset = tf.data.Dataset.zip(tuple(datasets))
        asteroid_dataset = asteroid_dataset.map(lambda *x: tf.concat(x, axis=-1)).batch(num_rows)
        return asteroid_dataset

    asteroid_dataset = load_asteroid_data()

    skip_row += num_rows

    return asteroid_dataset, earthquake_dataset, skip_row


def get_y(y_data):
    """OneHotEncode"""
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y_data)
    return y

def ann_model(x_shape, y_shape):
    """
    Creates an Artificial Neural Network model.
    
    Args:
        x_shape (tuple): Shape of the input data.
        y_shape (tuple): Shape of the output data.
    
    Returns:
        keras.Model: The ANN model.
    """
    model = Sequential()
    model.add(Dense(units=1000, input_shape=(x_shape[1],), activation=tf.keras.activations.softmax))
    model.add(BatchNormalization())
    model.add(Dense(units=500, activation=tf.keras.activations.softmax))
    model.add(BatchNormalization())
    model.add(Dense(y_shape[1], activation=tf.keras.activations.softmax))
    return model


def cnn_model(x_shape, y_shape):
    """
    Creates a Convolutional Neural Network model.
    
    Args:
        x_shape (tuple): Shape of the input data.
        y_shape (tuple): Shape of the output data.
    
    Returns:
        keras.Model: The CNN model.
    """
    model = tf.keras.models.Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, input_shape=(x_shape[1], 1), activation=tf.keras.activations.softmax))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation=tf.keras.activations.softmax))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=3, activation=tf.keras.activations.softmax))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())

    model.add(Dense(units=64, activation=tf.keras.activations.softmax))
    model.add(Dense(units=500, activation=tf.keras.activations.softmax))
    model.add(Dense(units=y_shape[1], activation=tf.keras.activations.softmax))
    return model


def compile_fit(model, x_train, y_train, x_val, y_val, epochs=5, early_stopping=False):
    """
    Compiles and fits the model with the given training and validation data.
    
    Args:
        model (tf.keras.Model): The model to compile and fit.
        x_train (ndarray): Training features.
        y_train (ndarray): Training labels.
        x_val (ndarray): Validation features.
        y_val (ndarray): Validation labels.
        epochs (int): Number of epochs to train. Default is 5.
        early_stopping (bool): Whether to use early stopping. Default is False.
    
    Returns:
        History: A record of training loss values and metrics values at successive epochs.
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
                  loss=tf.keras.losses.MeanSquaredError(), 
                  metrics=tf.keras.metrics.Accuracy())

    callbacks = []
    if early_stopping:
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        callbacks.append(early_stopping_callback)
        
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_data=(x_val, y_val), callbacks=callbacks)

    return history, model

def model_test(model, x_test, y_test):
    """
    Evaluates the trained model on the test data.
    
    Args:
        model (tf.keras.Model): The trained model.
        x_test (ndarray): Test features.
        y_test (ndarray): Test labels.
    
    Returns:
        float: Test loss.
        float: Test accuracy.
    """
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    return test_loss, test_accuracy

