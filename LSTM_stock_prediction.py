# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 00:57:01 2024

@author: SameerRangwala
"""

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

# Function to create lagged features
def create_lagged_features(data, n_lags=5):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(n_lags, 0, -1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.columns = [f'target-{i}' for i in range(n_lags, 0, -1)] + ['target']
    df.dropna(inplace=True)
    return df

ticker = 'TCS.NS'

ohlcv = yf.download(ticker, period = '3y')

data = ohlcv['Adj Close']

plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title("TCS stock price")
plt.show


# Define the split indices
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.15)
test_size = len(data) - train_size - val_size

train_data = data.iloc[:train_size]
val_data = data.iloc[train_size:train_size + val_size]
test_data = data.iloc[train_size + val_size:]

# Initialize and fit the MinMaxScaler on the training data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))

# Transform the validation and test data using the scaler fitted on training data
val_scaled = scaler.transform(val_data.values.reshape(-1, 1))
test_scaled = scaler.transform(test_data.values.reshape(-1, 1))

# Convert scaled data back to pandas Series (optional)
train_scaled_series = pd.Series(train_scaled.flatten(), index=train_data.index)
val_scaled_series = pd.Series(val_scaled.flatten(), index=val_data.index)
test_scaled_series = pd.Series(test_scaled.flatten(), index=test_data.index)

# Print the index boundaries for each subset
print("Training data range:", train_data.index[0], "to", train_data.index[-1])
print("Validation data range:", val_data.index[0], "to", val_data.index[-1])
print("Test data range:", test_data.index[0], "to", test_data.index[-1])

# Plotting the data
plt.figure(figsize=(14, 7))

plt.plot(train_scaled_series, label='Training Data', color='blue')
plt.plot(val_scaled_series, label='Validation Data', color='orange')
plt.plot(test_scaled_series, label='Test Data', color='green')

plt.title('Scaled Stock Data')
plt.xlabel('Date Index')
plt.ylabel('Scaled Adjusted Close Price')
plt.legend()

plt.show()

# Combine the scaled data back into a single series
scaled_data = np.concatenate((train_scaled, val_scaled, test_scaled), axis=0)
scaled_series = pd.Series(scaled_data.flatten(), index=data.index)

# Create lagged features for the combined dataset
lagged_data = create_lagged_features(scaled_series)

# Define new split indices, adjusted for the lagged features
train_lagged_size = train_size - 5
val_lagged_size = val_size
test_lagged_size = test_size

# Split the lagged data back into training, validation, and test sets
train_lagged = lagged_data[:train_lagged_size]
val_lagged = lagged_data[train_lagged_size:train_lagged_size + val_lagged_size]
test_lagged = lagged_data[train_lagged_size + val_lagged_size:]

model = Sequential()
model.add(InputLayer((5,1)))
model.add(LSTM(64))
model.add(Dense(32, 'relu'))
model.add(Dense(32, 'relu'))
model.add(Dense(1))

model.summary()

cp1 = ModelCheckpoint('/model.keras', save_best_only=True)
model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])


# Prepare the data for LSTM input
def create_lstm_dataset(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

X_train, y_train = create_lstm_dataset(train_lagged)
X_val, y_val = create_lstm_dataset(val_lagged)
X_test, y_test = create_lstm_dataset(test_lagged)

# Reshape the input to be [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs =100)

train_predictions= model.predict(X_train).flatten()
val_predictions = model.predict(X_val).flatten()
test_predictions = model.predict(X_test).flatten()


# Inverse transform the scaled data
train_predictions = scaler.inverse_transform(train_predictions.reshape(-1, 1)).flatten()
val_predictions = scaler.inverse_transform(val_predictions.reshape(-1, 1)).flatten()
test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()

y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_val_actual = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Plotting the training predictions vs actual values
plt.figure(figsize=(14, 7))
plt.plot(y_train_actual, label='Actual Training Values', color='blue')
plt.plot(train_predictions, label='Predicted Training Values', color='red')
plt.title('Training Data')
plt.xlabel('Index')
plt.ylabel('Actual Price')
plt.legend()
plt.show()

# Plotting the validation predictions vs actual values
plt.figure(figsize=(14, 7))
plt.plot(y_val_actual, label='Actual Validation Values', color='blue')
plt.plot(val_predictions, label='Predicted Validation Values', color='red')
plt.title('Validation Data')
plt.xlabel('Index')
plt.ylabel('Actual Price')
plt.legend()
plt.show()

# Plotting the test predictions vs actual values
plt.figure(figsize=(14, 7))
plt.plot(y_test_actual, label='Actual Test Values', color='blue')
plt.plot(test_predictions, label='Predicted Test Values', color='red')
plt.title('Test Data')
plt.xlabel('Index')
plt.ylabel('Actual Price')
plt.legend()
plt.show()