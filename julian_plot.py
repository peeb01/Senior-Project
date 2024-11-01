import keras.activations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf
from keras.layers import Dense, LSTM, InputLayer
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


df = pd.read_csv('juain_date405.csv').tail(50000)
df['time'] = pd.to_datetime(df['time'])
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
df['hour'] = df['time'].dt.hour
df['minute'] = df['time'].dt.minute

def windows(df, window_size = 5):
    df_np = df
    x = []
    y = []
    for i in range(len(df_np)-window_size):
        row = [[a] for a in df_np[i:i+window_size]]
        x.append(row)
        label = df_np[i+window_size]
        y.append(label)
    return np.array(x), np.array(y)

WINDOWS_SIZE = 48

x, y = windows(df['hour'].values, WINDOWS_SIZE)


x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.7, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.33, shuffle=False)

model = Sequential()
model.add(InputLayer((WINDOWS_SIZE, 1)))
model.add(LSTM(32))
model.add(Dense(64, activation=keras.activations.relu))
model.add(Dense(32, activation=keras.activations.relu))
model.add(Dense(1, activation=keras.activations.linear))

patience = 2
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='min')
model.compile(loss=MeanSquaredError(), optimizer=Adam())

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=30, callbacks=[early_stopping])

pred = model.predict(x_test).flatten()
print(pred.shape)
print(y_test.shape)

train_result = pd.DataFrame({'Pred': pred, 'Actual': y_test})

plt.plot(train_result['Actual'][-200:], label='Actual')
plt.plot(train_result['Pred'][-200:], label='Predict')
plt.legend()
plt.show()

# model.save('minute_train_windows48.h5')