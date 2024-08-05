import keras.activations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf
from keras.models import load_model
from sklearn.model_selection import train_test_split

model = load_model('models\day_train.h5')

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

WINDOWS_SIZE = 5

x, y = windows(df['day'].values, WINDOWS_SIZE)


x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.7, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.33, shuffle=False)

pred = model.predict(x_test).flatten()
print(pred.shape)
print(y_test.shape)

train_result = pd.DataFrame({'Pred': pred, 'Actual': y_test})

plt.plot(train_result['Actual'][-500:], label='Actual')
plt.plot(train_result['Pred'][-500:], label='Predict')
plt.legend()
plt.show()