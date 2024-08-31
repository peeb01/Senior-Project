import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import f1_score

y_label = pd.read_csv('dataset\Plate_dataset_mag405.csv')
x_feature = pd.read_csv('Asteroids Pos CAL/position_plannet.csv')

x = x_feature.iloc[:, 2:].values

one_encode = OneHotEncoder()
y = one_encode.fit_transform(y_label['nearest_cluster'].values.reshape(-1, 1)).toarray()

# Balance dataset
stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_index, test_index in stratified_split.split(x, y_label['nearest_cluster']):
    x_train, x_r = x[train_index], x[test_index]
    y_train, y_r = y[train_index], y[test_index]
    y_train_cluster, y_r_cluster = y_label['nearest_cluster'].iloc[train_index], y_label['nearest_cluster'].iloc[test_index]

stratified_split_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.333, random_state=45)

for val_index, test_index in stratified_split_val_test.split(x_r, y_r_cluster):
    x_val, x_test = x_r[val_index], x_r[test_index]
    y_val, y_test = y_r[val_index], y_r[test_index]


# pd.DataFrame(y_test).to_csv('Planets_ydata.csv', index=False)
# pd.DataFrame(x_test).to_csv('Planets_xdata.csv', index=False)

# model = Sequential()
# model.add(Dense(units=250, input_shape=(x_train.shape[1],), activation='softmax'))
# model.add(Dense(units=64, activation='softmax'))
# model.add(Dense(units=y_train.shape[1], activation='softmax'))


model = tf.keras.models.Sequential()
model.add(Conv1D(filters=64, kernel_size=3, input_shape=(x_train.shape[1], 1), activation=tf.keras.activations.softmax))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, activation=tf.keras.activations.softmax))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())

model.add(Dense(units=64, activation=tf.keras.activations.softmax))
model.add(Dense(units=500, activation=tf.keras.activations.softmax))
model.add(Dense(units=y_train.shape[1], activation=tf.keras.activations.softmax))


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=['accuracy'])

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_val, y_val), callbacks=[early_stopping_callback])

# model.save('ANNPlanets.h5')

y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_actual = np.argmax(y_test, axis=1)


f1 = f1_score(y_actual, y_pred, average='weighted')
print("F1-Score:", f1)


conf_matrix = confusion_matrix(y_actual, y_pred)

plt.figure(figsize=(20, 20))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



# ANN F1-Score 0.13115822721346582
# CNN F1-Score 0.11598655816231973


