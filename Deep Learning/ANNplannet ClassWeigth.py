import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.utils import class_weight



y_label = pd.read_csv('dataset/Plate_dataset_mag405.csv')
Lb = LabelEncoder()
y_labe = Lb.fit_transform(y_label['nearest_cluster'])

x_feature = pd.read_csv('Asteroids Pos CAL/position_plannet.csv')
x = x_feature.iloc[:, 2:].values

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_labe),
    y=y_labe
)


class_weights_dict = dict(enumerate(class_weights))

one_encode = OneHotEncoder()
y = one_encode.fit_transform(y_labe.reshape(-1, 1)).toarray()

stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_index, test_index in stratified_split.split(x, y_label['nearest_cluster']):
    x_train, x_r = x[train_index], x[test_index]
    y_train, y_r = y[train_index], y[test_index]
    y_train_cluster, y_r_cluster = y_label['nearest_cluster'].iloc[train_index], y_label['nearest_cluster'].iloc[test_index]

stratified_split_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.333, random_state=45)

for val_index, test_index in stratified_split_val_test.split(x_r, y_r_cluster):
    x_val, x_test = x_r[val_index], x_r[test_index]
    y_val, y_test = y_r[val_index], y_r[test_index]

print(class_weights_dict)

model = Sequential()
model.add(Dense(units=250, input_shape=(x_train.shape[1],), activation='softmax')) 
model.add(Dense(units=64, activation='softmax')) 
model.add(Dense(units=y_train.shape[1], activation='softmax'))


# model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=3, input_shape=(x_train.shape[1], 1), activation=tf.keras.activations.softmax))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=32, kernel_size=3, activation=tf.keras.activations.softmax))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())

# model.add(Dense(units=64, activation=tf.keras.activations.softmax))
# model.add(Dense(units=32, activation=tf.keras.activations.softmax))
# model.add(Dense(units=y_train.shape[1], activation=tf.keras.activations.softmax))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_val, y_val), callbacks=[early_stopping_callback], class_weight=class_weights_dict)

y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_actual = np.argmax(y_test, axis=1)

model.save('WeightANNPlanets.h5')

from sklearn.metrics import f1_score
f1 = f1_score(y_actual, y_pred, average='macro')
print(f1)

conf_matrix = confusion_matrix(y_actual, y_pred)

plt.figure(figsize=(20, 20))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



# ANN F1-Score 0.016390118832910475
# CNN F1-Score 0.006174966107685057

