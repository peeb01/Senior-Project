import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import load_model

model = load_model('Deep Learning\model\Ann Training.h5')
model.summary()
# feature = pd.read_csv('dataset\X_Testing_data.csv')
# y_label = pd.read_csv('dataset\Y_Testing_data.csv')



