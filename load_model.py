import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

print(tf.__version__)
print(keras.__version__)

df = pd.read_csv(sys.argv[1], header=None)
traget_cat = pd.Categorical(df.iloc[:, -1])
df.iloc[:, -1] = traget_cat.codes

X_VAL = df.values[:, :-1]
X_TAR = df.values[:, -1]

new_model = keras.models.load_model('aut_model.h5')

loss, acc = new_model.evaluate(X_VAL[0:9], X_TAR[0:9])
print("Model, accuracy: {:5.2f}%".format(100*acc))
