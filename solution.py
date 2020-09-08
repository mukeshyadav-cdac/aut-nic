import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

print(tf.__version__)
print(keras.__version__)

df = pd.read_csv("./data.csv", header=None)
traget_cat = pd.Categorical(df.iloc[:, -1])
df.iloc[:, -1] = traget_cat.codes

X_VAL = df.values[:, :-1]
X_TAR = df.values[:, -1]

model = Sequential()
model.add(keras.layers.Flatten(input_shape=(35, )))
model.add(keras.layers.Dense(50, activation="relu"))
model.add(keras.layers.Dense(20, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

print(model.summary())

model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_VAL, X_TAR, epochs=10)

model.save('aut_model.h5')

# result = model.predict_classes(X_VAL[4:6])
# print(result)
