import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# LOAD DATA
data = pd.read_csv('D:/PRAKTIKUM KECERDASAN BUATAN/MODUL 10/emotions.csv')

# TRANSFORM LABEL TO NUMBER
def Transform_data(data):
    encoding_data = {'NEUTRAL': 0, 'POSITIVE': 1, 'NEGATIVE': 2}
    data_encoded = data.replace(encoding_data)
    x = data_encoded.drop(["label"], axis=1)
    y = data_encoded.loc[:, 'label'].values
    scaler = StandardScaler()
    scaler.fit(x)
    X = scaler.transform(x)
    Y = to_categorical(y)
    return X, Y

X, Y = Transform_data(data)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

# CREATE MODEL WITH LSTM
def create_model():
    inputs = tf.keras.Input(shape=(x_train.shape[1],))
    reshape = tf.keras.layers.Reshape((x_train.shape[1], 1))(inputs)
    lstm = tf.keras.layers.LSTM(256, return_sequences=True)(reshape)
    flatten = tf.keras.layers.Flatten()(lstm)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(flatten)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model


lsmtmodel = create_model()
lsmtmodel.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# LEARNING PROCESS
history = lsmtmodel.fit(x_train, y_train, epochs=10, validation_split=0.1)
loss, acc = lsmtmodel.evaluate(x_test, y_test)

print(f"Loss On Testing: {loss*100}", f"\nAccuracy On Training: {acc*100}")

# PREDICTION
pred = lsmtmodel.predict(x_test)
pred1 = np.argmax(pred, axis=1)
y_test1 = np.argmax(y_test, axis=1)
