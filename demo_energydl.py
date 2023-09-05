import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from energydl import energy_aware_train
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define model
keras.mixed_precision.set_global_policy("float32")
base_model = tf.keras.applications.ResNet50(input_shape=(32,32,3), include_top=False, weights="imagenet")
x = base_model.output
x = keras.layers.Flatten()(x)  # flatten from convolution tensor output
x = keras.layers.Dense(4096, activation='relu')(x) # number of layers and units are hyperparameters, as usual
x = keras.layers.Dense(4096, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
predictions = keras.layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=base_model.input, outputs=predictions)
optimizer = keras.optimizers.Adam(0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

# Define data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
x_train, x_test = x_train / 255, x_test / 255

# Fit the model
energy_aware_train(
    model,
    x_train,
    y_train,
    verbose = 1,
    validation_data = (x_test, y_test),
    energy_importance = 0.3,
    subset=["train"],
    start_predicting = 7
)